"""
Phase 3: Masked beat modeling pre-training (DDP-aware).

Single GPU:
    python -m training.pretrain.train --config configs/pretrain/masked_beat_heedb_cb1024_v4.yaml

Multi-GPU:
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 torchrun --nproc_per_node=7 \
        -m training.pretrain.train --config configs/pretrain/masked_beat_heedb_cb1024_v4.yaml
"""

import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import time
import yaml
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.auto import tqdm

from torch.utils.tensorboard import SummaryWriter

from models.tokenizer.vqvae import VQVAE
from models.transformer.ecg_model import ECGFoundationModel
from models.heads.mlm_head import MaskedBeatModelingHead, MaskedRhythmHead, MaskedFiducialHead
from models.heads.contrastive_head import ProjectionHead, nt_xent_loss
from training.pretrain.masking import apply_masking, lead_dropout_schedule, mask_ratio_schedule
from utils.checkpointing import save_checkpoint
from utils.logging_utils import MetricLogger


def setup_ddp():
    """Initialize DDP from torchrun env. Returns (False, 0, 1, 0) if not launched via torchrun."""
    if "LOCAL_RANK" not in os.environ:
        return False, 0, 1, 0
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    return True, rank, world_size, local_rank


def cleanup_ddp():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _fmt_dur(sec: float) -> str:
    sec = int(sec)
    h, r = divmod(sec, 3600)
    m, s = divmod(r, 60)
    if h: return f"{h}h{m:02d}m"
    if m: return f"{m}m{s:02d}s"
    return f"{s}s"


def train(cfg: dict, resume: str | None = None):
    ddp, rank, world_size, local_rank = setup_ddp()
    is_main = (rank == 0)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    if is_main:
        print(f"[Pretrain] DDP={ddp}  world_size={world_size}  device={device}")

    # ---------- Frozen tokenizer ----------
    tok_ckpt_path = cfg["tokenizer"]["ckpt"]
    tok_ckpt = torch.load(tok_ckpt_path, map_location="cpu")
    tok_model_cfg = _load_tok_cfg(cfg, tok_ckpt)
    tokenizer = VQVAE(tok_model_cfg)
    tokenizer.load_state_dict(tok_ckpt["model"])
    tokenizer.eval().to(device)
    for p in tokenizer.parameters():
        p.requires_grad_(False)
    if is_main:
        print(f"[Pretrain] Tokenizer loaded from {tok_ckpt_path} and frozen.")

    # ---------- ECG-FM + heads ----------
    # Inject tokenizer/data fields that ECGFoundationModel expects.
    model_cfg = dict(cfg["model"])
    model_cfg["codebook_size"] = int(cfg["tokenizer"]["codebook_size"])
    model_cfg["n_leads"]       = int(cfg["data"].get("n_leads", 12))
    model_cfg["max_beats"]     = int(cfg["data"].get("max_beats_per_lead", 15))
    # Persist normalization mode so downstream consumers (ECGFMHBEncoder etc.)
    # can self-describe the input contract from the checkpoint alone.
    model_cfg["normalize"]        = cfg["data"].get("normalize", "record_mad")
    model_cfg["record_mad_scale"] = float(cfg["data"].get("record_mad_scale", 5.0))

    # Record the contrastive head spec so a finetune adapter can re-instantiate
    # the projection head if it wants to (e.g. for similarity-based retrieval).
    _ctr_cfg_persist = cfg.get("training", {}).get("contrastive", {}) or {}
    if float(_ctr_cfg_persist.get("weight", 0.0)) > 0:
        model_cfg["contrastive"] = {
            "proj_hidden": int(_ctr_cfg_persist.get(
                "proj_hidden", int(cfg["model"]["d_model"]))),
            "proj_out":    int(_ctr_cfg_persist.get("proj_out", 128)),
            "temperature": float(_ctr_cfg_persist.get("temperature", 0.1)),
            "weight":      float(_ctr_cfg_persist.get("weight", 0.0)),
        }

    # If the data config provides RR normalization stats, propagate them into
    # context so RhythmMLP registers them as buffers (and downstream encoders
    # automatically inherit the same z-scoring on input).
    norm_cfg = (cfg.get("data", {}) or {}).get("normalization", {}) or {}
    rr_mean = norm_cfg.get("rr_mean")
    rr_std  = norm_cfg.get("rr_std")
    if rr_mean is not None and rr_std is not None:
        ctx = dict(model_cfg.get("context", {}) or {})
        ctx["rhythm_mean"] = list(map(float, rr_mean))
        ctx["rhythm_std"]  = list(map(float, rr_std))
        model_cfg["context"] = ctx
        if is_main:
            print(f"[Pretrain] RhythmMLP normalize: mean={ctx['rhythm_mean']} std={ctx['rhythm_std']}")

    model    = ECGFoundationModel(model_cfg).to(device)
    mlm_head = MaskedBeatModelingHead(
        d_model=int(cfg["model"]["d_model"]),
        codebook_size=int(cfg["tokenizer"]["codebook_size"]),
    ).to(device)
    rr_head  = MaskedRhythmHead(d_model=int(cfg["model"]["d_model"])).to(device)
    fid_head = MaskedFiducialHead(d_model=int(cfg["model"]["d_model"])).to(device)

    # SimCLR-style contrastive auxiliary head. Built unconditionally so the
    # parameter set is identical across configs; the loss is gated below by
    # contrastive_weight (= 0 → no-op, identical to legacy training).
    ctr_cfg = cfg.get("training", {}).get("contrastive", {}) or {}
    ctr_proj_hidden = int(ctr_cfg.get("proj_hidden", int(cfg["model"]["d_model"])))
    ctr_proj_out    = int(ctr_cfg.get("proj_out", 128))
    ctr_temperature = float(ctr_cfg.get("temperature", 0.1))
    ctr_w_max       = float(ctr_cfg.get("weight", 0.0))
    ctr_warmup      = int(ctr_cfg.get("warmup_epochs", 0))
    proj_head = ProjectionHead(
        d_in=int(cfg["model"]["d_model"]),
        d_hidden=ctr_proj_hidden,
        d_out=ctr_proj_out,
    ).to(device)
    if is_main:
        print(f"[Pretrain] contrastive: weight={ctr_w_max} τ={ctr_temperature} "
              f"proj={ctr_proj_hidden}->{ctr_proj_out} warmup={ctr_warmup}ep")

    if ddp:
        # find_unused_parameters=True covers the case where masking/dropout
        # leaves some head with no contributing positions in a given step.
        # Also covers the proj_head being unused when contrastive_weight=0.
        model    = DDP(model, device_ids=[local_rank], output_device=local_rank,
                       broadcast_buffers=False, find_unused_parameters=True)
        mlm_head = DDP(mlm_head, device_ids=[local_rank], output_device=local_rank,
                       find_unused_parameters=True)
        rr_head  = DDP(rr_head, device_ids=[local_rank], output_device=local_rank,
                       find_unused_parameters=True)
        fid_head = DDP(fid_head, device_ids=[local_rank], output_device=local_rank,
                       find_unused_parameters=True)
        proj_head = DDP(proj_head, device_ids=[local_rank], output_device=local_rank,
                        find_unused_parameters=True)
    raw_model     = model.module     if ddp else model
    raw_mlm_head  = mlm_head.module  if ddp else mlm_head
    raw_rr_head   = rr_head.module   if ddp else rr_head
    raw_fid_head  = fid_head.module  if ddp else fid_head
    raw_proj_head = proj_head.module if ddp else proj_head

    params = (list(model.parameters()) +
              list(mlm_head.parameters()) +
              list(rr_head.parameters()) +
              list(fid_head.parameters()) +
              list(proj_head.parameters()))
    if is_main:
        print(f"[Pretrain] Parameters: {sum(p.numel() for p in params):,}")

    # ---------- Data ----------
    source = cfg["data"].get("source", "npy")
    if source == "heedb":
        from data.datasets.heedb_ecg_dataset import HEEDBECGDataset as _DS
    else:
        from data.datasets.ecg_dataset import ECGDataset as _DS
    train_ds = _DS(cfg["data"], split="train")
    val_ds   = _DS(cfg["data"], split="val")

    if ddp:
        train_sampler = DistributedSampler(
            train_ds, num_replicas=world_size, rank=rank,
            shuffle=True, seed=int(cfg["data"].get("seed", 42)),
        )
        val_sampler = DistributedSampler(
            val_ds, num_replicas=world_size, rank=rank, shuffle=False,
        )
    else:
        train_sampler = None
        val_sampler   = None

    nw = int(cfg["training"]["num_workers"])
    pf = int(cfg["training"].get("prefetch_factor", 4))
    loader_kwargs = dict(
        num_workers=nw,
        pin_memory=True,
        persistent_workers=(nw > 0),
        prefetch_factor=(pf if nw > 0 else None),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=False,
        sampler=val_sampler,
        **loader_kwargs,
    )

    # ---------- Optimizer / Scheduler ----------
    opt = AdamW(
        params,
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    max_epochs = int(cfg["training"]["max_epochs"])
    scheduler  = CosineAnnealingLR(opt, T_max=max_epochs)

    mask_cfg = cfg["masking"]
    loss_cfg = cfg["training"]["loss"]
    ckpt_dir = cfg["training"]["ckpt_dir"]
    log_dir  = cfg["training"]["log_dir"]
    # TensorBoard dir — when consolidated under logs/tb/pretrain/, all cb runs
    # can be opened with a single `tensorboard --logdir logs/tb/pretrain`.
    tb_dir   = cfg["training"].get("tb_dir") or os.path.join(log_dir, "tb")
    logger   = MetricLogger(log_dir) if is_main else None
    tb       = SummaryWriter(log_dir=tb_dir) if is_main else None
    if is_main:
        os.makedirs(ckpt_dir, exist_ok=True)
        print(f"[Pretrain] log_dir={log_dir}  tb_dir={tb_dir}")

    mask_token_id = int(cfg["tokenizer"]["codebook_size"])
    morph_w       = float(loss_cfg["morphology_weight"])
    rhythm_w      = float(loss_cfg["rhythm_weight"])
    # Fiducial (Q-R, R-S) regression on masked beat positions. 0 disables it.
    fid_w         = float(loss_cfg.get("fiducial_weight", 0.0))

    # Target z-score stats: regression heads are trained against normalized
    # targets so the MSE has a meaningful magnitude (raw RR/fid values are
    # tiny numbers in seconds and would otherwise produce a ~0 loss).
    def _to_tensor(x):
        return None if x is None else torch.tensor(x, dtype=torch.float32, device=device)

    rr_mean_t  = _to_tensor(rr_mean)
    rr_std_t   = _to_tensor(rr_std)
    fid_mean_t = _to_tensor(norm_cfg.get("fid_mean"))
    fid_std_t  = _to_tensor(norm_cfg.get("fid_std"))
    if is_main and (rr_mean_t is not None or fid_mean_t is not None):
        print(f"[Pretrain] target z-score: rr={'on' if rr_mean_t is not None else 'off'}  "
              f"fid={'on' if fid_mean_t is not None else 'off'}")

    def _normalize_target(t, mean, std):
        return t if mean is None or std is None else (t - mean) / (std + 1e-8)

    # Masking strategy and curricula. beat_mask_ratio / rhythm_mask_ratio can
    # be ramped up over the first `mask_warmup_epochs` from `mask_start_ratio`
    # to their max — gives the model an easier task early so it can establish
    # a useful representation before the harder MAE-style mask kicks in.
    mask_strategy = str(mask_cfg.get("mask_strategy", "span"))
    span_length   = int(mask_cfg.get("span_length", 3))

    beat_mask_max     = float(mask_cfg.get("beat_mask_ratio", 0.5))
    rhythm_mask_max   = float(mask_cfg.get("rhythm_mask_ratio", 0.5))
    mask_warmup_ep    = int(mask_cfg.get("mask_warmup_epochs", 0))
    mask_warmup_sched = str(mask_cfg.get("mask_warmup_schedule", "linear"))
    mask_start_ratio  = float(mask_cfg.get("mask_start_ratio", 0.15))

    ld_max_prob   = float(mask_cfg.get("lead_dropout_prob", 0.0))
    ld_schedule   = str(mask_cfg.get("lead_dropout_schedule", "constant"))
    ld_warmup     = int(mask_cfg.get("lead_dropout_warmup_epochs", 0))
    if is_main:
        print(f"[Pretrain] mask_strategy={mask_strategy} span={span_length} "
              f"beat_ratio: {mask_warmup_sched} {mask_start_ratio}->{beat_mask_max} "
              f"(warmup={mask_warmup_ep}ep) "
              f"rhythm_ratio_max={rhythm_mask_max} "
              f"lead_dropout: {ld_schedule} {ld_max_prob} (warmup={ld_warmup}ep)")

    # Early stopping (0 disables). The metric is configurable:
    #   "val_loss"        — lower is better (legacy default)
    #   "val_acc_nontop"  — higher is better. Tracks MLM accuracy on tokens
    #                       OTHER than the dominant codebook token. Catches
    #                       the v2 failure mode where val_loss falls while
    #                       the model collapses to predicting the top-1 code.
    es_patience = int(cfg["training"].get("early_stop_patience", 0) or 0)
    es_metric   = str(cfg["training"].get("early_stop_metric", "val_loss"))
    if es_metric not in ("val_loss", "val_acc_nontop"):
        raise ValueError(f"unknown early_stop_metric: {es_metric}")
    es_higher_better = (es_metric == "val_acc_nontop")
    es_bad = 0
    if is_main:
        print(f"[Pretrain] early-stop metric={es_metric} "
              f"({'higher' if es_higher_better else 'lower'} is better) "
              f"patience={es_patience}")

    # Tracking variables. Keep `best_val_loss` for backward compat (saved on
    # last.pt and used by callers that grep our checkpoint structure), but
    # gate save/early-stop on the chosen metric `best_metric`.
    best_val_loss = float("inf")
    best_metric   = float("-inf") if es_higher_better else float("inf")
    start_epoch   = 1
    global_step   = 0

    def _is_better(new: float, ref: float) -> bool:
        return (new > ref) if es_higher_better else (new < ref)

    # ---------- Resume ----------
    if resume is None:
        last_path = os.path.join(ckpt_dir, "last.pt")
        if os.path.exists(last_path):
            resume = last_path
    if resume and os.path.exists(resume):
        ck = torch.load(resume, map_location=device)
        raw_model.load_state_dict(ck["model"])
        if "mlm_head" in ck:
            raw_mlm_head.load_state_dict(ck["mlm_head"])
        if "rr_head" in ck:
            raw_rr_head.load_state_dict(ck["rr_head"])
        if "fid_head" in ck:
            raw_fid_head.load_state_dict(ck["fid_head"])
        if "proj_head" in ck:
            raw_proj_head.load_state_dict(ck["proj_head"])
        if "optimizer" in ck:
            opt.load_state_dict(ck["optimizer"])
        if "scheduler" in ck:
            scheduler.load_state_dict(ck["scheduler"])
        start_epoch   = int(ck.get("epoch", 0)) + 1
        best_val_loss = float(ck.get("best_val_loss", ck.get("metric") or float("inf")))
        # Restore best_metric if previously saved under same metric, else fall
        # back to (best_val_loss for val_loss / -inf for higher-is-better).
        if "best_metric" in ck and ck.get("best_metric_name") == es_metric:
            best_metric = float(ck["best_metric"])
        elif es_metric == "val_loss":
            best_metric = best_val_loss
        global_step   = int(ck.get("global_step", 0))
        if is_main:
            print(f"[Resume] Loaded {resume}  -> start_epoch={start_epoch}  "
                  f"best_val_loss={best_val_loss:.4f}  "
                  f"best_{es_metric}={best_metric:.4f}", flush=True)

    # ---------- Class-balanced CE weights (Phase E) ----------
    # Tokenizer codebooks are heavily skewed (top-1 code can hold 60%+ of all
    # tokens), so plain CE collapses to "predict the dominant code". Inverse-
    # frequency weighting (freq^-alpha) raises the gradient on rare codes.
    cb_alpha = float(loss_cfg.get("class_balanced_alpha", 0.0))
    K = mask_token_id  # = codebook_size
    ce_weight = None
    top1_token_id = None
    if cb_alpha > 0:
        freq_path = os.path.join(ckpt_dir, f"codebook_freq_K{K}.pt")
        if os.path.exists(freq_path):
            freq = torch.load(freq_path, map_location=device)["freq"].to(device)
            if is_main:
                print(f"[Pretrain] codebook freq loaded from {freq_path}")
        else:
            if is_main:
                print(f"[Pretrain] computing codebook freq on training subset...")
            counts = torch.zeros(K, dtype=torch.long, device=device)
            target_records = 2000
            per_rank_target = max(target_records // max(world_size, 1), 64)
            seen = 0
            for batch in train_loader:
                beats = batch["beats"].to(device, non_blocking=True)
                Bb, Nb, Lb, Wb = beats.shape
                with torch.no_grad():
                    _, idx_flat = tokenizer.encode(beats.view(Bb * Nb * Lb, 1, Wb))
                counts += torch.bincount(idx_flat, minlength=K)
                seen += Bb
                if seen >= per_rank_target:
                    break
            if ddp:
                dist.all_reduce(counts, op=dist.ReduceOp.SUM)
                seen_t = torch.tensor([seen], device=device)
                dist.all_reduce(seen_t, op=dist.ReduceOp.SUM)
                seen_total = int(seen_t.item())
            else:
                seen_total = seen
            freq = counts.float() / counts.sum().clamp(min=1)
            if is_main:
                torch.save({"freq": freq.cpu(), "n_records": seen_total}, freq_path)
                print(f"[Pretrain] codebook freq saved → {freq_path}  "
                      f"(records={seen_total}, top-1={freq.max().item():.3f})")
            if ddp:
                dist.barrier()
        ce_weight = (freq + 1e-8).pow(-cb_alpha)
        ce_weight = ce_weight / ce_weight.mean()
        # Clamp to prevent rare-code gradient explosion: a freq~0 code yields
        # huge raw weight, and a single sample of it can dominate a batch.
        clamp_max = float(loss_cfg.get("class_balanced_clamp_max", 10.0))
        clamp_min = float(loss_cfg.get("class_balanced_clamp_min", 0.01))
        ce_weight = ce_weight.clamp(min=clamp_min, max=clamp_max)
        ce_weight = ce_weight / ce_weight.mean()
        top1_token_id = int(freq.argmax().item())
        if is_main:
            print(f"[Pretrain] class-balanced CE: alpha={cb_alpha}  "
                  f"weight=[{ce_weight.min().item():.3f}, {ce_weight.max().item():.3f}]  "
                  f"clamp=[{clamp_min}, {clamp_max}]  "
                  f"top-1 code id={top1_token_id} ({freq.max().item()*100:.1f}%)")

    t_global = time.time()

    def _apply_mask(indices, rr_feats, current_lead_dropout: float,
                    current_beat_ratio: float, current_rhythm_ratio: float):
        return apply_masking(
            indices, rr_feats,
            beat_mask_ratio=current_beat_ratio,
            rhythm_mask_ratio=current_rhythm_ratio,
            span_length=span_length,
            lead_dropout_prob=current_lead_dropout,
            lead_min_leads=int(mask_cfg.get("lead_dropout_min_leads", 1)),
            mask_token_id=mask_token_id,
            mask_strategy=mask_strategy,
        )

    def _compute_view(masked, indices, rr_feats, fid_feats, stft, B, N, L):
        """One forward + MLM/RR/fid losses + CLS for one masked view.

        Returns dict with float tensors for losses and the (B, d) CLS output.
        Diagnostic acc / acc_nontop are returned as Python floats.
        """
        stft_in = stft * (~masked["lead_mask"]).to(stft.dtype).view(B, L, 1, 1)
        out = model(masked["masked_indices"], masked["masked_rr_feats"], stft_in)
        cls       = out[:, 0, :]
        token_out = out[:, 1:, :].view(B, N, L, -1)

        beat_mask = masked["beat_mask"]
        if beat_mask.any():
            hidden_masked = token_out[beat_mask]
            logits_mlm = mlm_head(hidden_masked)
            targets    = indices[beat_mask]
            loss_mlm   = F.cross_entropy(logits_mlm, targets, weight=ce_weight)
            with torch.no_grad():
                pred = logits_mlm.argmax(-1)
                acc = (pred == targets).float().mean().item()
                if top1_token_id is not None:
                    nt = targets != top1_token_id
                    acc_nontop = (pred[nt] == targets[nt]).float().mean().item() if nt.any() else 0.0
                else:
                    acc_nontop = acc
            if fid_w > 0:
                pred_fid = fid_head(hidden_masked)
                true_fid = _normalize_target(fid_feats[beat_mask], fid_mean_t, fid_std_t)
                loss_fid = F.mse_loss(pred_fid, true_fid)
            else:
                loss_fid = torch.tensor(0.0, device=device)
        else:
            loss_mlm = torch.tensor(0.0, device=device)
            loss_fid = torch.tensor(0.0, device=device)
            acc, acc_nontop = 0.0, 0.0

        rr_mask = masked["rhythm_mask"]
        if rr_mask.any():
            pred_rr = rr_head(token_out[rr_mask])
            true_rr = _normalize_target(rr_feats[rr_mask], rr_mean_t, rr_std_t)
            loss_rr = F.mse_loss(pred_rr, true_rr)
        else:
            loss_rr = torch.tensor(0.0, device=device)

        return {
            "cls":       cls,
            "loss_mlm":  loss_mlm,
            "loss_rr":   loss_rr,
            "loss_fid":  loss_fid,
            "acc":       acc,
            "acc_nontop": acc_nontop,
        }

    def _contrastive_weight(epoch: int) -> float:
        """Linear warmup of contrastive weight: 0 at ep1, ctr_w_max at warmup+1."""
        if ctr_w_max <= 0 or ctr_warmup <= 0:
            return float(ctr_w_max)
        t = max(0, epoch - 1)
        if t >= ctr_warmup:
            return float(ctr_w_max)
        return float(ctr_w_max) * (t / ctr_warmup)

    two_view = (ctr_w_max > 0)
    if is_main and two_view:
        print("[Pretrain] two-view forward enabled (contrastive auxiliary loss active)")

    for epoch in range(start_epoch, max_epochs + 1):
        if ddp and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # Curricula for this epoch. Lead dropout ramps 0 → ld_max over ld_warmup.
        # Beat / rhythm mask ratios ramp mask_start → max over mask_warmup_ep.
        cur_lead_dropout = lead_dropout_schedule(
            epoch=epoch,
            max_prob=ld_max_prob,
            schedule=ld_schedule,
            warmup_epochs=ld_warmup,
        )
        cur_beat_ratio = mask_ratio_schedule(
            epoch=epoch, max_ratio=beat_mask_max,
            schedule=mask_warmup_sched, warmup_epochs=mask_warmup_ep,
            start_ratio=mask_start_ratio,
        )
        cur_rhythm_ratio = mask_ratio_schedule(
            epoch=epoch, max_ratio=rhythm_mask_max,
            schedule=mask_warmup_sched, warmup_epochs=mask_warmup_ep,
            start_ratio=mask_start_ratio,
        )
        if is_main:
            print(f"[Pretrain] epoch {epoch}: "
                  f"beat_mask={cur_beat_ratio:.3f}  rhythm_mask={cur_rhythm_ratio:.3f}  "
                  f"lead_dropout={cur_lead_dropout:.3f}", flush=True)

        # ---------- Train ----------
        model.train(); mlm_head.train(); rr_head.train(); fid_head.train()
        proj_head.train()
        t_epoch = time.time()
        running = {"loss": 0.0, "loss_mlm": 0.0, "loss_rr": 0.0,
                   "loss_fid": 0.0, "loss_ctr": 0.0,
                   "acc": 0.0, "acc_nontop": 0.0,
                   "ctr_pos_sim": 0.0, "ctr_neg_sim": 0.0, "ctr_acc": 0.0}
        n_steps = 0
        cur_ctr_w = _contrastive_weight(epoch)

        pbar = tqdm(
            train_loader,
            desc=f"ep{epoch:03d}/{max_epochs:03d}",
            disable=(not is_main),
            dynamic_ncols=True, mininterval=1.0, leave=False,
        )

        for batch in pbar:
            beats    = batch["beats"].to(device, non_blocking=True)
            rr_feats = batch["rr_feats"].to(device, non_blocking=True)
            stft     = batch["stft"].to(device, non_blocking=True)
            # fid_feats is only emitted by HEEDBECGDataset; zero-fill for the
            # plain ECGDataset path (the loss is also gated on fid_w > 0).
            if "fid_feats" in batch:
                fid_feats = batch["fid_feats"].to(device, non_blocking=True)
            else:
                fid_feats = torch.zeros(*rr_feats.shape[:3], 2, device=device)

            B, N, L, W = beats.shape
            with torch.no_grad():
                _, idx_flat = tokenizer.encode(beats.view(B * N * L, 1, W))
            indices = idx_flat.view(B, N, L)

            # ── View 1 (always) ────────────────────────────────────────────
            masked1 = _apply_mask(indices, rr_feats, cur_lead_dropout,
                                  cur_beat_ratio, cur_rhythm_ratio)
            v1 = _compute_view(masked1, indices, rr_feats, fid_feats, stft, B, N, L)

            if two_view:
                # ── View 2: independent masking on the same record ─────────
                masked2 = _apply_mask(indices, rr_feats, cur_lead_dropout,
                                      cur_beat_ratio, cur_rhythm_ratio)
                v2 = _compute_view(masked2, indices, rr_feats, fid_feats, stft, B, N, L)

                loss_mlm = 0.5 * (v1["loss_mlm"] + v2["loss_mlm"])
                loss_rr  = 0.5 * (v1["loss_rr"]  + v2["loss_rr"])
                loss_fid = 0.5 * (v1["loss_fid"] + v2["loss_fid"])
                acc        = 0.5 * (v1["acc"]        + v2["acc"])
                acc_nontop = 0.5 * (v1["acc_nontop"] + v2["acc_nontop"])

                # SimCLR-style NT-Xent on projected CLS embeddings.
                z1 = proj_head(v1["cls"])
                z2 = proj_head(v2["cls"])
                loss_ctr, ctr_stats = nt_xent_loss(
                    z1, z2,
                    temperature=ctr_temperature,
                    gather_distributed=ddp,
                )
            else:
                loss_mlm = v1["loss_mlm"]; loss_rr = v1["loss_rr"]; loss_fid = v1["loss_fid"]
                acc = v1["acc"]; acc_nontop = v1["acc_nontop"]
                loss_ctr = torch.tensor(0.0, device=device)
                ctr_stats = {"ctr_pos_sim": 0.0, "ctr_neg_sim": 0.0, "ctr_acc": 0.0}

            loss = (morph_w   * loss_mlm
                    + rhythm_w * loss_rr
                    + fid_w    * loss_fid
                    + cur_ctr_w * loss_ctr)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, float(cfg["training"]["grad_clip"]))
            opt.step()

            if is_main:
                vals = {
                    "loss":     loss.item(),
                    "loss_mlm": loss_mlm.item(),
                    "loss_rr":  loss_rr.item(),
                    "loss_fid": loss_fid.item(),
                    "loss_ctr": loss_ctr.item(),
                    "acc":      acc,
                    "acc_nontop": acc_nontop,
                    "ctr_pos_sim": ctr_stats["ctr_pos_sim"],
                    "ctr_neg_sim": ctr_stats["ctr_neg_sim"],
                    "ctr_acc":     ctr_stats["ctr_acc"],
                }
                for k, v in vals.items():
                    running[k] += v
                n_steps += 1
                global_step += 1
                logger.update(split="train", epoch=epoch, **vals)
                for k, v in vals.items():
                    tb.add_scalar(f"train/{k}", v, global_step)
                tb.add_scalar("train/ctr_weight", cur_ctr_w, global_step)
                tb.add_scalar("train/beat_mask_ratio", cur_beat_ratio, global_step)
                tb.add_scalar("train/rhythm_mask_ratio", cur_rhythm_ratio, global_step)
                tb.add_scalar("train/lead_dropout_prob", cur_lead_dropout, global_step)
                if n_steps % 20 == 0:
                    pbar.set_postfix({
                        "loss": f"{running['loss']/n_steps:.3f}",
                        "mlm":  f"{running['loss_mlm']/n_steps:.3f}",
                        "rr":   f"{running['loss_rr']/n_steps:.3f}",
                        "fid":  f"{running['loss_fid']/n_steps:.4f}",
                        "ctr":  f"{running['loss_ctr']/n_steps:.3f}",
                        "acc":  f"{running['acc']/n_steps:.3f}",
                        "acc_nt": f"{running['acc_nontop']/n_steps:.3f}",
                        "c_acc": f"{running['ctr_acc']/n_steps:.3f}",
                    })
        pbar.close()
        scheduler.step()

        # ---------- Epoch summary (rank 0) ----------
        if is_main and n_steps > 0:
            avg = {k: v / n_steps for k, v in running.items()}
            for k, v in avg.items():
                tb.add_scalar(f"train_epoch/{k}", v, epoch)
            tb.add_scalar("lr", scheduler.get_last_lr()[0], epoch)
            elapsed       = time.time() - t_epoch
            total_elapsed = time.time() - t_global
            eta           = elapsed * (max_epochs - epoch)
            ctr_str = (f"  ctr={avg['loss_ctr']:.3f}(w={cur_ctr_w:.2f}) "
                       f"c_acc={avg['ctr_acc']:.3f}") if two_view else ""
            print(
                f"[ep{epoch:03d}/{max_epochs:03d}] "
                f"loss={avg['loss']:.4f}  mlm={avg['loss_mlm']:.4f}  "
                f"rr={avg['loss_rr']:.4f}  fid={avg['loss_fid']:.5f}{ctr_str}  "
                f"acc={avg['acc']:.3f}  acc_nt={avg['acc_nontop']:.3f}  "
                f"epoch_time={_fmt_dur(elapsed)}  "
                f"elapsed={_fmt_dur(total_elapsed)}  eta={_fmt_dur(eta)}",
                flush=True,
            )

        # ---------- Eval ----------
        if epoch % int(cfg["training"]["eval_every"]) == 0:
            model.eval(); mlm_head.eval(); rr_head.eval(); fid_head.eval()
            proj_head.eval()
            local_sums = {"loss": 0.0, "loss_mlm": 0.0, "loss_rr": 0.0,
                          "loss_fid": 0.0, "acc": 0.0, "acc_nontop": 0.0}
            local_bs = 0
            with torch.no_grad():
                for batch in val_loader:
                    beats    = batch["beats"].to(device, non_blocking=True)
                    rr_feats = batch["rr_feats"].to(device, non_blocking=True)
                    stft     = batch["stft"].to(device, non_blocking=True)
                    if "fid_feats" in batch:
                        fid_feats = batch["fid_feats"].to(device, non_blocking=True)
                    else:
                        fid_feats = torch.zeros(*rr_feats.shape[:3], 2, device=device)
                    B, N, L, W = beats.shape
                    _, idx_flat = tokenizer.encode(beats.view(B * N * L, 1, W))
                    indices = idx_flat.view(B, N, L)
                    masked = _apply_mask(indices, rr_feats, cur_lead_dropout,
                                         cur_beat_ratio, cur_rhythm_ratio)
                    stft_in = stft * (~masked["lead_mask"]).to(stft.dtype).view(B, L, 1, 1)
                    out = model(masked["masked_indices"], masked["masked_rr_feats"], stft_in)
                    token_out = out[:, 1:, :].view(B, N, L, -1)

                    bm = masked["beat_mask"]
                    if bm.any():
                        hidden_bm = token_out[bm]
                        logits = mlm_head(hidden_bm)
                        tgts   = indices[bm]
                        l_mlm  = F.cross_entropy(logits, tgts, weight=ce_weight).item()
                        pred = logits.argmax(-1)
                        accv = (pred == tgts).float().mean().item()
                        if top1_token_id is not None:
                            nt_v = tgts != top1_token_id
                            accv_nt = (pred[nt_v] == tgts[nt_v]).float().mean().item() if nt_v.any() else 0.0
                        else:
                            accv_nt = accv
                        if fid_w > 0:
                            tgt_fid = _normalize_target(fid_feats[bm], fid_mean_t, fid_std_t)
                            l_fid = F.mse_loss(fid_head(hidden_bm), tgt_fid).item()
                        else:
                            l_fid = 0.0
                    else:
                        l_mlm, accv, accv_nt, l_fid = 0.0, 0.0, 0.0, 0.0

                    rmask = masked["rhythm_mask"]
                    if rmask.any():
                        tgt_rr = _normalize_target(rr_feats[rmask], rr_mean_t, rr_std_t)
                        l_rr = F.mse_loss(rr_head(token_out[rmask]), tgt_rr).item()
                    else:
                        l_rr = 0.0

                    total_loss = morph_w * l_mlm + rhythm_w * l_rr + fid_w * l_fid
                    local_sums["loss"]       += total_loss * B
                    local_sums["loss_mlm"]   += l_mlm * B
                    local_sums["loss_rr"]    += l_rr * B
                    local_sums["loss_fid"]   += l_fid * B
                    local_sums["acc"]        += accv * B
                    local_sums["acc_nontop"] += accv_nt * B
                    local_bs += B

            keys = ["loss", "loss_mlm", "loss_rr", "loss_fid", "acc", "acc_nontop"]
            stats = torch.tensor(
                [local_sums[k] for k in keys] + [float(local_bs)],
                device=device,
            )
            if ddp:
                dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            cnt = stats[-1].clamp(min=1)
            val_metrics = {k: (stats[i] / cnt).item() for i, k in enumerate(keys)}
            val_loss = val_metrics["loss"]

            # The metric we early-stop on (and use to pick "best.pt").
            cur_metric = (val_metrics["acc_nontop"]
                          if es_metric == "val_acc_nontop"
                          else val_loss)
            improved = _is_better(cur_metric, best_metric)

            if is_main:
                logger.update(split="val", epoch=epoch, **val_metrics)
                for k, v in val_metrics.items():
                    tb.add_scalar(f"val/{k}", v, epoch)
                tag = " *best" if improved else ""
                print(
                    f"          val  loss={val_loss:.4f}  "
                    f"mlm={val_metrics['loss_mlm']:.4f}  "
                    f"rr={val_metrics['loss_rr']:.4f}  "
                    f"fid={val_metrics['loss_fid']:.5f}  "
                    f"acc={val_metrics['acc']:.3f}  "
                    f"acc_nt={val_metrics['acc_nontop']:.3f}  "
                    f"[best by {es_metric}={cur_metric:.4f}]{tag}",
                    flush=True,
                )

                # Track val_loss continuously (used as a fallback metric on
                # checkpoints), but base save/early-stop on `best_metric`.
                if val_loss < best_val_loss:
                    best_val_loss = val_loss

                if improved:
                    best_metric = cur_metric
                    es_bad = 0
                    save_checkpoint(
                        raw_model, opt, epoch, cur_metric,
                        path=os.path.join(ckpt_dir, "best.pt"),
                        model_cfg=model_cfg,
                        extra={
                            "mlm_head":         raw_mlm_head.state_dict(),
                            "rr_head":          raw_rr_head.state_dict(),
                            "fid_head":         raw_fid_head.state_dict(),
                            "proj_head":        raw_proj_head.state_dict(),
                            "best_metric_name": es_metric,
                            "best_metric":      best_metric,
                            "best_val_loss":    best_val_loss,
                        },
                    )
                else:
                    es_bad += 1

        # ---------- Periodic + last checkpoint ----------
        if epoch % int(cfg["training"]["save_every"]) == 0 and is_main:
            save_checkpoint(
                raw_model, opt, epoch, None,
                path=os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pt"),
                model_cfg=model_cfg,
                extra={
                    "mlm_head":  raw_mlm_head.state_dict(),
                    "rr_head":   raw_rr_head.state_dict(),
                    "fid_head":  raw_fid_head.state_dict(),
                    "proj_head": raw_proj_head.state_dict(),
                },
            )

        if is_main:
            save_checkpoint(
                raw_model, opt, epoch, best_val_loss,
                path=os.path.join(ckpt_dir, "last.pt"),
                model_cfg=model_cfg,
                extra={
                    "mlm_head":         raw_mlm_head.state_dict(),
                    "rr_head":          raw_rr_head.state_dict(),
                    "fid_head":         raw_fid_head.state_dict(),
                    "proj_head":        raw_proj_head.state_dict(),
                    "scheduler":        scheduler.state_dict(),
                    "best_val_loss":    best_val_loss,
                    "best_metric":      best_metric,
                    "best_metric_name": es_metric,
                    "global_step":      global_step,
                },
            )

        if ddp:
            dist.barrier()

        # ---------- Early stopping (only on epochs that ran eval) ----------
        evaled = (epoch % int(cfg["training"]["eval_every"]) == 0)
        if es_patience > 0 and evaled:
            stop_flag = 0
            if is_main and es_bad >= es_patience:
                stop_flag = 1
                print(f"[EarlyStop] no val_loss improvement for {es_patience} evals; "
                      f"stopping at epoch {epoch}.", flush=True)
            if ddp:
                t = torch.tensor(stop_flag, device=device)
                dist.broadcast(t, src=0)
                stop_flag = int(t.item())
            if stop_flag:
                break

    if is_main:
        print(f"[Pretrain] Training complete. "
              f"Best {es_metric}={best_metric:.4f}  best_val_loss={best_val_loss:.4f}")
        if tb is not None:
            tb.close()
    cleanup_ddp()


def _load_tok_cfg(cfg, ckpt=None):
    """
    Resolve the tokenizer model_cfg with this priority:
      1) ckpt["model_cfg"] (the v2 path always populates this).
      2) cfg["tokenizer"]["model_cfg_yaml"] -> external YAML.
      3) cfg["tokenizer"]["model"] inline dict.
    """
    if ckpt is None:
        ckpt = torch.load(cfg["tokenizer"]["ckpt"], map_location="cpu")
    if "model_cfg" in ckpt and ckpt["model_cfg"]:
        return ckpt["model_cfg"]
    tok = cfg.get("tokenizer", {})
    if tok.get("model_cfg_yaml"):
        with open(tok["model_cfg_yaml"]) as f:
            return yaml.safe_load(f)["model"]
    if "model" in tok:
        return tok["model"]
    raise ValueError("Tokenizer model cfg not found.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pretrain/masked_beat_heedb_cb1024_v4.yaml")
    parser.add_argument("--resume", default=None,
                        help="Checkpoint path to resume from. "
                             "If omitted, ckpt_dir/last.pt is auto-loaded if present.")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    train(cfg, resume=args.resume)
