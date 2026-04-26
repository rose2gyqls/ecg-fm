"""
Phase 3: Masked beat modeling pre-training (DDP-aware).

Single GPU:
    python -m training.pretrain.train --config configs/pretrain/masked_beat_heedb.yaml

Multi-GPU:
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 torchrun --nproc_per_node=7 \
        -m training.pretrain.train --config configs/pretrain/masked_beat_heedb.yaml
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
from training.pretrain.masking import apply_masking, lead_dropout_schedule
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

    if ddp:
        # find_unused_parameters=True covers the case where masking/dropout
        # leaves some head with no contributing positions in a given step.
        model    = DDP(model, device_ids=[local_rank], output_device=local_rank,
                       broadcast_buffers=False, find_unused_parameters=True)
        mlm_head = DDP(mlm_head, device_ids=[local_rank], output_device=local_rank,
                       find_unused_parameters=True)
        rr_head  = DDP(rr_head, device_ids=[local_rank], output_device=local_rank,
                       find_unused_parameters=True)
        fid_head = DDP(fid_head, device_ids=[local_rank], output_device=local_rank,
                       find_unused_parameters=True)
    raw_model    = model.module    if ddp else model
    raw_mlm_head = mlm_head.module if ddp else mlm_head
    raw_rr_head  = rr_head.module  if ddp else rr_head
    raw_fid_head = fid_head.module if ddp else fid_head

    params = (list(model.parameters()) +
              list(mlm_head.parameters()) +
              list(rr_head.parameters()) +
              list(fid_head.parameters()))
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

    # Masking strategy and lead-dropout curriculum.
    mask_strategy = str(mask_cfg.get("mask_strategy", "span"))
    span_length   = int(mask_cfg.get("span_length", 3))
    ld_max_prob   = float(mask_cfg.get("lead_dropout_prob", 0.0))
    ld_schedule   = str(mask_cfg.get("lead_dropout_schedule", "constant"))
    ld_warmup     = int(mask_cfg.get("lead_dropout_warmup_epochs", 0))
    if is_main:
        print(f"[Pretrain] mask_strategy={mask_strategy} span={span_length} "
              f"beat_ratio={mask_cfg.get('beat_mask_ratio')} "
              f"rhythm_ratio={mask_cfg.get('rhythm_mask_ratio')} "
              f"lead_dropout: {ld_schedule} -> {ld_max_prob} (warmup={ld_warmup}ep)")

    # Early stopping (0 disables).
    es_patience = int(cfg["training"].get("early_stop_patience", 0) or 0)
    es_bad = 0

    best_val_loss = float("inf")
    start_epoch   = 1
    global_step   = 0

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
        if "optimizer" in ck:
            opt.load_state_dict(ck["optimizer"])
        if "scheduler" in ck:
            scheduler.load_state_dict(ck["scheduler"])
        start_epoch   = int(ck.get("epoch", 0)) + 1
        best_val_loss = float(ck.get("best_val_loss", ck.get("metric") or float("inf")))
        global_step   = int(ck.get("global_step", 0))
        if is_main:
            print(f"[Resume] Loaded {resume}  -> start_epoch={start_epoch}  "
                  f"best_val_loss={best_val_loss:.4f}", flush=True)

    t_global = time.time()

    def _apply_mask(indices, rr_feats, current_lead_dropout: float):
        return apply_masking(
            indices, rr_feats,
            beat_mask_ratio=float(mask_cfg["beat_mask_ratio"]),
            rhythm_mask_ratio=float(mask_cfg["rhythm_mask_ratio"]),
            span_length=span_length,
            lead_dropout_prob=current_lead_dropout,
            lead_min_leads=int(mask_cfg.get("lead_dropout_min_leads", 1)),
            mask_token_id=mask_token_id,
            mask_strategy=mask_strategy,
        )

    for epoch in range(start_epoch, max_epochs + 1):
        if ddp and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # Lead-dropout curriculum: ramps from 0 to ld_max_prob over ld_warmup epochs.
        cur_lead_dropout = lead_dropout_schedule(
            epoch=epoch,
            max_prob=ld_max_prob,
            schedule=ld_schedule,
            warmup_epochs=ld_warmup,
        )
        if is_main:
            print(f"[Pretrain] epoch {epoch}: lead_dropout_prob={cur_lead_dropout:.3f}", flush=True)

        # ---------- Train ----------
        model.train(); mlm_head.train(); rr_head.train(); fid_head.train()
        t_epoch = time.time()
        running = {"loss": 0.0, "loss_mlm": 0.0, "loss_rr": 0.0,
                   "loss_fid": 0.0, "acc": 0.0}
        n_steps = 0

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

            masked = _apply_mask(indices, rr_feats, cur_lead_dropout)

            stft_in = stft * (~masked["lead_mask"]).to(stft.dtype).view(B, L, 1, 1)

            out = model(masked["masked_indices"], masked["masked_rr_feats"], stft_in)
            token_out = out[:, 1:, :].view(B, N, L, -1)

            beat_mask = masked["beat_mask"]
            if beat_mask.any():
                hidden_masked = token_out[beat_mask]
                logits_mlm = mlm_head(hidden_masked)
                targets    = indices[beat_mask]
                loss_mlm   = F.cross_entropy(logits_mlm, targets)
                with torch.no_grad():
                    acc = (logits_mlm.argmax(-1) == targets).float().mean().item()

                # Fiducial (Q-R, R-S) regression on the same masked positions.
                # Target is z-scored so the MSE has a meaningful magnitude.
                if fid_w > 0:
                    pred_fid = fid_head(hidden_masked)
                    true_fid = _normalize_target(fid_feats[beat_mask], fid_mean_t, fid_std_t)
                    loss_fid = F.mse_loss(pred_fid, true_fid)
                else:
                    loss_fid = torch.tensor(0.0, device=device)
            else:
                loss_mlm = torch.tensor(0.0, device=device)
                loss_fid = torch.tensor(0.0, device=device)
                acc = 0.0

            rr_mask = masked["rhythm_mask"]
            if rr_mask.any():
                pred_rr = rr_head(token_out[rr_mask])
                true_rr = _normalize_target(rr_feats[rr_mask], rr_mean_t, rr_std_t)
                loss_rr = F.mse_loss(pred_rr, true_rr)
            else:
                loss_rr = torch.tensor(0.0, device=device)

            loss = morph_w * loss_mlm + rhythm_w * loss_rr + fid_w * loss_fid

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
                    "acc":      acc,
                }
                for k, v in vals.items():
                    running[k] += v
                n_steps += 1
                global_step += 1
                logger.update(split="train", epoch=epoch, **vals)
                for k, v in vals.items():
                    tb.add_scalar(f"train/{k}", v, global_step)
                if n_steps % 20 == 0:
                    pbar.set_postfix({
                        "loss": f"{running['loss']/n_steps:.3f}",
                        "mlm":  f"{running['loss_mlm']/n_steps:.3f}",
                        "rr":   f"{running['loss_rr']/n_steps:.3f}",
                        "fid":  f"{running['loss_fid']/n_steps:.4f}",
                        "acc":  f"{running['acc']/n_steps:.3f}",
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
            print(
                f"[ep{epoch:03d}/{max_epochs:03d}] "
                f"loss={avg['loss']:.4f}  mlm={avg['loss_mlm']:.4f}  "
                f"rr={avg['loss_rr']:.4f}  fid={avg['loss_fid']:.5f}  "
                f"acc={avg['acc']:.3f}  "
                f"epoch_time={_fmt_dur(elapsed)}  "
                f"elapsed={_fmt_dur(total_elapsed)}  eta={_fmt_dur(eta)}",
                flush=True,
            )

        # ---------- Eval ----------
        if epoch % int(cfg["training"]["eval_every"]) == 0:
            model.eval(); mlm_head.eval(); rr_head.eval(); fid_head.eval()
            local_sums = {"loss": 0.0, "loss_mlm": 0.0, "loss_rr": 0.0,
                          "loss_fid": 0.0, "acc": 0.0}
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
                    masked = _apply_mask(indices, rr_feats, cur_lead_dropout)
                    stft_in = stft * (~masked["lead_mask"]).to(stft.dtype).view(B, L, 1, 1)
                    out = model(masked["masked_indices"], masked["masked_rr_feats"], stft_in)
                    token_out = out[:, 1:, :].view(B, N, L, -1)

                    bm = masked["beat_mask"]
                    if bm.any():
                        hidden_bm = token_out[bm]
                        logits = mlm_head(hidden_bm)
                        tgts   = indices[bm]
                        l_mlm  = F.cross_entropy(logits, tgts).item()
                        accv   = (logits.argmax(-1) == tgts).float().mean().item()
                        if fid_w > 0:
                            tgt_fid = _normalize_target(fid_feats[bm], fid_mean_t, fid_std_t)
                            l_fid = F.mse_loss(fid_head(hidden_bm), tgt_fid).item()
                        else:
                            l_fid = 0.0
                    else:
                        l_mlm, accv, l_fid = 0.0, 0.0, 0.0

                    rmask = masked["rhythm_mask"]
                    if rmask.any():
                        tgt_rr = _normalize_target(rr_feats[rmask], rr_mean_t, rr_std_t)
                        l_rr = F.mse_loss(rr_head(token_out[rmask]), tgt_rr).item()
                    else:
                        l_rr = 0.0

                    total_loss = morph_w * l_mlm + rhythm_w * l_rr + fid_w * l_fid
                    local_sums["loss"]     += total_loss * B
                    local_sums["loss_mlm"] += l_mlm * B
                    local_sums["loss_rr"]  += l_rr * B
                    local_sums["loss_fid"] += l_fid * B
                    local_sums["acc"]      += accv * B
                    local_bs += B

            keys = ["loss", "loss_mlm", "loss_rr", "loss_fid", "acc"]
            stats = torch.tensor(
                [local_sums[k] for k in keys] + [float(local_bs)],
                device=device,
            )
            if ddp:
                dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            cnt = stats[-1].clamp(min=1)
            val_metrics = {k: (stats[i] / cnt).item() for i, k in enumerate(keys)}
            val_loss = val_metrics["loss"]

            if is_main:
                logger.update(split="val", epoch=epoch, **val_metrics)
                for k, v in val_metrics.items():
                    tb.add_scalar(f"val/{k}", v, epoch)
                tag = " *best" if val_loss < best_val_loss else ""
                print(
                    f"          val  loss={val_loss:.4f}  "
                    f"mlm={val_metrics['loss_mlm']:.4f}  "
                    f"rr={val_metrics['loss_rr']:.4f}  "
                    f"fid={val_metrics['loss_fid']:.5f}  "
                    f"acc={val_metrics['acc']:.3f}{tag}",
                    flush=True,
                )

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    es_bad = 0
                    save_checkpoint(
                        raw_model, opt, epoch, val_loss,
                        path=os.path.join(ckpt_dir, "best.pt"),
                        model_cfg=model_cfg,
                        extra={
                            "mlm_head": raw_mlm_head.state_dict(),
                            "rr_head":  raw_rr_head.state_dict(),
                            "fid_head": raw_fid_head.state_dict(),
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
                    "mlm_head": raw_mlm_head.state_dict(),
                    "rr_head":  raw_rr_head.state_dict(),
                    "fid_head": raw_fid_head.state_dict(),
                },
            )

        if is_main:
            save_checkpoint(
                raw_model, opt, epoch, best_val_loss,
                path=os.path.join(ckpt_dir, "last.pt"),
                model_cfg=model_cfg,
                extra={
                    "mlm_head":      raw_mlm_head.state_dict(),
                    "rr_head":       raw_rr_head.state_dict(),
                    "fid_head":      raw_fid_head.state_dict(),
                    "scheduler":     scheduler.state_dict(),
                    "best_val_loss": best_val_loss,
                    "global_step":   global_step,
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
        print(f"[Pretrain] Training complete. Best val_loss={best_val_loss:.4f}")
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
    parser.add_argument("--config", default="configs/pretrain/masked_beat_heedb.yaml")
    parser.add_argument("--resume", default=None,
                        help="Checkpoint path to resume from. "
                             "If omitted, ckpt_dir/last.pt is auto-loaded if present.")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    train(cfg, resume=args.resume)
