"""Post-hoc tokenizer checkpoint evaluation.

Evaluates every saved checkpoint in a given ckpt dir on a *fixed, cached* val
subset, so all ckpts see the exact same beats. This addresses the case where
best.pt was selected early due to sliding-window val_loss noise, and a later
epoch_N.pt may actually have better reconstruction or codebook quality.

Reports rec / vq / fid / spec / ent / ppl / coverage per ckpt.
With --lead-analysis: per-lead JS distance + V1 vs V6 distinct top-1 count.

Usage:
    python -m scripts.eval_tokenizer_checkpoints \
        --config configs/tokenizer/vqvae_heedb_full_cb512_v4.yaml \
        --ckpt-dir checkpoints/tokenizer_heedb_full_cb512_v4 \
        --val-records 2000 --gpu 0 [--lead-analysis]
"""
from __future__ import annotations
import argparse, glob, json, os, sys, time
import numpy as np
import torch, yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from data.datasets.heedb_beat_dataset import HEEDBBeatDataset
from models.tokenizer.vqvae import VQVAE
from training.tokenizer.losses import total_vqvae_loss


@torch.no_grad()
def evaluate_one(model, val_loader, device, *,
                 alpha, beta, gamma, delta, spec_n_ffts, use_grad_loss):
    model.eval()
    K = model.codebook.K
    sums = {"loss": 0.0, "rec": 0.0, "vq": 0.0, "fid": 0.0,
            "spec": 0.0, "ent": 0.0}
    cnt = 0
    use_counts = torch.zeros(K, dtype=torch.long)
    for batch in val_loader:
        x = batch["beat"].to(device, non_blocking=True)
        x_hat, vq_dict = model(x)
        losses = total_vqvae_loss(
            x, x_hat, vq_dict["loss_vq"],
            alpha=alpha, beta=beta, gamma=gamma, delta=delta,
            neg_entropy=vq_dict.get("neg_entropy"),
            use_gradient_loss=use_grad_loss,
            spec_n_ffts=spec_n_ffts,
        )
        bs = x.size(0)
        sums["loss"] += losses["loss"].item() * bs
        sums["rec"]  += losses["loss_rec"].item() * bs
        sums["vq"]   += losses["loss_vq"].item() * bs
        sums["fid"]  += losses["loss_fid"].item() * bs
        sums["spec"] += losses["loss_spec"].item() * bs
        sums["ent"]  += losses["loss_ent"].item() * bs
        use_counts += torch.bincount(vq_dict["indices"].cpu(), minlength=K)
        cnt += bs
    avg = {k: v / max(cnt, 1) for k, v in sums.items()}
    avg["n_samples"] = cnt
    p = use_counts.float() / max(cnt, 1)
    nz = p[p > 0]
    avg["perplexity"] = float(torch.exp(-(nz * nz.log()).sum())) if nz.numel() else 0.0
    avg["active_codes"] = int((use_counts > 0).sum())
    avg["dead_codes"] = K - avg["active_codes"]
    avg["coverage_pct"] = avg["active_codes"] / K * 100.0
    return avg


@torch.no_grad()
def per_lead_metrics(model, cfg, device, n_records=200, seed=42):
    from data.preprocessing.heedb_io import (
        load_heedb_record, align_to_heedb_order, HEEDB_LEAD_ORDER,
    )
    from data.preprocessing.beat_segmentor import (
        detect_rpeaks, extract_beats, LEAD_II_INDEX,
    )
    from data.preprocessing.resampler import (
        resample_signal, resample_beat,
        compute_record_norm_stats, apply_record_norm,
    )
    LEADS = HEEDB_LEAD_ORDER
    K = model.codebook.K
    NUM_LEADS = 12
    counts = torch.zeros(NUM_LEADS, K, dtype=torch.long)
    rng = np.random.RandomState(seed)
    with open(cfg["data"]["val_list"]) as f:
        files = [l.strip() for l in f if l.strip()]
    sel = rng.choice(len(files), min(n_records, len(files)), replace=False)
    fs = int(cfg["data"].get("target_fs", 500))
    beat_len = int(cfg["data"].get("beat_length", 256))
    before_ms = int(cfg["data"].get("before_ms", 200))
    after_ms = int(cfg["data"].get("after_ms", 400))
    norm_mode = cfg["data"].get("normalize", "record_mad")
    rm_scale = float(cfg["data"].get("record_mad_scale", 5.0))
    rm_min = float(cfg["data"].get("record_mad_min_scale", 0.05))
    rm_clip = cfg["data"].get("record_mad_clip", None)
    if rm_clip is not None:
        rm_clip = float(rm_clip)
    used = 0
    for i in sel:
        rec = load_heedb_record(files[int(i)], load_rpeaks=False)
        if rec is None: continue
        sig = align_to_heedb_order(rec["signal"], rec["sig_name"])
        if sig is None: continue
        if rec["fs"] != fs:
            sig = resample_signal(sig, rec["fs"], fs)
        try:
            rpk = detect_rpeaks(sig[LEAD_II_INDEX], fs, method="neurokit")
        except Exception:
            continue
        if len(rpk) < 2: continue
        beats = extract_beats(sig, rpk, fs, before_ms=before_ms, after_ms=after_ms)
        if len(beats) == 0: continue
        beats = np.stack(beats, axis=0)
        N, L, W = beats.shape
        flat = resample_beat(beats.reshape(N * L, W), beat_len)
        if norm_mode == "record_mad":
            med, mad = compute_record_norm_stats(sig, min_scale=rm_min)
            flat = apply_record_norm(flat, med, mad, scale=rm_scale, clip=rm_clip)
        arr = flat.reshape(N, L, beat_len).astype(np.float32)
        x = torch.from_numpy(arr).to(device).reshape(N * L, 1, beat_len)
        z = model.encoder(x)
        _, idx, *_ = model.codebook(z)
        idx = idx.view(N, L).cpu()
        for li in range(L):
            counts[li] += torch.bincount(idx[:, li], minlength=K)
        used += 1
    eps = 1e-12
    P = counts.float()
    P = P / P.sum(1, keepdim=True).clamp(min=1)

    def kl(a, b):
        return (a * (a.clamp(min=eps).log() - b.clamp(min=eps).log())).sum(-1)

    js = torch.zeros(NUM_LEADS, NUM_LEADS)
    for i in range(NUM_LEADS):
        for j in range(NUM_LEADS):
            m = 0.5 * (P[i] + P[j])
            js[i, j] = (0.5 * kl(P[i], m) + 0.5 * kl(P[j], m)).sqrt()
    off = js[~torch.eye(NUM_LEADS, dtype=torch.bool)]
    top1 = counts.argmax(dim=1)
    return {
        "n_records": used,
        "lead_js_mean": float(off.mean()) if off.numel() else 0.0,
        "lead_js_min":  float(off.min())  if off.numel() else 0.0,
        "lead_top1_distinct": int(torch.unique(top1).numel()),
        "lead_top1": [(LEADS[i], int(top1[i].item())) for i in range(NUM_LEADS)],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--val-records", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lead-analysis", action="store_true")
    ap.add_argument("--lead-records", type=int, default=200)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    val_data_cfg = dict(cfg["data"])
    val_data_cfg["cache"] = True
    val_data_cfg.pop("virtual_len", None)
    val_data_cfg["record_cache_size"] = 0

    with open(val_data_cfg["val_list"]) as f:
        files = [l.strip() for l in f if l.strip()]
    rng = np.random.RandomState(args.seed)
    sel_idx = sorted(rng.choice(
        len(files), size=min(args.val_records, len(files)), replace=False
    ).tolist())
    # Use a per-ckpt-dir tmp path so parallel evaluations don't race on the
    # same file (matters when running CPU eval for multiple cbs concurrently).
    _tag = os.path.basename(args.ckpt_dir.rstrip("/")) or "default"
    eval_list = f"/tmp/eval_val_list_{_tag}.txt"
    with open(eval_list, "w") as f:
        for i in sel_idx:
            f.write(files[i] + "\n")
    val_data_cfg["val_list"] = eval_list

    print(f"[eval] Caching {len(sel_idx)} records ...", flush=True)
    t0 = time.time()
    val_ds = HEEDBBeatDataset(val_data_cfg, split="val")
    print(f"  -> {len(val_ds):,} beats ({time.time() - t0:.1f}s)")

    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    L = cfg["training"]["loss"]
    alpha = float(L.get("alpha", 1.0))
    beta = float(L.get("beta", 0.5))
    gamma = float(L.get("gamma", 0.0))
    delta = float(L.get("delta", 0.0))
    spec_n_ffts = tuple(L.get("spec_n_ffts", (32, 64, 128)))
    use_grad_loss = bool(L.get("use_gradient_loss", True))

    paths = []
    for n in ("best.pt", "last.pt"):
        p = os.path.join(args.ckpt_dir, n)
        if os.path.exists(p):
            paths.append(p)
    paths += sorted(glob.glob(os.path.join(args.ckpt_dir, "epoch_*.pt")))
    print(f"[eval] {len(paths)} checkpoints")

    cfg["model"].setdefault("normalize", cfg["data"].get("normalize", "record_mad"))
    head = (f"{'ckpt':<22s} {'epoch':>5s} {'rec':>7s} {'vq':>9s} "
            f"{'fid':>7s} {'spec':>7s} {'ent':>7s} {'ppl':>7s} {'cov%':>5s}")
    if args.lead_analysis:
        head += f"  {'js_mean':>9s} {'top1_d':>6s}"
    print()
    print(head)
    print("-" * len(head))

    rows = []
    for cp in paths:
        sd = torch.load(cp, map_location=device)
        ep = sd.get("epoch")
        m_cfg = sd.get("model_cfg") or cfg["model"]
        model = VQVAE(m_cfg).to(device)
        model.load_state_dict(sd["model"])
        a = evaluate_one(model, val_loader, device,
                         alpha=alpha, beta=beta, gamma=gamma, delta=delta,
                         spec_n_ffts=spec_n_ffts, use_grad_loss=use_grad_loss)
        a["ckpt"] = os.path.basename(cp)
        a["epoch"] = ep
        if args.lead_analysis:
            a.update(per_lead_metrics(model, cfg, device,
                                      n_records=args.lead_records, seed=args.seed))
        rows.append(a)
        line = (f"{a['ckpt']:<22s} {str(ep):>5s} {a['rec']:>7.4f} {a['vq']:>9.6f} "
                f"{a['fid']:>7.4f} {a['spec']:>7.4f} {a['ent']:>7.3f} "
                f"{a['perplexity']:>7.1f} {a['coverage_pct']:>5.1f}")
        if args.lead_analysis:
            line += f"  {a['lead_js_mean']:>9.4f} {a['lead_top1_distinct']:>6d}"
        print(line, flush=True)

    def top(rows, key, n=5, rev=False, fmt="{:.4f}"):
        for r in sorted(rows, key=lambda r: r[key], reverse=rev)[:n]:
            print(f"  {r['ckpt']:<22s} ep={str(r['epoch']):<4s} "
                  f"{key}={fmt.format(r[key])}  rec={r['rec']:.4f}  "
                  f"ppl={r['perplexity']:.1f}  cov={r['coverage_pct']:.1f}%")

    print("\n=== Top-5 by val_rec ===")
    top(rows, "rec")
    print("\n=== Top-5 by perplexity ===")
    top(rows, "perplexity", rev=True, fmt="{:.1f}")
    print("\n=== Top-5 by coverage% ===")
    top(rows, "coverage_pct", rev=True, fmt="{:.1f}")
    if args.lead_analysis:
        print("\n=== Top-5 by lead_js_mean ===")
        top(rows, "lead_js_mean", rev=True, fmt="{:.4f}")
        print("\n=== Top-5 by lead_top1_distinct (max=12) ===")
        top(rows, "lead_top1_distinct", rev=True, fmt="{:d}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(rows, f, indent=2, default=str)
        print(f"\n[eval] wrote {args.out}")


if __name__ == "__main__":
    main()
