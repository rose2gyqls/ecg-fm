#!/usr/bin/env python
"""
Extract CLS embeddings from our pretrained ECG foundation model
into the format consumed by /home/irteam/local-node-d/tykim/visuallize/
(umap_view.py, quick_dx, etc.).

For each codebook size K (one of 128/256/512/1024/2048, v4 pretrain),
runs inference on PTB-XL and ZZU-pECG H5 records *in the exact row
order of their respective table_csv files* — this is what the visualize
project assumes when joining paper labels by `filepath`.

Output (default `--model-prefix Ours`):

  /home/irteam/local-node-d/tykim/visuallize/results/embeddings/
      Ours-cb{K}_ptbxl.npy           (21837, 512)
      Ours-cb{K}_zzu.npy             (12327, 512)
      Ours-cb{K}_meta.json
      Ours-cb{K}_meta.npz            (legacy fallback)

Usage:
  conda activate hbkim
  python scripts/extract_embeddings_for_visualize.py --cb 128
  python scripts/extract_embeddings_for_visualize.py --cb 256 --datasets ptbxl
  python scripts/extract_embeddings_for_visualize.py --cb all      # 모든 cb (시간 소요)
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = "/home/irteam/local-node-d/hbkimi/ecg-fm"
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from data.datasets.heedb_ecg_dataset import HEEDBECGDataset  # noqa: E402
from models.tokenizer.vqvae import VQVAE  # noqa: E402
from models.transformer.ecg_model import ECGFoundationModel  # noqa: E402

DATASETS = {
    "ptbxl": {
        "table_csv": "/home/irteam/ddn-opendata1/h5/physionet/v2.0/ptbxl_table.csv",
        "h5_root":   "/home/irteam/ddn-opendata1/h5/physionet/v2.0",
    },
    "zzu": {
        "table_csv": "/home/irteam/ddn-opendata1/h5/ZZU-pECG/v2.0/ecg_table.csv",
        "h5_root":   "/home/irteam/ddn-opendata1/h5/ZZU-pECG/v2.0",
    },
}

EMB_OUT = Path("/home/irteam/local-node-d/tykim/visuallize/results/embeddings")
CKPT_ROOT = Path(PROJECT_ROOT) / "checkpoints"
CFG_DIR = Path(PROJECT_ROOT) / "configs" / "pretrain"

ALL_CBS = (128, 256, 512, 1024, 2048)


def load_pretrain_cfg(cb: int) -> dict:
    p = CFG_DIR / f"masked_beat_heedb_cb{cb}_v4.yaml"
    with open(p) as f:
        return yaml.safe_load(f)


def load_tokenizer(pre_cfg: dict, device) -> VQVAE:
    tok_p = pre_cfg["tokenizer"]["ckpt"]
    tok_p = tok_p if os.path.isabs(tok_p) else os.path.join(PROJECT_ROOT, tok_p)
    ckpt = torch.load(tok_p, map_location="cpu")
    tcfg = ckpt.get("model_cfg")
    if tcfg is None:
        ymp = pre_cfg["tokenizer"].get("model_cfg_yaml")
        if not ymp:
            raise RuntimeError("tokenizer model_cfg missing in ckpt and no fallback yaml")
        with open(ymp) as f:
            tcfg = yaml.safe_load(f)["model"]
    tok = VQVAE(tcfg)
    tok.load_state_dict(ckpt["model"])
    tok.eval().to(device)
    return tok


def load_model(pre_cfg: dict, device, ckpt_path: Path) -> ECGFoundationModel:
    mcfg = dict(pre_cfg["model"])
    mcfg["codebook_size"] = int(pre_cfg["tokenizer"]["codebook_size"])
    mcfg["n_leads"]       = int(pre_cfg["data"].get("n_leads", 12))
    mcfg["max_beats"]     = int(pre_cfg["data"].get("max_beats_per_lead", 15))

    # Mirror training/pretrain/train.py: propagate RR normalization stats from
    # data.normalization into context so RhythmMLP registers norm_mean/norm_std
    # buffers (state_dict keys must match the saved checkpoint).
    norm_cfg = (pre_cfg.get("data", {}) or {}).get("normalization", {}) or {}
    rr_mean = norm_cfg.get("rr_mean")
    rr_std  = norm_cfg.get("rr_std")
    if rr_mean is not None and rr_std is not None:
        ctx = dict(mcfg.get("context", {}) or {})
        ctx["rhythm_mean"] = list(map(float, rr_mean))
        ctx["rhythm_std"]  = list(map(float, rr_std))
        mcfg["context"] = ctx

    m = ECGFoundationModel(mcfg).to(device).eval()
    state = torch.load(str(ckpt_path), map_location=device)
    m.load_state_dict(state["model"])
    print(f"  [model] loaded {ckpt_path.name}  epoch={state.get('epoch','?')} "
          f"metric={state.get('metric', state.get('best_val_loss','?'))}")
    return m


def write_filepath_list(table_csv: str, h5_root: str, out_path: str) -> int:
    """table_csv 의 filepath 컬럼을 그대로 (절대경로 prefix만 붙여) 기록.

    visualize 의 _load_paper_labels_aligned 가 *table_csv 행 순서*로 라벨을
    join 하므로, 임베딩도 **반드시 같은 순서** 여야 한다.
    """
    df = pd.read_csv(table_csv, low_memory=False, usecols=["filepath"])
    paths = [os.path.join(h5_root, p) for p in df["filepath"].tolist()]
    with open(out_path, "w") as f:
        f.writelines(p + "\n" for p in paths)
    return len(paths)


@torch.no_grad()
def extract_for_dataset(model: ECGFoundationModel, tokenizer: VQVAE,
                        pre_cfg: dict, ds_name: str, ds_meta: dict,
                        batch_size: int, num_workers: int,
                        device: torch.device, list_path: str) -> np.ndarray:
    n_total = write_filepath_list(ds_meta["table_csv"], ds_meta["h5_root"], list_path)
    dcfg = dict(pre_cfg["data"])
    dcfg["train_list"] = list_path
    ds = HEEDBECGDataset(dcfg, split="train")
    assert len(ds) == n_total, (len(ds), n_total)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False,
        persistent_workers=False,
    )
    embs: list[np.ndarray] = []
    n_done = 0
    t0 = time.time()
    log_every = max(1, n_total // batch_size // 50)  # 50 progress lines per ds
    for bi, batch in enumerate(loader):
        beats = batch["beats"].to(device, non_blocking=True)
        rr    = batch["rr_feats"].to(device, non_blocking=True)
        stft  = batch["stft"].to(device, non_blocking=True)
        B, N, L, W = beats.shape
        _, idx_flat = tokenizer.encode(beats.view(B * N * L, 1, W))
        idx = idx_flat.view(B, N, L)
        out = model(idx, rr, stft)              # (B, 1+N*L, d)
        cls = out[:, 0, :].cpu().numpy().astype(np.float32)
        embs.append(cls)
        n_done += B
        if (bi + 1) % log_every == 0:
            elapsed = time.time() - t0
            rate = n_done / max(elapsed, 1e-3)
            eta = (n_total - n_done) / max(rate, 1e-3)
            print(f"  [{ds_name}] {n_done:>6,}/{n_total:,}  "
                  f"({rate:5.1f} rec/s, ETA {eta/60:5.1f} min)")
    embs_np = np.concatenate(embs, axis=0)
    assert len(embs_np) == n_total, (len(embs_np), n_total)
    elapsed = time.time() - t0
    print(f"  [{ds_name}] done {n_total:,} in {elapsed/60:.1f} min "
          f"({n_total/elapsed:.1f} rec/s)  shape={embs_np.shape}")
    return embs_np


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cb", required=True,
                    help="codebook size: 128 | 256 | 512 | 1024 | 2048 | all")
    ap.add_argument("--datasets", default="ptbxl,zzu",
                    help="콤마 구분 (default: ptbxl,zzu)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--model-prefix", default="Ours",
                    help="저장 모델명 prefix (default: Ours → 'Ours-cb128')")
    ap.add_argument("--out-dir", default=str(EMB_OUT),
                    help=f"임베딩 출력 dir (default: {EMB_OUT})")
    ap.add_argument("--ckpt-suffix", default="v4",
                    help="pretrain ckpt 디렉토리 suffix (default: v4 → "
                         "checkpoints/pretrain_heedb_cb{K}_v4/best.pt)")
    args = ap.parse_args()

    if args.cb.lower() == "all":
        cbs = list(ALL_CBS)
    else:
        cbs = [int(args.cb)]
        if cbs[0] not in ALL_CBS:
            raise SystemExit(f"--cb must be one of {ALL_CBS} or 'all'")

    ds_names = [d.strip() for d in args.datasets.split(",") if d.strip()]
    for d in ds_names:
        if d not in DATASETS:
            raise SystemExit(f"unknown dataset {d!r} (have {list(DATASETS)})")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)
    print(f"device: {device}")
    print(f"out_dir: {out_dir}")

    for cb in cbs:
        print("\n" + "=" * 64)
        print(f"  pretrain cb={cb}  ({args.ckpt_suffix})")
        print("=" * 64)
        pre_cfg = load_pretrain_cfg(cb)
        tok = load_tokenizer(pre_cfg, device)
        ckpt_path = CKPT_ROOT / f"pretrain_heedb_cb{cb}_{args.ckpt_suffix}" / "best.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(ckpt_path)
        model = load_model(pre_cfg, device, ckpt_path)
        model_name = f"{args.model_prefix}-cb{cb}"

        for d in ds_names:
            list_path = f"/tmp/_extract_emb_{model_name}_{d}.txt"
            embs = extract_for_dataset(
                model, tok, pre_cfg, d, DATASETS[d],
                args.batch_size, args.num_workers, device, list_path,
            )
            out_npy = out_dir / f"{model_name}_{d}.npy"
            np.save(out_npy, embs)
            print(f"  saved {out_npy.name} {embs.shape}")

        meta = {
            "feature_dim": int(model.d_model),
            "model_name": model_name,
            "datasets": ds_names,
            "pretrain_ckpt": str(ckpt_path),
            "tokenizer_ckpt": pre_cfg["tokenizer"]["ckpt"],
            "codebook_size": int(cb),
            "extracted_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        with open(out_dir / f"{model_name}_meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        # legacy fallback recognized by discover_models_and_datasets
        np.savez(out_dir / f"{model_name}_meta.npz",
                 feature_dim=np.int64(model.d_model),
                 safe=model_name)
        print(f"  meta -> {model_name}_meta.json (+ .npz)")

        # GPU 메모리 회수
        del model, tok
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
