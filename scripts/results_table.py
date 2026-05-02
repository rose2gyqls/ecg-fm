#!/usr/bin/env python
"""
results_table.py
=================
Pivot benchmark/results/<TS>_ecg_fm_hb/results_all.csv into tasks × codebook
sizes table with macro-AUROC / macro-AUPRC values.

Usage:
    # Latest results dir, all 3 eval modes
    python scripts/results_table.py

    # Specific eval mode
    python scripts/results_table.py --mode finetune_linear

    # Specific results CSV
    python scripts/results_table.py --csv /path/to/results_all.csv

    # Markdown output (pipe-friendly)
    python scripts/results_table.py --format md

Codebook size is parsed from save_dir basename (ecg_fm_hb_cb<K>__<task>__<mode>),
since the CSV's `model` column doesn't distinguish sizes (all share the same
encoder class). Empty cells = run not yet completed (no test_metrics.txt).
"""
from __future__ import annotations
import argparse
import re
import sys
from pathlib import Path

import pandas as pd

SIZES = [128, 256, 512, 1024, 2048]

# Task display order matching the paper canonical table.
TASK_ORDER = [
    # Adult ECG
    ("ptb",            "PTB"),
    ("ningbo",         "Ningbo"),
    ("cpsc2018",       "CPSC2018"),
    ("cpsc_extra",     "CPSC-Extra"),
    ("georgia",        "Georgia"),
    ("chapman",        "Chapman"),
    ("chapman_rhythm", "Chapman (rhythm)"),
    ("sph_diag",       "SPH"),
    ("code15",         "CODE-15%"),
    ("ptbxl_all",      "PTB-XL (all)"),
    ("ptbxl_diag",     "PTB-XL (diag)"),
    ("ptbxl_form",     "PTB-XL (form)"),
    ("ptbxl_rhythm",   "PTB-XL (rhythm)"),
    ("ptbxl_sub",      "PTB-XL (sub)"),
    ("ptbxl_super",    "PTB-XL (super)"),
    # Pediatric ECG
    ("zzu_pecg",       "ZZU pECG"),
    # Cardiac structure
    ("echonext",       "EchoNext"),
]
TASK_DISPLAY = dict(TASK_ORDER)
TASK_INDEX = {t: i for i, (t, _) in enumerate(TASK_ORDER)}

CB_RE = re.compile(r"ecg_fm_hb_cb(\d+)__")


def latest_results_csv() -> Path | None:
    candidates = sorted(
        Path("/home/irteam/local-node-d/hbkimi/benchmark/results").glob("*_ecg_fm_hb"),
        key=lambda p: p.stat().st_mtime,
    )
    for d in reversed(candidates):
        csv = d / "results_all.csv"
        if csv.exists():
            return csv
    return None


def extract_codebook(row) -> int | None:
    sd = str(row.get("save_dir", ""))
    m = CB_RE.search(sd)
    return int(m.group(1)) if m else None


def fmt_cell(v: float | None, decimals: int = 4) -> str:
    if v is None or pd.isna(v):
        return "-"
    return f"{v:.{decimals}f}"


def render_table(df: pd.DataFrame, metric: str, mode: str, fmt: str) -> str:
    """Render one (mode, metric) pivot as a string."""
    sub = df[df["eval_mode"] == mode].copy()
    if sub.empty:
        return f"## {mode} — {metric}\n\n(no completed runs)\n"

    sub["cb"] = sub.apply(extract_codebook, axis=1)
    sub = sub.dropna(subset=["cb", metric])
    sub["cb"] = sub["cb"].astype(int)

    # Pivot (task, cb) -> metric. If duplicates (re-runs), take latest.
    pivot = (
        sub.sort_values("timestamp")
        .groupby(["task", "cb"])[metric]
        .last()
        .unstack("cb")
    )
    # reindex to display order
    pivot = pivot.reindex(index=[t for t, _ in TASK_ORDER if t in pivot.index])
    pivot = pivot.reindex(columns=SIZES)

    # Build output
    if fmt == "md":
        header = "| Task | " + " | ".join(str(s) for s in SIZES) + " |"
        sep = "|------|" + "|".join("------:" for _ in SIZES) + "|"
        lines = [f"## {mode} — {metric}", "", header, sep]
        for task in pivot.index:
            disp = TASK_DISPLAY.get(task, task)
            cells = [fmt_cell(pivot.loc[task, c]) for c in SIZES]
            lines.append(f"| {disp} | " + " | ".join(cells) + " |")
        return "\n".join(lines) + "\n"

    # plain text (column-aligned)
    col_w = 12
    name_w = max(18, max(len(TASK_DISPLAY.get(t, t)) for t in pivot.index) + 2) if len(pivot.index) > 0 else 18
    out = [f"\n=== {mode} — {metric} ==="]
    head = f"{'Task':<{name_w}}" + "".join(f"{s:>{col_w}}" for s in SIZES)
    out.append(head)
    out.append("-" * len(head))
    for task in pivot.index:
        disp = TASK_DISPLAY.get(task, task)
        row = f"{disp:<{name_w}}" + "".join(
            f"{fmt_cell(pivot.loc[task, c]):>{col_w}}" for c in SIZES
        )
        out.append(row)
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default=None,
                    help="Path to results_all.csv (default: latest under benchmark/results)")
    ap.add_argument("--mode", type=str, default=None,
                    choices=["linear_probe", "attention_probe",
                             "finetune_linear", "finetune_attention"],
                    help="Filter by eval mode (default: print tables for all modes found)")
    ap.add_argument("--format", choices=["plain", "md"], default="plain")
    ap.add_argument("--metric", choices=["auroc", "auprc", "both"], default="both")
    args = ap.parse_args()

    csv_path = Path(args.csv) if args.csv else latest_results_csv()
    if csv_path is None or not csv_path.exists():
        print(f"[error] results_all.csv not found "
              f"({'no completed runs yet' if csv_path is None else csv_path})",
              file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(csv_path)
    if df.empty:
        print(f"[info] {csv_path}: 0 completed runs")
        return

    print(f"# results: {csv_path}")
    print(f"# completed runs: {len(df)}")
    print(f"# tasks: {sorted(df['task'].unique())}")
    print(f"# modes:  {sorted(df['eval_mode'].unique())}")
    print()

    modes = [args.mode] if args.mode else sorted(df["eval_mode"].unique())
    metrics = (
        ["test_auroc_macro"] if args.metric == "auroc"
        else ["test_auprc_macro"] if args.metric == "auprc"
        else ["test_auroc_macro", "test_auprc_macro"]
    )

    for mode in modes:
        for metric in metrics:
            print(render_table(df, metric, mode, args.format))


if __name__ == "__main__":
    main()
