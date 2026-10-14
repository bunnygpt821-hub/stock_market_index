#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = Path(__file__).with_name("paradise_lost_calibrate.py")
OUT_DIR = ROOT / "analysis_outputs/paradise_lost_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

spec = importlib.util.spec_from_file_location("plbase", BASE_PATH)
plbase = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(plbase)


def dedup_first(e: pd.DataFrame, gap: int) -> pd.DataFrame:
    if gap <= 0 or e.empty:
        return e.copy()
    e = e.sort_values(["code", "bar_index", "date"], kind="stable")
    keep_idx = []
    for _, g in e.groupby("code", sort=False):
        last = None
        for idx, row in g.iterrows():
            b = int(row["bar_index"])
            if last is None or (b - last) > gap:
                keep_idx.append(idx)
                last = b
    return e.loc[keep_idx].sort_values(["date", "code"], kind="stable")


def dedup_last(e: pd.DataFrame, gap: int) -> pd.DataFrame:
    if gap <= 0 or e.empty:
        return e.copy()
    # Cluster consecutive signals whose bar distance from the previously observed
    # signal is <= gap, and retain the final signal in each cluster.
    e = e.sort_values(["code", "bar_index", "date"], kind="stable")
    keep_idx = []
    for _, g in e.groupby("code", sort=False):
        cluster_last_idx = None
        prev_b = None
        for idx, row in g.iterrows():
            b = int(row["bar_index"])
            if prev_b is None or (b - prev_b) <= gap:
                cluster_last_idx = idx
            else:
                if cluster_last_idx is not None:
                    keep_idx.append(cluster_last_idx)
                cluster_last_idx = idx
            prev_b = b
        if cluster_last_idx is not None:
            keep_idx.append(cluster_last_idx)
    return e.loc[keep_idx].sort_values(["date", "code"], kind="stable")


def main() -> None:
    frames = []
    missing = []
    for code in plbase.load_codes():
        x = plbase.prepare_one(code)
        if x is None:
            missing.append(code)
        else:
            x = x.sort_values("date", kind="stable").copy()
            x["bar_index"] = np.arange(len(x), dtype=int)
            frames.append(x)
    allrows = pd.concat(frames, ignore_index=True)

    pct_metrics = [c for c in allrows.columns if c.endswith("_pct")]
    atr_metrics = [c for c in allrows.columns if c.endswith("_atr")]
    base_modes = ["base_cross", "base_above2", "base_above3", "base_above5"]
    gaps = [0, 3, 5, 7, 10, 12, 15, 20]
    dedup_modes = ["first", "last"]

    results: list[dict] = []

    def evaluate(base: str, metric: str, threshold: float, unit: str, raw: pd.DataFrame) -> None:
        for gap in gaps:
            for dm in dedup_modes:
                if gap == 0 and dm == "last":
                    continue
                e = dedup_first(raw, gap) if dm == "first" else dedup_last(raw, gap)
                if len(e) < 150 or len(e) > 500:
                    continue
                sc_cur, sc_peak, total, cur, p50, p100, pmode = plbase.score_set(e)
                n_penalty = abs(len(e) - 295)
                # Current50 is the strongest fingerprint, then exact N, then Peak maps.
                fingerprint = sc_cur * 20 + n_penalty * 5 + sc_peak
                results.append({
                    "base": base,
                    "metric": metric,
                    "threshold": threshold,
                    "threshold_unit": unit,
                    "dedup_mode": dm,
                    "dedup_gap": gap,
                    "N": len(e),
                    "current50_counts": cur,
                    "peak50_counts": p50,
                    "peak100_counts": p100,
                    "score_current": sc_cur,
                    "score_peak": sc_peak,
                    "n_penalty": n_penalty,
                    "score_fingerprint": fingerprint,
                    "peak_mode": pmode,
                })

    for base in base_modes:
        b = allrows.loc[allrows[base]].copy()
        for metric in pct_metrics:
            vals = b[metric]
            for thr in np.arange(0.0, 0.2001, 0.001):
                raw = b.loc[vals <= thr]
                # prefilter loose range because dedup can only reduce N
                if len(raw) < 150 or len(raw) > 900:
                    continue
                evaluate(base, metric, round(float(thr), 4), "fraction", raw)
        for metric in atr_metrics:
            vals = b[metric]
            for thr in np.arange(0.0, 5.0001, 0.05):
                raw = b.loc[vals <= thr]
                if len(raw) < 150 or len(raw) > 900:
                    continue
                evaluate(base, metric, round(float(thr), 3), "ATR", raw)

    r = pd.DataFrame(results).sort_values(
        ["score_fingerprint", "score_current", "n_penalty", "score_peak", "base", "metric", "threshold", "dedup_gap"],
        kind="stable",
    ).reset_index(drop=True)
    r.to_csv(OUT_DIR / "calibration_candidates_v2.csv", index=False)
    r.head(200).to_csv(OUT_DIR / "calibration_top200_v2.csv", index=False)

    exact_n = r.loc[r["N"] == 295].copy()
    exact_n.head(200).to_csv(OUT_DIR / "exact_N295_top200.csv", index=False)
    exact_cur = r.loc[r["current50_counts"] == "55/171/62/7"].copy()
    exact_cur.head(200).to_csv(OUT_DIR / "exact_current50_top200.csv", index=False)

    best = r.iloc[0]
    b = allrows.loc[allrows[str(best["base"])]].copy()
    raw = b.loc[b[str(best["metric"])] <= float(best["threshold"])].copy()
    if str(best["dedup_mode"]) == "first":
        e = dedup_first(raw, int(best["dedup_gap"]))
    else:
        e = dedup_last(raw, int(best["dedup_gap"]))
    e.to_csv(OUT_DIR / "best_candidate_events_v2.csv", index=False)

    with (OUT_DIR / "calibration_summary_v2.txt").open("w", encoding="utf-8") as f:
        f.write(f"codes_loaded={len(frames)} missing={len(missing)}\n")
        f.write("target_N=295\n")
        f.write("target_current50=55/171/62/7\n")
        f.write("target_peak50=59/110/96/30\n")
        f.write("target_peak100=114/93/42/46\n\n")
        f.write("BEST\n")
        f.write(best.to_string())
        f.write("\n\nTOP30\n")
        f.write(r.head(30).to_string(index=False))
        f.write("\n\nBEST_EXACT_N295\n")
        if exact_n.empty:
            f.write("NONE\n")
        else:
            f.write(exact_n.head(20).to_string(index=False))
            f.write("\n")
        f.write("\nBEST_EXACT_CURRENT50\n")
        if exact_cur.empty:
            f.write("NONE\n")
        else:
            f.write(exact_cur.head(20).to_string(index=False))
            f.write("\n")

    print((OUT_DIR / "calibration_summary_v2.txt").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
