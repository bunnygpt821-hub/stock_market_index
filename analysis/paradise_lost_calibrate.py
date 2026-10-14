#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PRICE_DIR = ROOT / "data/prices/stooq/jp"
CONSTITUENTS = ROOT / "data/constituents/nikkei225/current.csv"
OUT_DIR = ROOT / "analysis_outputs/paradise_lost"
OUT_DIR.mkdir(parents=True, exist_ok=True)

START = pd.Timestamp("2016-01-01")
END = pd.Timestamp("2026-12-31")

# Fingerprints recovered from the earlier A+ backtest (same 295 events).
TARGET_CURRENT50 = np.array([55, 171, 62, 7])  # <0, 0-5, 5-10, >=10
TARGET_PEAK50 = np.array([59, 110, 96, 30])    # <5, 5-10, 10-15, >=15
TARGET_PEAK100 = np.array([114, 93, 42, 46])   # <10, 10-15, 15-20, >=20


def bin_current50(x: pd.Series) -> np.ndarray:
    a = x.to_numpy(float)
    return np.array([
        np.sum(a < 0),
        np.sum((a >= 0) & (a < 5)),
        np.sum((a >= 5) & (a < 10)),
        np.sum(a >= 10),
    ], dtype=int)


def bin_peak50(x: pd.Series) -> np.ndarray:
    a = x.to_numpy(float)
    return np.array([
        np.sum(a < 5),
        np.sum((a >= 5) & (a < 10)),
        np.sum((a >= 10) & (a < 15)),
        np.sum(a >= 15),
    ], dtype=int)


def bin_peak100(x: pd.Series) -> np.ndarray:
    a = x.to_numpy(float)
    return np.array([
        np.sum(a < 10),
        np.sum((a >= 10) & (a < 15)),
        np.sum((a >= 15) & (a < 20)),
        np.sum(a >= 20),
    ], dtype=int)


def arrtxt(a: np.ndarray) -> str:
    return "/".join(str(int(v)) for v in a)


def load_codes() -> list[str]:
    with CONSTITUENTS.open(newline="", encoding="utf-8-sig") as f:
        return [str(r["code"]).strip() for r in csv.DictReader(f) if str(r.get("code", "")).strip()]


def prepare_one(code: str) -> pd.DataFrame | None:
    path = PRICE_DIR / f"{code}.csv"
    if not path.exists():
        return None
    d = pd.read_csv(path, usecols=["date", "open", "high", "low", "close"])
    if d.empty:
        return None
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    for c in ["open", "high", "low", "close"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["date", "open", "high", "low", "close"]).sort_values("date")
    d = d.drop_duplicates("date", keep="last").reset_index(drop=True)
    if len(d) < 120:
        return None

    pc = d["close"].shift(1)
    tr = pd.concat([
        d["high"] - d["low"],
        (d["high"] - pc).abs(),
        (d["low"] - pc).abs(),
    ], axis=1).max(axis=1)
    d["atr20"] = tr.rolling(20, min_periods=20).mean()
    for n in (15, 25, 50, 100):
        d[f"ma{n}"] = d["close"].rolling(n, min_periods=n).mean()
        d[f"dev{n}"] = (d["close"] / d[f"ma{n}"] - 1.0) * 100.0

    d["peak50_inc20"] = d["dev50"].rolling(20, min_periods=1).max()
    d["peak100_inc20"] = d["dev100"].rolling(20, min_periods=1).max()
    d["peak50_exc20"] = d["dev50"].shift(1).rolling(20, min_periods=1).max()
    d["peak100_exc20"] = d["dev100"].shift(1).rolling(20, min_periods=1).max()

    # High-zone references, both including and excluding the signal bar.
    h60i = d["high"].rolling(60, min_periods=60).max()
    h60e = d["high"].shift(1).rolling(60, min_periods=60).max()
    c60i = d["close"].rolling(60, min_periods=60).max()
    c60e = d["close"].shift(1).rolling(60, min_periods=60).max()

    d["close_h60i_pct"] = (h60i - d["close"]) / h60i
    d["close_h60e_pct"] = (h60e - d["close"]) / h60e
    d["high_h60i_pct"] = (h60i - d["high"]) / h60i
    d["high_h60e_pct"] = (h60e - d["high"]) / h60e
    d["open_h60e_pct"] = (h60e - d["open"]) / h60e
    d["prevclose_h60e_pct"] = (h60e - d["close"].shift(1)) / h60e
    d["prevhigh_h60e_pct"] = (h60e - d["high"].shift(1)) / h60e
    d["close_c60i_pct"] = (c60i - d["close"]) / c60i
    d["close_c60e_pct"] = (c60e - d["close"]) / c60e
    d["high_c60e_pct"] = (c60e - d["high"]) / c60e

    d["close_h60e_atr"] = (h60e - d["close"]) / d["atr20"]
    d["high_h60e_atr"] = (h60e - d["high"]) / d["atr20"]
    d["open_h60e_atr"] = (h60e - d["open"]) / d["atr20"]
    d["prevclose_h60e_atr"] = (h60e - d["close"].shift(1)) / d["atr20"]

    body = d["open"] - d["close"]
    rng = d["high"] - d["low"]
    close_pos = (d["close"] - d["low"]) / rng.replace(0, np.nan)

    d["base_cross"] = (
        (d["date"] >= START) & (d["date"] <= END)
        & (d["open"] > d["close"])
        & (body >= 0.8 * d["atr20"])
        & (close_pos <= 0.30)
        & (d["close"].shift(1) >= d["ma15"].shift(1))
        & (d["close"] < d["ma15"])
    )
    # Stricter interpretation of 'first break': prior 2/3/5 closes all above their MA15.
    for k in (2, 3, 5):
        ok = pd.Series(True, index=d.index)
        for j in range(1, k + 1):
            ok &= d["close"].shift(j) >= d["ma15"].shift(j)
        d[f"base_above{k}"] = (
            (d["date"] >= START) & (d["date"] <= END)
            & (d["open"] > d["close"])
            & (body >= 0.8 * d["atr20"])
            & (close_pos <= 0.30)
            & ok
            & (d["close"] < d["ma15"])
        )

    keep = [
        "date", "dev25", "dev50", "dev100",
        "peak50_inc20", "peak100_inc20", "peak50_exc20", "peak100_exc20",
        "close_h60i_pct", "close_h60e_pct", "high_h60i_pct", "high_h60e_pct",
        "open_h60e_pct", "prevclose_h60e_pct", "prevhigh_h60e_pct",
        "close_c60i_pct", "close_c60e_pct", "high_c60e_pct",
        "close_h60e_atr", "high_h60e_atr", "open_h60e_atr", "prevclose_h60e_atr",
        "base_cross", "base_above2", "base_above3", "base_above5",
    ]
    out = d[keep].copy()
    out["code"] = code
    return out


def score_set(e: pd.DataFrame) -> tuple[int, int, int, str, str, str, str]:
    cur = bin_current50(e["dev50"])
    sc_cur = int(np.abs(cur - TARGET_CURRENT50).sum())

    p50i = bin_peak50(e["peak50_inc20"].dropna())
    p100i = bin_peak100(e["peak100_inc20"].dropna())
    sc_inc = int(np.abs(p50i - TARGET_PEAK50).sum() + np.abs(p100i - TARGET_PEAK100).sum())

    p50e = bin_peak50(e["peak50_exc20"].dropna())
    p100e = bin_peak100(e["peak100_exc20"].dropna())
    sc_exc = int(np.abs(p50e - TARGET_PEAK50).sum() + np.abs(p100e - TARGET_PEAK100).sum())

    if sc_inc <= sc_exc:
        peak_mode = "include_signal"
        peak_score = sc_inc
        p50, p100 = p50i, p100i
    else:
        peak_mode = "exclude_signal"
        peak_score = sc_exc
        p50, p100 = p50e, p100e
    return sc_cur, peak_score, sc_cur * 10 + peak_score, arrtxt(cur), arrtxt(p50), arrtxt(p100), peak_mode


def main() -> None:
    frames = []
    missing = []
    for code in load_codes():
        x = prepare_one(code)
        if x is None:
            missing.append(code)
        else:
            frames.append(x)
    allrows = pd.concat(frames, ignore_index=True)

    pct_metrics = [c for c in allrows.columns if c.endswith("_pct")]
    atr_metrics = [c for c in allrows.columns if c.endswith("_atr")]
    base_modes = ["base_cross", "base_above2", "base_above3", "base_above5"]

    results: list[dict] = []
    for base in base_modes:
        b = allrows.loc[allrows[base]].copy()
        for metric in pct_metrics:
            vals = b[metric]
            for thr in np.arange(0.0, 0.2001, 0.001):
                e = b.loc[vals <= thr]
                if len(e) < 150 or len(e) > 500:
                    continue
                sc_cur, sc_peak, total, cur, p50, p100, pmode = score_set(e)
                results.append({
                    "base": base, "metric": metric, "threshold": round(float(thr), 4),
                    "threshold_unit": "fraction", "N": len(e),
                    "current50_counts": cur, "peak50_counts": p50, "peak100_counts": p100,
                    "score_current": sc_cur, "score_peak": sc_peak, "score_total": total,
                    "peak_mode": pmode,
                })
        for metric in atr_metrics:
            vals = b[metric]
            for thr in np.arange(0.0, 5.0001, 0.05):
                e = b.loc[vals <= thr]
                if len(e) < 150 or len(e) > 500:
                    continue
                sc_cur, sc_peak, total, cur, p50, p100, pmode = score_set(e)
                results.append({
                    "base": base, "metric": metric, "threshold": round(float(thr), 3),
                    "threshold_unit": "ATR", "N": len(e),
                    "current50_counts": cur, "peak50_counts": p50, "peak100_counts": p100,
                    "score_current": sc_cur, "score_peak": sc_peak, "score_total": total,
                    "peak_mode": pmode,
                })

    r = pd.DataFrame(results).sort_values(
        ["score_total", "score_current", "score_peak", "N", "base", "metric", "threshold"],
        kind="stable",
    ).reset_index(drop=True)
    r.to_csv(OUT_DIR / "calibration_candidates.csv", index=False)
    r.head(100).to_csv(OUT_DIR / "calibration_top100.csv", index=False)

    # Export the best event set for inspection / next-stage tests.
    best = r.iloc[0]
    b = allrows.loc[allrows[str(best["base"])]].copy()
    e = b.loc[b[str(best["metric"])] <= float(best["threshold"])].copy()
    e = e.sort_values(["date", "code"])
    e.to_csv(OUT_DIR / "best_candidate_events.csv", index=False)

    with (OUT_DIR / "calibration_summary.txt").open("w", encoding="utf-8") as f:
        f.write(f"codes_loaded={len(frames)} missing={len(missing)} missing_codes={','.join(missing)}\n")
        f.write(f"base_cross_count={int(allrows['base_cross'].sum())}\n")
        f.write("target_current50=55/171/62/7\n")
        f.write("target_peak50=59/110/96/30\n")
        f.write("target_peak100=114/93/42/46\n\n")
        f.write("BEST\n")
        f.write(best.to_string())
        f.write("\n\nTOP20\n")
        f.write(r.head(20).to_string(index=False))
        f.write("\n")

    print((OUT_DIR / "calibration_summary.txt").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
