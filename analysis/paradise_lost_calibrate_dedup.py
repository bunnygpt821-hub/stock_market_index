#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
import paradise_lost_calibrate as base  # noqa: E402

OUT_DIR = ROOT / "analysis_outputs/paradise_lost_dedup"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_N = 295
TARGET_CURRENT50 = np.array([55,171,62,7])
TARGET_PEAK50 = np.array([59,110,96,30])
TARGET_PEAK100 = np.array([114,93,42,46])


def trading_index_map(code: str) -> dict[pd.Timestamp, int]:
    p = base.PRICE_DIR / f"{code}.csv"
    d = pd.read_csv(p, usecols=["date"])
    s = pd.to_datetime(d["date"], errors="coerce").dropna().drop_duplicates().sort_values().reset_index(drop=True)
    return {pd.Timestamp(v): i for i, v in enumerate(s)}


def add_seq(allrows: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for code, g in allrows.groupby("code", sort=False):
        m = trading_index_map(str(code))
        x = g.copy()
        x["seq"] = x["date"].map(m)
        parts.append(x)
    return pd.concat(parts, ignore_index=True)


def dedup_keep_first(e: pd.DataFrame, gap: int) -> pd.DataFrame:
    out=[]
    for code,g in e.sort_values(["code","seq"]).groupby("code", sort=False):
        last=None
        for _,r in g.iterrows():
            s=int(r["seq"])
            if last is None or s-last>gap:
                out.append(r)
                last=s
    return pd.DataFrame(out)


def dedup_keep_last(e: pd.DataFrame, gap: int) -> pd.DataFrame:
    out=[]
    for code,g in e.sort_values(["code","seq"], ascending=[True,False]).groupby("code", sort=False):
        last=None
        for _,r in g.iterrows():
            s=int(r["seq"])
            if last is None or last-s>gap:
                out.append(r)
                last=s
    return pd.DataFrame(out)


def score(e: pd.DataFrame):
    cur=base.bin_current50(e["dev50"])
    p50i=base.bin_peak50(e["peak50_inc20"].dropna()); p100i=base.bin_peak100(e["peak100_inc20"].dropna())
    p50e=base.bin_peak50(e["peak50_exc20"].dropna()); p100e=base.bin_peak100(e["peak100_exc20"].dropna())
    inc=int(np.abs(p50i-TARGET_PEAK50).sum()+np.abs(p100i-TARGET_PEAK100).sum())
    exc=int(np.abs(p50e-TARGET_PEAK50).sum()+np.abs(p100e-TARGET_PEAK100).sum())
    if inc<=exc:
        p50,p100,ps,pm=p50i,p100i,inc,"include_signal"
    else:
        p50,p100,ps,pm=p50e,p100e,exc,"exclude_signal"
    cs=int(np.abs(cur-TARGET_CURRENT50).sum())
    ns=abs(len(e)-TARGET_N)
    total=ns*20+cs*10+ps
    return total,ns,cs,ps,base.arrtxt(cur),base.arrtxt(p50),base.arrtxt(p100),pm


def main():
    frames=[]
    for code in base.load_codes():
        x=base.prepare_one(code)
        if x is not None: frames.append(x)
    allrows=add_seq(pd.concat(frames, ignore_index=True))
    pct_metrics=[c for c in allrows.columns if c.endswith("_pct")]
    atr_metrics=[c for c in allrows.columns if c.endswith("_atr")]
    base_modes=["base_cross","base_above2","base_above3","base_above5"]
    rows=[]
    for bm in base_modes:
        b=allrows.loc[allrows[bm]].copy()
        for metric,grid,unit in [
            *[(m,np.arange(0.0,0.2001,0.001),"fraction") for m in pct_metrics],
            *[(m,np.arange(0.0,5.0001,0.05),"ATR") for m in atr_metrics],
        ]:
            vals=b[metric]
            for thr in grid:
                raw=b.loc[vals<=thr].copy()
                if len(raw)<150 or len(raw)>700: continue
                for gap in (5,10,15):
                    for mode,fn in (("first",dedup_keep_first),("last",dedup_keep_last)):
                        e=fn(raw,gap)
                        if len(e)<180 or len(e)>420: continue
                        total,ns,cs,ps,cur,p50,p100,pm=score(e)
                        rows.append({
                            "base":bm,"metric":metric,"threshold":round(float(thr),4),"unit":unit,
                            "dedup_gap":gap,"dedup_mode":mode,"raw_N":len(raw),"N":len(e),
                            "current50_counts":cur,"peak50_counts":p50,"peak100_counts":p100,
                            "score_N":ns,"score_current":cs,"score_peak":ps,"score_total":total,"peak_mode":pm,
                        })
    r=pd.DataFrame(rows).sort_values(["score_total","score_N","score_current","score_peak","N"], kind="stable").reset_index(drop=True)
    r.to_csv(OUT_DIR/"dedup_calibration_candidates.csv",index=False)
    r.head(100).to_csv(OUT_DIR/"dedup_calibration_top100.csv",index=False)
    with (OUT_DIR/"dedup_summary.txt").open("w",encoding="utf-8") as f:
        f.write("TARGET N=295 current50=55/171/62/7 peak50=59/110/96/30 peak100=114/93/42/46\n\n")
        f.write("TOP30\n")
        f.write(r.head(30).to_string(index=False))
        f.write("\n")
    print((OUT_DIR/"dedup_summary.txt").read_text(encoding="utf-8"))

if __name__=="__main__": main()
