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
BASE_CANDS = ROOT / "analysis_outputs/paradise_lost/calibration_candidates.csv"

TARGET_N = 295
TARGET_CURRENT50 = np.array([55,171,62,7])
TARGET_PEAK50 = np.array([59,110,96,30])
TARGET_PEAK100 = np.array([114,93,42,46])


def add_seq(allrows: pd.DataFrame) -> pd.DataFrame:
    parts=[]
    for code,g in allrows.groupby("code",sort=False):
        p=base.PRICE_DIR/f"{code}.csv"
        ds=pd.to_datetime(pd.read_csv(p,usecols=["date"])["date"],errors="coerce").dropna().drop_duplicates().sort_values().reset_index(drop=True)
        mp={pd.Timestamp(v):i for i,v in enumerate(ds)}
        x=g.copy(); x["seq"]=x["date"].map(mp).astype("Int64")
        parts.append(x)
    return pd.concat(parts,ignore_index=True)


def dedup_indices(e: pd.DataFrame, gap: int, keep: str) -> np.ndarray:
    idx=[]; asc=(keep=="first")
    srt=e.sort_values(["code","seq"],ascending=[True,asc])
    for _,g in srt.groupby("code",sort=False):
        seq=g["seq"].to_numpy(dtype=np.int64); inds=g.index.to_numpy(dtype=np.int64)
        last=None
        for s,i in zip(seq,inds):
            if last is None or (s-last>gap if asc else last-s>gap):
                idx.append(i); last=int(s)
    return np.asarray(idx,dtype=np.int64)


def score(e: pd.DataFrame):
    cur=base.bin_current50(e["dev50"])
    p50i=base.bin_peak50(e["peak50_inc20"].dropna()); p100i=base.bin_peak100(e["peak100_inc20"].dropna())
    p50e=base.bin_peak50(e["peak50_exc20"].dropna()); p100e=base.bin_peak100(e["peak100_exc20"].dropna())
    inc=int(np.abs(p50i-TARGET_PEAK50).sum()+np.abs(p100i-TARGET_PEAK100).sum())
    exc=int(np.abs(p50e-TARGET_PEAK50).sum()+np.abs(p100e-TARGET_PEAK100).sum())
    if inc<=exc: p50,p100,ps,pm=p50i,p100i,inc,"include_signal"
    else: p50,p100,ps,pm=p50e,p100e,exc,"exclude_signal"
    cs=int(np.abs(cur-TARGET_CURRENT50).sum()); ns=abs(len(e)-TARGET_N)
    return ns*20+cs*10+ps,ns,cs,ps,base.arrtxt(cur),base.arrtxt(p50),base.arrtxt(p100),pm


def main():
    frames=[x for c in base.load_codes() if (x:=base.prepare_one(c)) is not None]
    allrows=add_seq(pd.concat(frames,ignore_index=True))
    cands=pd.read_csv(BASE_CANDS)
    cands=cands[(cands["N"]>=TARGET_N)&(cands["N"]<=500)&(cands["score_current"]<=160)].copy()
    cands["pre_score"]=20*(cands["N"]-TARGET_N).abs()+10*cands["score_current"]+cands["score_peak"]
    cands=cands.sort_values(["pre_score","score_current","score_peak"]).head(3000)

    rows=[]; cache={}
    for _,c in cands.iterrows():
        key=(c["base"],c["metric"],float(c["threshold"]))
        if key not in cache:
            b=allrows.loc[allrows[str(c["base"])]].copy()
            cache[key]=b.loc[b[str(c["metric"])]<=float(c["threshold"])].copy()
        raw=cache[key]
        for gap in (3,5,7,10,15):
            for keep in ("first","last"):
                inds=dedup_indices(raw,gap,keep); e=raw.loc[inds]
                if len(e)<220 or len(e)>380: continue
                total,ns,cs,ps,cur,p50,p100,pm=score(e)
                rows.append({"base":c["base"],"metric":c["metric"],"threshold":c["threshold"],"unit":c["threshold_unit"],
                    "dedup_gap":gap,"dedup_mode":keep,"raw_N":len(raw),"N":len(e),"current50_counts":cur,
                    "peak50_counts":p50,"peak100_counts":p100,"score_N":ns,"score_current":cs,"score_peak":ps,
                    "score_total":total,"peak_mode":pm})
    r=pd.DataFrame(rows).drop_duplicates().sort_values(["score_total","score_N","score_current","score_peak","N"],kind="stable").reset_index(drop=True)
    r.to_csv(OUT_DIR/"dedup_calibration_candidates.csv",index=False)
    r.head(200).to_csv(OUT_DIR/"dedup_calibration_top200.csv",index=False)
    exact=r[r["N"]==295]
    with (OUT_DIR/"dedup_summary.txt").open("w",encoding="utf-8") as f:
        f.write("TARGET N=295 current50=55/171/62/7 peak50=59/110/96/30 peak100=114/93/42/46\n")
        f.write(f"first_stage_candidates_tested={len(cands)} dedup_results={len(r)} exact_N295={len(exact)}\n\n")
        f.write("TOP40\n"); f.write(r.head(40).to_string(index=False)); f.write("\n\n")
        f.write("BEST EXACT N=295\n"); f.write(exact.head(30).to_string(index=False)); f.write("\n")
    print((OUT_DIR/"dedup_summary.txt").read_text(encoding="utf-8"))

if __name__=="__main__": main()
