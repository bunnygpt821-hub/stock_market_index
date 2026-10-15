#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'analysis_outputs/paradise_lost_neck_retest_failure/03_oos_results.csv'
OUT=ROOT/'analysis_outputs/paradise_lost_neck_retest_failure/05_robust_rank.csv'
SUM=ROOT/'analysis_outputs/paradise_lost_neck_retest_failure/robust_summary.txt'

d=pd.read_csv(SRC)
keys=['Hold','Pattern','NeckType','Trigger','IndexRule']
rows=[]
for k,g in d.groupby(keys,dropna=False):
    # exactly the three predeclared periods
    if g['Period'].nunique()!=3: continue
    pfs={r.Period:float(r.PF) for _,r in g.iterrows()}
    ns={r.Period:int(r.N) for _,r in g.iterrows()}
    means={r.Period:float(r.MeanR) for _,r in g.iterrows()}
    vals=np.array(list(pfs.values()),float)
    if not np.all(np.isfinite(vals)): continue
    rows.append({**dict(zip(keys,k)),
      'PF_train':pfs.get('TRAIN_2016_2020',np.nan),'PF_valid':pfs.get('VALID_2021_2023',np.nan),'PF_test':pfs.get('TEST_2024_2026',np.nan),
      'N_train':ns.get('TRAIN_2016_2020',0),'N_valid':ns.get('VALID_2021_2023',0),'N_test':ns.get('TEST_2024_2026',0),
      'MeanR_train':means.get('TRAIN_2016_2020',np.nan),'MeanR_valid':means.get('VALID_2021_2023',np.nan),'MeanR_test':means.get('TEST_2024_2026',np.nan),
      'MinPF':float(np.min(vals)),'MedianPF':float(np.median(vals)),'MeanPF':float(np.mean(vals)),
      'MinN':min(ns.values()),'AllPFgt1':bool(np.all(vals>1.0)),'AllPFgt09':bool(np.all(vals>0.9))})
r=pd.DataFrame(rows)
r['RobustScore']=r['MinPF']
r=r.sort_values(['RobustScore','MinN','MedianPF'],ascending=[False,False,False]).reset_index(drop=True)
r.to_csv(OUT,index=False)

# Also rank only reasonably populated cells to avoid tiny-N winners.
r20=r[r.MinN>=20].copy(); r50=r[r.MinN>=50].copy()
lines=['PARADISE LOST — RETEST FAILURE ROBUSTNESS RANK','Rank criterion = maximum worst-period PF across 2016-20 / 2021-23 / 2024-26.','Do not select by test-period maximum.','','TOP 25, MinN>=20',r20.head(25).round(3).to_string(index=False),'','TOP 25, MinN>=50',r50.head(25).round(3).to_string(index=False),'',f'All three PF>1, MinN>=20: {int((r20.AllPFgt1).sum())}',f'All three PF>0.9, MinN>=20: {int((r20.AllPFgt09).sum())}']
SUM.write_text('\n'.join(lines),encoding='utf-8')
print(SUM.read_text(encoding='utf-8'))
