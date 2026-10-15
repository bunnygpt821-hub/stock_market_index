#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'analysis_outputs/paradise_lost_neck_retest_failure/01_retest_events.csv'
OUTDIR=ROOT/'analysis_outputs/paradise_lost_survivor_validate'
OUTDIR.mkdir(parents=True,exist_ok=True)

rng=np.random.default_rng(821)

def pf(x):
    r=pd.to_numeric(x,errors='coerce').dropna();gp=r[r>0].sum();gl=-r[r<0].sum()
    if gl==0:return float('inf') if gp>0 else np.nan
    return float(gp/gl)

def maxdd_r(x):
    a=np.asarray(pd.to_numeric(x,errors='coerce').dropna(),float)
    if len(a)==0:return np.nan
    eq=np.cumsum(a);peak=np.maximum.accumulate(np.r_[0,eq]);eq2=np.r_[0,eq]
    return float(np.max(peak-eq2))

def stats(g):
    if len(g)==0:return {'N':0,'MeanR':np.nan,'MedianR':np.nan,'PF':np.nan,'WinRate':np.nan,'StopRate':np.nan,'Hit1R':np.nan,'Hit2R':np.nan,'MaxDD_R':np.nan}
    return {'N':len(g),'MeanR':g.R.mean(),'MedianR':g.R.median(),'PF':pf(g.R),'WinRate':(g.R>0).mean(),'StopRate':g.stopped.mean(),'Hit1R':g.hit1.mean(),'Hit2R':g.hit2.mean(),'MaxDD_R':maxdd_r(g.sort_values('TriggerDate').R)}

def bootstrap(r,nboot=10000):
    a=np.asarray(pd.to_numeric(r,errors='coerce').dropna(),float);n=len(a)
    means=[];pfs=[]
    for _ in range(nboot):
        s=a[rng.integers(0,n,n)]
        means.append(float(np.mean(s)));pfs.append(pf(pd.Series(s)))
    return {'MeanR_lo':np.quantile(means,.025),'MeanR_hi':np.quantile(means,.975),'PF_lo':np.quantile(pfs,.025),'PF_hi':np.quantile(pfs,.975),'ProbMeanPos':np.mean(np.array(means)>0),'ProbPFgt1':np.mean(np.array(pfs)>1)}

d=pd.read_csv(SRC)
d['ADate']=pd.to_datetime(d.ADate);d['TriggerDate']=pd.to_datetime(d.TriggerDate)
# Sole survivor from robustness screen:
# B is 1-3% below A, two-valley sloping neckline (HS_LINE), neckline break, Nikkei bear stack, hold 10.
z=d[(d.Hold==10)&(d.NeckType=='HS_LINE')&(d.Trigger=='NECK_BREAK')&(d.BGapAbs>=1)&(d.BGapAbs<3)&(d.IDX_BEARSTACK.astype(bool))].copy()
z=z.sort_values(['TriggerDate','Ticker']).reset_index(drop=True)
z.to_csv(OUTDIR/'01_survivor_trades.csv',index=False)

rows=[]
for y,g in z.groupby(z.TriggerDate.dt.year):
    rows.append({'Year':int(y),**stats(g)})
yr=pd.DataFrame(rows);yr.to_csv(OUTDIR/'02_yearly.csv',index=False)

periods=[('2016_2020','2016-01-01','2020-12-31'),('2021_2023','2021-01-01','2023-12-31'),('2024_2026','2024-01-01','2026-09-18'),('ALL','2016-01-01','2026-09-18')]
prows=[]
for name,a,b in periods:
    g=z[(z.ADate>=a)&(z.ADate<=b)]
    rec={'Period':name,**stats(g)}
    if len(g)>=20: rec.update(bootstrap(g.R,10000))
    prows.append(rec)
pr=pd.DataFrame(prows);pr.to_csv(OUTDIR/'03_period_bootstrap.csv',index=False)

# Sensitivity around B-gap and index filter, without picking the best test result.
sens=[]
for bglo,bghi,label in [(0,1,'0_1'),(1,2,'1_2'),(2,3,'2_3'),(1,3,'1_3'),(0,3,'0_3'),(3,5,'3_5')]:
    for idxname,idxmask in [('BEARSTACK',d.IDX_BEARSTACK.astype(bool)),('IDX25_WEAK',d.IDX_WEAK25.astype(bool)),('ANY',pd.Series(True,index=d.index))]:
        q=d[(d.Hold==10)&(d.NeckType=='HS_LINE')&(d.Trigger=='NECK_BREAK')&(d.BGapAbs>=bglo)&(d.BGapAbs<bghi)&idxmask].copy()
        for name,a,b in periods[:3]:
            g=q[(q.ADate>=a)&(q.ADate<=b)]
            sens.append({'BGap':label,'IndexRule':idxname,'Period':name,**stats(g)})
sens=pd.DataFrame(sens);sens.to_csv(OUTDIR/'04_sensitivity.csv',index=False)

lines=['PARADISE LOST — SOLE ROBUST SURVIVOR VALIDATION',
       'Rule: Hold10, B 1-3% below A, HS_LINE neckline break, Nikkei bear stack (5MA<15MA<25MA and close<5MA), 1ATR stop.',
       'This rule was selected because it was the only MinN>=20 configuration with PF>1 in all three predeclared periods; therefore these diagnostics are post-selection and must be treated conservatively.','',
       'YEARLY',yr.round(3).to_string(index=False),'','PERIOD + BOOTSTRAP',pr.round(3).to_string(index=False),'','SENSITIVITY',sens.round(3).to_string(index=False)]
(OUTDIR/'summary.txt').write_text('\n'.join(lines),encoding='utf-8')
print((OUTDIR/'summary.txt').read_text(encoding='utf-8'))
