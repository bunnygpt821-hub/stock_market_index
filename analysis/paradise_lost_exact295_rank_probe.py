#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'runtime/nikkei225_yahoo_clean_2016_present.csv'
OUT=ROOT/'analysis_outputs/paradise_lost_exact295_rank'; OUT.mkdir(parents=True,exist_ok=True)
TC=np.array([55,171,62,7]); TP50=np.array([59,110,96,30]); TP100=np.array([114,93,42,46])

def qbins(a,e):
 a=np.asarray(a,float);a=a[np.isfinite(a)]
 return np.array([(a<e[0]).sum(),((a>=e[0])&(a<e[1])).sum(),((a>=e[1])&(a<e[2])).sum(),(a>=e[2]).sum()])
def txt(a):return '/'.join(map(str,a.astype(int)))

def prep(g):
 d=g.sort_values('Date').copy().reset_index(drop=True)
 # keep both raw and repaired H/L variants
 d['Hraw']=d.High; d['Lraw']=d.Low
 d['Hrep']=d[['Open','High','Close']].max(axis=1); d['Lrep']=d[['Open','Low','Close']].min(axis=1)
 pc=d.Close.shift(1)
 for tag,H,L in [('raw',d.Hraw,d.Lraw),('rep',d.Hrep,d.Lrep)]:
  tr=pd.concat([H-L,(H-pc).abs(),(L-pc).abs()],axis=1).max(axis=1)
  d[f'ATR_{tag}']=tr.rolling(20).mean(); d[f'ATRprev_{tag}']=d[f'ATR_{tag}'].shift(1)
  h60i=H.rolling(60,min_periods=60).max(); h60e=H.shift(1).rolling(60,min_periods=60).max()
  l60i=L.rolling(60,min_periods=60).min(); l60e=L.shift(1).rolling(60,min_periods=60).min()
  c60i=d.Close.rolling(60,min_periods=60).max(); c60e=d.Close.shift(1).rolling(60,min_periods=60).max()
  d[f'prox_H_H60_{tag}']=(h60i-H)/h60i
  d[f'prox_C_H60_{tag}']=(h60i-d.Close)/h60i
  d[f'prox_PC_H60e_{tag}']=(h60e-d.Close.shift(1))/h60e
  d[f'prox_PH_H60e_{tag}']=(h60e-H.shift(1))/h60e
  d[f'dist_H_H60_ATR_{tag}']=(h60i-H)/d[f'ATR_{tag}']
  d[f'dist_C_H60_ATR_{tag}']=(h60i-d.Close)/d[f'ATR_{tag}']
  d[f'dist_H_H60_prevATR_{tag}']=(h60i-H)/d[f'ATRprev_{tag}']
  d[f'dist_PC_H60e_prevATR_{tag}']=(h60e-d.Close.shift(1))/d[f'ATRprev_{tag}']
  d[f'range_H60_{tag}']=(H-l60i)/(h60i-l60i)
  d[f'range_C60_{tag}']=(d.Close-l60i)/(h60i-l60i)
  # historical maximum closeness/highness over several lookbacks; larger is higher-zone score
  ratio=H/h60i
  for Lb in (2,3,5,7,10,15,20,30,40,60): d[f'histHratio_{tag}_{Lb}']=ratio.shift(1).rolling(Lb,min_periods=1).max()
 # MAs use close only
 for n in (15,50,100): d[f'MA{n}']=d.Close.rolling(n).mean()
 d['dev50']=(d.Close/d.MA50-1)*100;d['dev100']=(d.Close/d.MA100-1)*100
 # several peak definitions
 d['p50_daily_inc']=d.dev50.rolling(20,min_periods=1).max(); d['p100_daily_inc']=d.dev100.rolling(20,min_periods=1).max()
 d['p50_daily_exc']=d.dev50.shift(1).rolling(20,min_periods=1).max(); d['p100_daily_exc']=d.dev100.shift(1).rolling(20,min_periods=1).max()
 # max close over lookback divided by signal-day MA (alternative implementation)
 d['p50_sigMA_inc']=(d.Close.rolling(20,min_periods=1).max()/d.MA50-1)*100
 d['p100_sigMA_inc']=(d.Close.rolling(20,min_periods=1).max()/d.MA100-1)*100
 d['p50_sigMA_exc']=(d.Close.shift(1).rolling(20,min_periods=1).max()/d.MA50-1)*100
 d['p100_sigMA_exc']=(d.Close.shift(1).rolling(20,min_periods=1).max()/d.MA100-1)*100
 # candle/break rules for raw and repaired
 for tag,H,L in [('raw',d.Hraw,d.Lraw),('rep',d.Hrep,d.Lrep)]:
  for atrn in (f'ATR_{tag}',f'ATRprev_{tag}'):
   body=d.Open-d.Close; pos=(d.Close-L)/(H-L).replace(0,np.nan)
   common=(d.Open>d.Close)&(body>=.8*d[atrn])&(pos<=.30)
   d[f'base_{tag}_{atrn}_cc']=common&(d.Close<d.MA15)&(d.Close.shift(1)>=d.MA15.shift(1))
   d[f'base_{tag}_{atrn}_lc']=common&(L<d.MA15)&(d.Close.shift(1)>=d.MA15.shift(1))
   for k in (2,3,5):
    ok=pd.Series(True,index=d.index)
    for j in range(1,k+1):ok&=d.Close.shift(j)>=d.MA15.shift(j)
    d[f'base_{tag}_{atrn}_c{k}']=common&ok&(d.Close<d.MA15)
 d['Ticker']=g.Ticker.iloc[0]
 return d

def evalset(e):
 cur=qbins(e.dev50,[0,5,10]); sc=int(np.abs(cur-TC).sum())
 best=None
 for pm in ('daily_inc','daily_exc','sigMA_inc','sigMA_exc'):
  p50=qbins(e[f'p50_{pm}'],[5,10,15]);p100=qbins(e[f'p100_{pm}'],[10,15,20]);sp=int(np.abs(p50-TP50).sum()+np.abs(p100-TP100).sum())
  z=(sc*10+sp,pm,p50,p100,sp)
  if best is None or z[0]<best[0]:best=z
 return sc*10+best[4],sc,best[4],txt(cur),txt(best[2]),txt(best[3]),best[1]

def main():
 d=pd.read_csv(SRC);d.Date=pd.to_datetime(d.Date)
 for c in ['Open','High','Low','Close','Volume']:d[c]=pd.to_numeric(d[c],errors='coerce')
 a=pd.concat([prep(g) for _,g in d.groupby('Ticker',sort=False)],ignore_index=True)
 basecols=[c for c in a.columns if c.startswith('base_')]
 # metrics: lower is more high-zone for prox/dist, higher is more high-zone for range/hist ratio
 metrics=[]
 for c in a.columns:
  if c.startswith(('prox_','dist_')):metrics.append((c,True))
  elif c.startswith(('range_','histHratio_')):metrics.append((c,False))
 rows=[]
 for bcname in basecols:
  b=a[a[bcname]].copy()
  if len(b)<295:continue
  for m,asc in metrics:
   z=b[np.isfinite(b[m])].sort_values(m,ascending=asc,kind='stable')
   if len(z)<295:continue
   e=z.head(295).copy();boundary=float(e[m].iloc[-1]);nextv=float(z[m].iloc[295]) if len(z)>295 else np.nan
   total,sc,sp,cur,p50,p100,pm=evalset(e)
   rows.append([bcname,m,'low' if asc else 'high',boundary,nextv,len(b),cur,p50,p100,sc,sp,total,pm])
 r=pd.DataFrame(rows,columns=['base','metric','direction','boundary_295','next_value','baseN','current50','peak50','peak100','score_current','score_peak','score_total','peak_mode'])
 r=r.sort_values(['score_total','score_current','score_peak'],kind='stable').reset_index(drop=True)
 r.to_csv(OUT/'rank295.csv',index=False);r.head(300).to_csv(OUT/'top300.csv',index=False)
 with (OUT/'summary.txt').open('w',encoding='utf-8') as f:
  f.write('Every row is forced to exactly N=295 by ranking the high-zone metric inside the base-rule universe.\nTARGET current50=55/171/62/7 peak50=59/110/96/30 peak100=114/93/42/46\n\nTOP120\n');f.write(r.head(120).to_string(index=False));f.write('\n')
 print((OUT/'summary.txt').read_text(encoding='utf-8'))
if __name__=='__main__':main()
