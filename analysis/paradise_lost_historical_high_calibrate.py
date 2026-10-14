#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'runtime/nikkei225_yahoo_clean_2016_present.csv'
OUT=ROOT/'analysis_outputs/paradise_lost_historical_high'
OUT.mkdir(parents=True,exist_ok=True)
T_CUR=np.array([55,171,62,7]); T_P50=np.array([59,110,96,30]); T_P100=np.array([114,93,42,46])

def bc(s):
 a=s.to_numpy(float); return np.array([np.sum(a<0),np.sum((a>=0)&(a<5)),np.sum((a>=5)&(a<10)),np.sum(a>=10)])
def b50(s):
 a=s.to_numpy(float); return np.array([np.sum(a<5),np.sum((a>=5)&(a<10)),np.sum((a>=10)&(a<15)),np.sum(a>=15)])
def b100(s):
 a=s.to_numpy(float); return np.array([np.sum(a<10),np.sum((a>=10)&(a<15)),np.sum((a>=15)&(a<20)),np.sum(a>=20)])
def st(a): return '/'.join(map(str,a.astype(int)))

def prep(g):
 d=g.sort_values('Date').copy().reset_index(drop=True)
 d['High']=d[['Open','High','Close']].max(axis=1); d['Low']=d[['Open','Low','Close']].min(axis=1)
 pc=d.Close.shift(1); tr=pd.concat([d.High-d.Low,(d.High-pc).abs(),(d.Low-pc).abs()],axis=1).max(axis=1); d['ATR20']=tr.rolling(20).mean()
 for n in (15,25,50,100):
  d[f'MA{n}']=d.Close.rolling(n).mean(); d[f'dev{n}']=(d.Close/d[f'MA{n}']-1)*100
 d['p50i']=d.dev50.rolling(20,min_periods=1).max(); d['p100i']=d.dev100.rolling(20,min_periods=1).max()
 d['p50e']=d.dev50.shift(1).rolling(20,min_periods=1).max(); d['p100e']=d.dev100.shift(1).rolling(20,min_periods=1).max()
 h60=d.High.rolling(60,min_periods=60).max(); c60=d.Close.rolling(60,min_periods=60).max()
 d['high_ratio60']=d.High/h60; d['close_ratio60']=d.Close/c60
 body=d.Open-d.Close; pos=(d.Close-d.Low)/(d.High-d.Low).replace(0,np.nan)
 common=(d.Open>d.Close)&(body>=.8*d.ATR20)&(pos<=.30)&(d.Close<d.MA15)
 d['base_cross']=common&(d.Close.shift(1)>=d.MA15.shift(1))
 for k in (2,3,5,10,15,20):
  ok=pd.Series(True,index=d.index)
  for j in range(1,k+1): ok &= d.Close.shift(j)>=d.MA15.shift(j)
  d[f'base{k}']=common&ok
 d['Ticker']=g['Ticker'].iloc[0]
 return d

def score(e):
 cur=bc(e.dev50); cs=int(np.abs(cur-T_CUR).sum())
 i50=b50(e.p50i);i100=b100(e.p100i);e50=b50(e.p50e);e100=b100(e.p100e)
 si=int(np.abs(i50-T_P50).sum()+np.abs(i100-T_P100).sum()); se=int(np.abs(e50-T_P50).sum()+np.abs(e100-T_P100).sum())
 if si<=se: p50,p100,ps,pm=i50,i100,si,'inc'
 else: p50,p100,ps,pm=e50,e100,se,'exc'
 ns=abs(len(e)-295); return ns*25+cs*12+ps,ns,cs,ps,st(cur),st(p50),st(p100),pm

def main():
 d=pd.read_csv(SRC); d['Date']=pd.to_datetime(d.Date)
 for c in ['Open','High','Low','Close','Volume']: d[c]=pd.to_numeric(d[c],errors='coerce')
 allrows=pd.concat([prep(g) for _,g in d.groupby('Ticker',sort=False)],ignore_index=True)
 rows=[]
 for bm in ['base_cross','base2','base3','base5','base10','base15','base20']:
  b=allrows[allrows[bm]].copy()
  for src in ['high_ratio60','close_ratio60']:
   for include_signal in [True,False]:
    x=b[src] if include_signal else b[src].where(False)  # placeholder; rolling recomputed below from allrows slices
    # compute historical-zone feature within ticker on full timeline, then select base rows
    for L in (1,2,3,5,7,10,15,20,30,40,60):
     for pct in (0.0,0.0025,0.005,0.0075,0.01,0.015,0.02,0.03,0.04,0.05,0.075,0.10,0.15):
      # threshold means price was >= (1-pct)*60d high sometime in lookback
      mask_parts=[]
      for ticker,g in allrows.groupby('Ticker',sort=False):
       z=g[src].shift(0 if include_signal else 1).rolling(L,min_periods=1).max() >= (1-pct)
       mask_parts.append(pd.Series(z.to_numpy(),index=g.index))
      hist=pd.concat(mask_parts).sort_index()
      e=b[hist.loc[b.index].to_numpy()]
      if not 200<=len(e)<=420: continue
      total,ns,cs,ps,cur,p50,p100,pm=score(e)
      rows.append([bm,src,include_signal,L,pct,len(e),cur,p50,p100,ns,cs,ps,total,pm])
 r=pd.DataFrame(rows,columns=['base','zone_source','include_signal','lookback','zone_pct','N','current50','peak50','peak100','score_N','score_current','score_peak','score_total','peak_mode'])
 r=r.sort_values(['score_total','score_N','score_current','score_peak','N'],kind='stable').reset_index(drop=True)
 r.to_csv(OUT/'candidates.csv',index=False); r.head(200).to_csv(OUT/'top200.csv',index=False)
 with (OUT/'summary.txt').open('w',encoding='utf-8') as f:
  f.write('TARGET N=295 current50=55/171/62/7 peak50=59/110/96/30 peak100=114/93/42/46\n\n')
  f.write(r.head(80).to_string(index=False)); f.write('\n')
 print((OUT/'summary.txt').read_text(encoding='utf-8'))

if __name__=='__main__': main()
