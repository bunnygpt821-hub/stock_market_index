#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'runtime/nikkei225_yahoo_clean_2016_present.csv'
OUT=ROOT/'analysis_outputs/paradise_lost_timing_probe'; OUT.mkdir(parents=True,exist_ok=True)
T=np.array([55,171,62,7]); P50=np.array([59,110,96,30]); P100=np.array([114,93,42,46])

def bins(a):
 a=np.asarray(a,float); a=a[np.isfinite(a)]; return np.array([(a<0).sum(),((a>=0)&(a<5)).sum(),((a>=5)&(a<10)).sum(),(a>=10).sum()])
def bp50(a):
 a=np.asarray(a,float);a=a[np.isfinite(a)];return np.array([(a<5).sum(),((a>=5)&(a<10)).sum(),((a>=10)&(a<15)).sum(),(a>=15).sum()])
def bp100(a):
 a=np.asarray(a,float);a=a[np.isfinite(a)];return np.array([(a<10).sum(),((a>=10)&(a<15)).sum(),((a>=15)&(a<20)).sum(),(a>=20).sum()])
def s(a): return '/'.join(map(str,a.astype(int)))

def prep(g):
 d=g.sort_values('Date').copy().reset_index(drop=True)
 d['High']=d[['Open','High','Close']].max(axis=1);d['Low']=d[['Open','Low','Close']].min(axis=1)
 pc=d.Close.shift(1);tr=pd.concat([d.High-d.Low,(d.High-pc).abs(),(d.Low-pc).abs()],axis=1).max(axis=1);d['ATR20']=tr.rolling(20).mean()
 for n in (15,25,50,100): d[f'MA{n}']=d.Close.rolling(n).mean()
 d['dev50_sig']=(d.Close/d.MA50-1)*100; d['dev100_sig']=(d.Close/d.MA100-1)*100
 d['dev50_prev']=(d.Close.shift(1)/d.MA50.shift(1)-1)*100
 d['p50_inc']=d.dev50_sig.rolling(20,min_periods=1).max(); d['p50_exc']=d.dev50_sig.shift(1).rolling(20,min_periods=1).max()
 d['p100_inc']=d.dev100_sig.rolling(20,min_periods=1).max(); d['p100_exc']=d.dev100_sig.shift(1).rolling(20,min_periods=1).max()
 # entry day measurements
 d['next_open']=d.Open.shift(-1);d['next_close']=d.Close.shift(-1);d['next_ma50']=d.MA50.shift(-1);d['next_ma100']=d.MA100.shift(-1)
 d['dev50_entryopen_sigMA']=(d.next_open/d.MA50-1)*100
 d['dev50_entryopen_nextMA']=(d.next_open/d.next_ma50-1)*100
 d['dev50_entryclose_nextMA']=(d.next_close/d.next_ma50-1)*100
 d['dev100_entryopen_sigMA']=(d.next_open/d.MA100-1)*100
 d['dev100_entryopen_nextMA']=(d.next_open/d.next_ma100-1)*100
 d['dev100_entryclose_nextMA']=(d.next_close/d.next_ma100-1)*100
 h60i=d.High.rolling(60,min_periods=60).max(); h60e=d.High.shift(1).rolling(60,min_periods=60).max()
 d['high_h60i']=(h60i-d.High)/h60i;d['high_h60e']=(h60e-d.High)/h60e
 d['prevclose_h60e']=(h60e-d.Close.shift(1))/h60e
 body=d.Open-d.Close;pos=(d.Close-d.Low)/(d.High-d.Low).replace(0,np.nan)
 common=(d.Open>d.Close)&(body>=.8*d.ATR20)&(pos<=.30)&(d.Close<d.MA15)
 d['cross']=common&(d.Close.shift(1)>=d.MA15.shift(1))
 for k in (2,3,5):
  ok=pd.Series(True,index=d.index)
  for j in range(1,k+1):ok &= d.Close.shift(j)>=d.MA15.shift(j)
  d[f'above{k}']=common&ok
 d['Ticker']=g.Ticker.iloc[0];d['idx']=np.arange(len(d));return d

def main():
 d=pd.read_csv(SRC);d.Date=pd.to_datetime(d.Date)
 for c in ['Open','High','Low','Close','Volume']:d[c]=pd.to_numeric(d[c],errors='coerce')
 a=pd.concat([prep(g) for _,g in d.groupby('Ticker',sort=False)],ignore_index=True)
 candidates=[
  ('N295_above2_high_h60i_0p45', (a.above2)&(a.high_h60i<=.0045)),
  ('above3_prevclose_h60e_0p70',(a.above3)&(a.prevclose_h60e<=.0070)),
  ('cross_prevclose_h60e_0p60',(a['cross'])&(a.prevclose_h60e<=.0060)),
  ('above2_high_h60e_0p45',(a.above2)&(a.high_h60e<=.0045)),
 ]
 rows=[]
 curcols=['dev50_sig','dev50_prev','dev50_entryopen_sigMA','dev50_entryopen_nextMA','dev50_entryclose_nextMA']
 pcols=[('p50_inc','p100_inc'),('p50_exc','p100_exc')]
 for name,mask in candidates:
  e=a[mask].copy()
  for cur in curcols:
   bc=bins(e[cur]);sc=int(np.abs(bc-T).sum())
   for p50,p100 in pcols:
    b50=bp50(e[p50]);b100=bp100(e[p100]);sp=int(np.abs(b50-P50).sum()+np.abs(b100-P100).sum())
    rows.append([name,len(e),cur,p50.replace('p50_',''),s(bc),s(b50),s(b100),sc,sp,sc*10+sp])
 r=pd.DataFrame(rows,columns=['candidate','N','current_timing','peak_mode','current50','peak50','peak100','score_current','score_peak','score_total']).sort_values(['score_total','score_current','score_peak'])
 r.to_csv(OUT/'timing_probe.csv',index=False)
 with (OUT/'summary.txt').open('w',encoding='utf-8') as f:
  f.write('TARGET current50=55/171/62/7 peak50=59/110/96/30 peak100=114/93/42/46\n\n');f.write(r.to_string(index=False));f.write('\n')
 print((OUT/'summary.txt').read_text(encoding='utf-8'))
if __name__=='__main__':main()
