#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'runtime/nikkei225_yahoo_clean_2016_present.csv'
OUT=ROOT/'analysis_outputs/paradise_lost_definition_search'; OUT.mkdir(parents=True,exist_ok=True)
TC=np.array([55,171,62,7]); TP50=np.array([59,110,96,30]); TP100=np.array([114,93,42,46])

def bin4(a,edges):
 a=np.asarray(a,float);a=a[np.isfinite(a)]
 return np.array([(a<edges[0]).sum(),((a>=edges[0])&(a<edges[1])).sum(),((a>=edges[1])&(a<edges[2])).sum(),(a>=edges[2]).sum()])
def txt(a):return '/'.join(map(str,a.astype(int)))
def score(e):
 c=bin4(e.dev50,[0,5,10]);p50=bin4(e.p50,[5,10,15]);p100=bin4(e.p100,[10,15,20])
 sc=int(np.abs(c-TC).sum());sp=int(np.abs(p50-TP50).sum()+np.abs(p100-TP100).sum());sn=abs(len(e)-295)
 return sn*30+sc*15+sp,sn,sc,sp,txt(c),txt(p50),txt(p100)

def prep(g):
 d=g.sort_values('Date').copy().reset_index(drop=True)
 d['High']=d[['Open','High','Close']].max(axis=1);d['Low']=d[['Open','Low','Close']].min(axis=1)
 pc=d.Close.shift(1);tr=pd.concat([d.High-d.Low,(d.High-pc).abs(),(d.Low-pc).abs()],axis=1).max(axis=1);d['ATR20']=tr.rolling(20).mean();d['ATR20_prev']=d.ATR20.shift(1)
 for n in (15,50,100):d[f'MA{n}']=d.Close.rolling(n).mean()
 d['dev50']=(d.Close/d.MA50-1)*100;d['dev100']=(d.Close/d.MA100-1)*100
 # default Peak: daily close divergence, prior 20 incl signal
 d['p50']=d.dev50.rolling(20,min_periods=1).max();d['p100']=d.dev100.rolling(20,min_periods=1).max()
 h60i=d.High.rolling(60,min_periods=60).max(); h60e=d.High.shift(1).rolling(60,min_periods=60).max()
 c60i=d.Close.rolling(60,min_periods=60).max(); c60e=d.Close.shift(1).rolling(60,min_periods=60).max()
 l60i=d.Low.rolling(60,min_periods=60).min(); l60e=d.Low.shift(1).rolling(60,min_periods=60).min()
 d['prox_high_hi']=(h60i-d.High)/h60i;d['prox_close_hi']=(h60i-d.Close)/h60i;d['prox_prevclose_hi']=(h60e-d.Close.shift(1))/h60e
 d['prox_high_closehi']=(c60i-d.High)/c60i;d['prox_close_closehi']=(c60i-d.Close)/c60i
 d['range_close_i']=(d.Close-l60i)/(h60i-l60i);d['range_close_e']=(d.Close-l60e)/(h60e-l60e)
 d['range_high_i']=(d.High-l60i)/(h60i-l60i)
 d['dist_close_atr_sig']=(h60i-d.Close)/d.ATR20;d['dist_high_atr_sig']=(h60i-d.High)/d.ATR20
 d['dist_close_atr_prev']=(h60e-d.Close)/d.ATR20_prev;d['dist_high_atr_prev']=(h60e-d.High)/d.ATR20_prev
 body=d.Open-d.Close; rng=(d.High-d.Low).replace(0,np.nan); pos=(d.Close-d.Low)/rng
 d['bear']=d.Open>d.Close;d['body']=body;d['pos']=pos
 # Break variants
 d['br_close_prevclose']=(d.Close<d.MA15)&(d.Close.shift(1)>=d.MA15.shift(1))
 d['br_close_prevlow']=(d.Close<d.MA15)&(d.Low.shift(1)>=d.MA15.shift(1))
 d['br_low_prevclose']=(d.Low<d.MA15)&(d.Close.shift(1)>=d.MA15.shift(1))
 d['br_low_prevlow']=(d.Low<d.MA15)&(d.Low.shift(1)>=d.MA15.shift(1))
 d['br_intraday']=(d.Open>=d.MA15)&(d.Close<d.MA15)
 for k in (2,3,5):
  okc=pd.Series(True,index=d.index);okl=pd.Series(True,index=d.index)
  for j in range(1,k+1):
   okc &= d.Close.shift(j)>=d.MA15.shift(j);okl &= d.Low.shift(j)>=d.MA15.shift(j)
  d[f'br_close_above{k}']=okc&(d.Close<d.MA15);d[f'br_low_above{k}']=okl&(d.Low<d.MA15)
 return d

def main():
 d=pd.read_csv(SRC);d.Date=pd.to_datetime(d.Date)
 for c in ['Open','High','Low','Close','Volume']:d[c]=pd.to_numeric(d[c],errors='coerce')
 a=pd.concat([prep(g) for _,g in d.groupby('Ticker',sort=False)],ignore_index=True)
 breakcols=['br_close_prevclose','br_close_prevlow','br_low_prevclose','br_low_prevlow','br_intraday','br_close_above2','br_close_above3','br_close_above5','br_low_above2','br_low_above3','br_low_above5']
 high_specs=[]
 for m in ['prox_high_hi','prox_close_hi','prox_prevclose_hi','prox_high_closehi','prox_close_closehi']:
  high_specs += [(m,'le',float(x)) for x in np.arange(0,0.2001,0.0025)]
 for m in ['range_close_i','range_close_e','range_high_i']:
  high_specs += [(m,'ge',float(x)) for x in np.arange(0.50,0.991,0.01)]
 for m in ['dist_close_atr_sig','dist_high_atr_sig','dist_close_atr_prev','dist_high_atr_prev']:
  high_specs += [(m,'le',float(x)) for x in np.arange(0,8.001,0.10)]
 rows=[]
 for atrref in ['ATR20','ATR20_prev']:
  candle=a.bear&(a.body>=0.8*a[atrref])&(a.pos<=.30)
  for br in breakcols:
   base=candle&a[br]
   b=a[base]
   if len(b)<200:continue
   for m,op,th in high_specs:
    e=b[b[m]<=th] if op=='le' else b[b[m]>=th]
    if not 220<=len(e)<=380:continue
    total,sn,sc,sp,cur,p50,p100=score(e)
    rows.append([atrref,br,m,op,round(th,4),len(e),cur,p50,p100,sn,sc,sp,total])
 r=pd.DataFrame(rows,columns=['atr_ref','break_rule','high_metric','op','threshold','N','current50','peak50','peak100','score_N','score_current','score_peak','score_total'])
 r=r.sort_values(['score_total','score_N','score_current','score_peak','N'],kind='stable').reset_index(drop=True)
 r.to_csv(OUT/'candidates.csv',index=False);r.head(300).to_csv(OUT/'top300.csv',index=False)
 exact=r[r.N==295]
 with (OUT/'summary.txt').open('w',encoding='utf-8') as f:
  f.write('TARGET N=295 current50=55/171/62/7 peak50=59/110/96/30 peak100=114/93/42/46\n')
  f.write(f'candidates={len(r)} exactN={len(exact)}\n\nTOP100\n');f.write(r.head(100).to_string(index=False));f.write('\n\nEXACT N295 TOP100\n');f.write(exact.head(100).to_string(index=False));f.write('\n')
 print((OUT/'summary.txt').read_text(encoding='utf-8'))
if __name__=='__main__':main()
