#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'runtime/nikkei225_yahoo_clean_2016_present.csv'
OUT=ROOT/'analysis_outputs/paradise_lost_exactsource'
OUT.mkdir(parents=True,exist_ok=True)
TARGET_CUR=np.array([55,171,62,7])
TARGET_P50=np.array([59,110,96,30])
TARGET_P100=np.array([114,93,42,46])


def bins_cur(s):
    a=s.to_numpy(float); return np.array([np.sum(a<0),np.sum((a>=0)&(a<5)),np.sum((a>=5)&(a<10)),np.sum(a>=10)])
def bins_p50(s):
    a=s.to_numpy(float); return np.array([np.sum(a<5),np.sum((a>=5)&(a<10)),np.sum((a>=10)&(a<15)),np.sum(a>=15)])
def bins_p100(s):
    a=s.to_numpy(float); return np.array([np.sum(a<10),np.sum((a>=10)&(a<15)),np.sum((a>=15)&(a<20)),np.sum(a>=20)])
def txt(a): return '/'.join(str(int(x)) for x in a)

def prep_one(g):
    d=g.sort_values('Date').copy().reset_index(drop=True)
    # Same quality-repair convention used in prior backtests for Yahoo-adjusted OHLC.
    d['High']=d[['Open','High','Close']].max(axis=1)
    d['Low']=d[['Open','Low','Close']].min(axis=1)
    pc=d['Close'].shift(1)
    tr=pd.concat([d['High']-d['Low'],(d['High']-pc).abs(),(d['Low']-pc).abs()],axis=1).max(axis=1)
    d['atr20']=tr.rolling(20,min_periods=20).mean()
    for n in (15,25,50,100):
        d[f'ma{n}']=d['Close'].rolling(n,min_periods=n).mean()
        d[f'dev{n}']=(d['Close']/d[f'ma{n}']-1)*100
    # exact prior definition: each past day's price / that day's MA - 1, then max over 20 sessions.
    d['peak50_inc20']=d['dev50'].rolling(20,min_periods=1).max()
    d['peak100_inc20']=d['dev100'].rolling(20,min_periods=1).max()
    d['peak50_exc20']=d['dev50'].shift(1).rolling(20,min_periods=1).max()
    d['peak100_exc20']=d['dev100'].shift(1).rolling(20,min_periods=1).max()
    h60i=d['High'].rolling(60,min_periods=60).max(); h60e=d['High'].shift(1).rolling(60,min_periods=60).max()
    c60i=d['Close'].rolling(60,min_periods=60).max(); c60e=d['Close'].shift(1).rolling(60,min_periods=60).max()
    for name,num,den in [
        ('close_h60i_pct',d['Close'],h60i),('close_h60e_pct',d['Close'],h60e),
        ('high_h60i_pct',d['High'],h60i),('high_h60e_pct',d['High'],h60e),
        ('open_h60e_pct',d['Open'],h60e),('prevclose_h60e_pct',d['Close'].shift(1),h60e),
        ('prevhigh_h60e_pct',d['High'].shift(1),h60e),('close_c60i_pct',d['Close'],c60i),
        ('close_c60e_pct',d['Close'],c60e),('high_c60e_pct',d['High'],c60e)]:
        d[name]=(den-num)/den
    d['close_h60e_atr']=(h60e-d['Close'])/d['atr20']; d['high_h60e_atr']=(h60e-d['High'])/d['atr20']
    d['prevclose_h60e_atr']=(h60e-d['Close'].shift(1))/d['atr20']
    body=d['Open']-d['Close']; rng=d['High']-d['Low']; pos=(d['Close']-d['Low'])/rng.replace(0,np.nan)
    common=(d['Open']>d['Close'])&(body>=0.8*d['atr20'])&(pos<=0.30)&(d['Close']<d['ma15'])
    d['base_cross']=common&(d['Close'].shift(1)>=d['ma15'].shift(1))
    for k in (2,3,5,10):
        ok=pd.Series(True,index=d.index)
        for j in range(1,k+1): ok &= d['Close'].shift(j)>=d['ma15'].shift(j)
        d[f'base_above{k}']=common&ok
    d['seq']=np.arange(len(d),dtype=int)
    return d

def score(e):
    cur=bins_cur(e['dev50']); cs=int(np.abs(cur-TARGET_CUR).sum())
    p50i=bins_p50(e['peak50_inc20']); p100i=bins_p100(e['peak100_inc20']); inc=int(np.abs(p50i-TARGET_P50).sum()+np.abs(p100i-TARGET_P100).sum())
    p50e=bins_p50(e['peak50_exc20']); p100e=bins_p100(e['peak100_exc20']); exc=int(np.abs(p50e-TARGET_P50).sum()+np.abs(p100e-TARGET_P100).sum())
    if inc<=exc: p50,p100,ps,pm=p50i,p100i,inc,'include_signal'
    else: p50,p100,ps,pm=p50e,p100e,exc,'exclude_signal'
    ns=abs(len(e)-295); total=ns*20+cs*10+ps
    return total,ns,cs,ps,txt(cur),txt(p50),txt(p100),pm

def dedup(e,gap=10,keep='first'):
    rows=[]; asc=keep=='first'
    for _,g in e.sort_values(['Ticker','seq'],ascending=[True,asc]).groupby('Ticker',sort=False):
        last=None
        for r in g.itertuples():
            s=int(r.seq)
            if last is None or (s-last>gap if asc else last-s>gap): rows.append(r.Index); last=s
    return e.loc[rows]

def main():
    d=pd.read_csv(SRC)
    d['Date']=pd.to_datetime(d['Date'])
    for c in ['Open','High','Low','Close','Volume']: d[c]=pd.to_numeric(d[c],errors='coerce')
    d=d.dropna(subset=['Ticker','Date','Open','High','Low','Close'])
    allrows=pd.concat([prep_one(g) for _,g in d.groupby('Ticker',sort=False)],ignore_index=True)
    metrics=[c for c in allrows.columns if c.endswith('_pct')]
    atrmetrics=[c for c in allrows.columns if c.endswith('_atr')]
    basemodes=['base_cross','base_above2','base_above3','base_above5','base_above10']
    rows=[]
    for bm in basemodes:
        b=allrows[allrows[bm]].copy()
        for m in metrics:
            for th in np.arange(0,0.1501,0.0005):
                raw=b[b[m]<=th]
                if not 200<=len(raw)<=450: continue
                for dd in ('none','first10','last10'):
                    e=raw if dd=='none' else dedup(raw,10,'first' if dd=='first10' else 'last')
                    if not 220<=len(e)<=380: continue
                    total,ns,cs,ps,cur,p50,p100,pm=score(e)
                    rows.append([bm,m,round(float(th),4),'fraction',dd,len(raw),len(e),cur,p50,p100,ns,cs,ps,total,pm])
        for m in atrmetrics:
            for th in np.arange(0,3.001,0.025):
                raw=b[b[m]<=th]
                if not 200<=len(raw)<=450: continue
                for dd in ('none','first10','last10'):
                    e=raw if dd=='none' else dedup(raw,10,'first' if dd=='first10' else 'last')
                    if not 220<=len(e)<=380: continue
                    total,ns,cs,ps,cur,p50,p100,pm=score(e)
                    rows.append([bm,m,round(float(th),3),'ATR',dd,len(raw),len(e),cur,p50,p100,ns,cs,ps,total,pm])
    cols=['base','metric','threshold','unit','dedup','raw_N','N','current50_counts','peak50_counts','peak100_counts','score_N','score_current','score_peak','score_total','peak_mode']
    r=pd.DataFrame(rows,columns=cols).sort_values(['score_total','score_N','score_current','score_peak','N'],kind='stable').reset_index(drop=True)
    r.to_csv(OUT/'calibration_candidates.csv',index=False)
    r.head(200).to_csv(OUT/'calibration_top200.csv',index=False)
    with (OUT/'summary.txt').open('w',encoding='utf-8') as f:
        f.write(f"source_rows={len(d)} tickers={d['Ticker'].nunique()} first={d['Date'].min().date()} last={d['Date'].max().date()}\n")
        f.write('TARGET N=295 current50=55/171/62/7 peak50=59/110/96/30 peak100=114/93/42/46\n\n')
        f.write('TOP50\n'); f.write(r.head(50).to_string(index=False)); f.write('\n')
    print((OUT/'summary.txt').read_text(encoding='utf-8'))

if __name__=='__main__': main()
