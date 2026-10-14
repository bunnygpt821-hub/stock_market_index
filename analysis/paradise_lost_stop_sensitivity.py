#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'runtime/nikkei225_yahoo_clean_2016_present.csv'
BASE=Path(__file__).with_name('paradise_lost_v2_analysis.py')
OUT=ROOT/'analysis_outputs/paradise_lost_stop_sensitivity'
OUT.mkdir(parents=True,exist_ok=True)

spec=importlib.util.spec_from_file_location('plv2',BASE)
plv2=importlib.util.module_from_spec(spec); assert spec and spec.loader; spec.loader.exec_module(plv2)

STOPS=[1.0,1.5,2.0,2.5]
HOLDS=[10,20]


def pf(r):
    r=pd.Series(r).dropna(); gp=r[r>0].sum(); gl=-r[r<0].sum()
    return np.inf if gl==0 and gp>0 else (gp/gl if gl>0 else np.nan)


def prep_source():
    s=pd.read_csv(SRC); s['Date']=pd.to_datetime(s['Date'],errors='coerce')
    for c in ['Open','High','Low','Close','Volume']:
        if c in s: s[c]=pd.to_numeric(s[c],errors='coerce')
    return s.dropna(subset=['Ticker','Date','Open','High','Low','Close'])


def get_events(src):
    events=[]; prepped={}
    for ticker,g in src.groupby('Ticker',sort=False):
        d=plv2.prep_ticker(g); prepped[str(ticker)]=d
        sig=np.flatnonzero(d['setup'].to_numpy(bool)); last=-10**9
        for i in sig:
            if i-last < plv2.DEDUP_SESSIONS: continue
            if i+1>=len(d): continue
            row={'Ticker':str(ticker),'i':int(i),'SignalDate':d.at[i,'Date'],'ATR20':float(d.at[i,'ATR20']),
                 'PeakD50_20':float(d.at[i,'PeakD50_20']) if np.isfinite(d.at[i,'PeakD50_20']) else np.nan,
                 'ret60':float(d.at[i,'ret60']) if np.isfinite(d.at[i,'ret60']) else np.nan,
                 'CollapseD50_20':float(d.at[i,'CollapseD50_20']) if np.isfinite(d.at[i,'CollapseD50_20']) else np.nan}
            events.append(row); last=int(i)
    return pd.DataFrame(events),prepped


def sim(d,i,atr,stop_atr,hold):
    if not np.isfinite(atr) or atr<=0 or i+hold>=len(d): return None
    entry=float(d.at[i+1,'Open']); stop=entry+stop_atr*atr
    outR=None; stopped=False; mfe=0.; mae=0.
    for j in range(i+1,i+hold+1):
        hi=float(d.at[j,'High']); lo=float(d.at[j,'Low'])
        mfe=max(mfe,(entry-lo)/atr); mae=max(mae,(hi-entry)/atr)
        if hi>=stop:
            outR=-stop_atr; stopped=True; break
    if outR is None:
        outR=(entry-float(d.at[i+hold,'Close']))/atr
    return outR,stopped,mfe,mae


def main():
    src=prep_source(); ev,prepped=get_events(src)
    ev['PeakQ']=pd.qcut(ev['PeakD50_20'].rank(method='first'),5,labels=['Q1','Q2','Q3','Q4','Q5'])
    rows=[]
    for stop in STOPS:
        for hold in HOLDS:
            rr=[]
            for _,e in ev.iterrows():
                d=prepped[e.Ticker]; x=sim(d,int(e.i),float(e.ATR20),stop,hold)
                if x is None: continue
                r,st,mfe,mae=x
                rr.append({**e.to_dict(),'StopATR':stop,'Hold':hold,'R2':r,'Stopped2':st,'MFE2':mfe,'MAE2':mae})
            z=pd.DataFrame(rr)
            for q in ['ALL','Q1','Q2','Q3','Q4','Q5']:
                g=z if q=='ALL' else z[z.PeakQ.astype(str)==q]
                r=g.R2
                rows.append({'StopATR':stop,'Hold':hold,'PeakQ':q,'N':len(g),'MeanR':r.mean(),'PF':pf(r),
                             'WinRate':(r>0).mean(),'StopRate':g.Stopped2.mean(),'MeanMFE':g.MFE2.mean(),'MeanMAE':g.MAE2.mean(),
                             'PeakMean':g.PeakD50_20.mean()})
    out=pd.DataFrame(rows); out.to_csv(OUT/'stop_sensitivity.csv',index=False)
    txt=[]; txt.append('PARADISE LOST STOP WIDTH SENSITIVITY — PeakD50 quintiles')
    txt.append('R remains normalized by signal ATR; stop loss in R equals chosen ATR multiple. Same A+ v2 signals.')
    for hold in HOLDS:
        txt.append(f'\nHOLD {hold}')
        txt.append(out[out.Hold==hold].round(3).to_string(index=False))
    (OUT/'summary.txt').write_text('\n'.join(txt),encoding='utf-8')
    print((OUT/'summary.txt').read_text(encoding='utf-8'))

if __name__=='__main__': main()
