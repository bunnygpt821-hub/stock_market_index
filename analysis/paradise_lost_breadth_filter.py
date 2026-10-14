#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'runtime/nikkei225_yahoo_clean_2016_present.csv'
BASE_PATH=Path(__file__).with_name('paradise_lost_v2_analysis.py')
OUT=ROOT/'analysis_outputs/paradise_lost_breadth'
OUT.mkdir(parents=True,exist_ok=True)

spec=importlib.util.spec_from_file_location('plv2',BASE_PATH)
plv2=importlib.util.module_from_spec(spec); assert spec and spec.loader; spec.loader.exec_module(plv2)

PERIODS=[
 ('TRAIN_2016_2020',pd.Timestamp('2016-01-01'),pd.Timestamp('2020-12-31')),
 ('VALID_2021_2023',pd.Timestamp('2021-01-01'),pd.Timestamp('2023-12-31')),
 ('TEST_2024_2026',pd.Timestamp('2024-01-01'),pd.Timestamp('2026-09-18')),
]


def pf(x):
    r=pd.to_numeric(x,errors='coerce').dropna(); gp=r[r>0].sum(); gl=-r[r<0].sum()
    if gl==0:return float('inf') if gp>0 else np.nan
    return float(gp/gl)


def stats(g):
    if len(g)==0:return {'N':0,'MeanR':np.nan,'PF':np.nan,'WinRate':np.nan,'StopRate':np.nan,'Hit1R':np.nan,'Hit2R':np.nan}
    return {'N':len(g),'MeanR':g.R.mean(),'PF':pf(g.R),'WinRate':(g.R>0).mean(),'StopRate':g.stopped.mean(),'Hit1R':g.hit1.mean(),'Hit2R':g.hit2.mean()}


def load_src():
    s=pd.read_csv(SRC);s['Date']=pd.to_datetime(s['Date'],errors='coerce')
    for c in ['Open','High','Low','Close','Volume']:
        if c in s:s[c]=pd.to_numeric(s[c],errors='coerce')
    return s.dropna(subset=['Ticker','Date','Open','High','Low','Close'])


def build_breadth(src):
    rows=[]
    for ticker,g in src.groupby('Ticker',sort=False):
        d=g.sort_values('Date').copy()
        for n in (25,50,100):
            d[f'MA{n}']=d.Close.rolling(n,min_periods=n).mean()
            d[f'above{n}']=(d.Close>d[f'MA{n}']).where(d[f'MA{n}'].notna())
            d[f'slope{n}']=(d[f'MA{n}']>d[f'MA{n}'].shift(5)).where(d[f'MA{n}'].notna()&d[f'MA{n}'].shift(5).notna())
        rows.append(d[['Date','Ticker']+[f'above{n}' for n in (25,50,100)]+[f'slope{n}' for n in (25,50,100)]])
    x=pd.concat(rows,ignore_index=True)
    agg=[]
    for dt,g in x.groupby('Date'):
        r={'Date':dt,'UniverseN':g.Ticker.nunique()}
        for n in (25,50,100):
            a=g[f'above{n}'].dropna();s=g[f'slope{n}'].dropna()
            r[f'B{n}']=100*a.mean() if len(a) else np.nan
            r[f'S{n}']=100*s.mean() if len(s) else np.nan
        agg.append(r)
    b=pd.DataFrame(agg).sort_values('Date').reset_index(drop=True)
    for n in (25,50,100):
        b[f'dB{n}_5']=b[f'B{n}']-b[f'B{n}'].shift(5)
        b[f'dS{n}_5']=b[f'S{n}']-b[f'S{n}'].shift(5)
    b['BreadthMean']=b[['B25','B50','B100']].mean(axis=1)
    b['BreadthMean_d5']=b.BreadthMean-b.BreadthMean.shift(5)
    return b


def qedges(train,s,q=4):
    z=train[s].dropna();return np.quantile(z,np.linspace(0,1,q+1))

def assign(s,e):
    e=np.array(e,float);e[0]=-np.inf;e[-1]=np.inf
    for i in range(1,len(e)):
        if e[i]<=e[i-1]:e[i]=np.nextafter(e[i-1],np.inf)
    return pd.cut(s,e,labels=[f'Q{i}' for i in range(1,len(e))],include_lowest=True)


def matrix_oos(df,level,change,scope):
    base=df.copy()
    if scope=='HIGH_Q4Q5':base=base[base.PeakQ.isin(['Q4','Q5'])]
    train=base[(base.SignalDate>=PERIODS[0][1])&(base.SignalDate<=PERIODS[0][2])].dropna(subset=[level,change])
    le=qedges(train,level);ce=qedges(train,change)
    rows=[]
    for pname,start,end in PERIODS:
        p=base[(base.SignalDate>=start)&(base.SignalDate<=end)].dropna(subset=[level,change]).copy()
        p['LQ']=assign(p[level],le);p['CQ']=assign(p[change],ce)
        for lq in [f'Q{i}' for i in range(1,5)]:
            for cq in [f'Q{i}' for i in range(1,5)]:
                g=p[(p.LQ==lq)&(p.CQ==cq)]
                rows.append({'Scope':scope,'Level':level,'Change':change,'Period':pname,'LQ':lq,'CQ':cq,
                             'LevelMean':g[level].mean() if len(g) else np.nan,'ChangeMean':g[change].mean() if len(g) else np.nan,**stats(g)})
    return pd.DataFrame(rows),le,ce


def named_rules(df):
    rows=[]
    # Simple interpretable, predeclared market states.
    defs={
      'HIGH_ROLLOVER':lambda x:(x.BreadthMean>=60)&(x.BreadthMean_d5<=-10),
      'MID_ROLLOVER':lambda x:(x.BreadthMean.between(40,60,inclusive='left'))&(x.BreadthMean_d5<=-10),
      'LOW_FALLING':lambda x:(x.BreadthMean<40)&(x.BreadthMean_d5<0),
      'HIGH_STABLE_UP':lambda x:(x.BreadthMean>=60)&(x.BreadthMean_d5>=0),
      'FAST_DAMAGE':lambda x:(x.BreadthMean_d5<=-20),
      'B25_BREAK':lambda x:(x.B25<50)&(x.dB25_5<=-10),
      'B25_HIGH_ROLLOVER':lambda x:(x.B25>=60)&(x.dB25_5<=-15),
    }
    for scope in ['ALL','HIGH_Q4Q5']:
        z=df if scope=='ALL' else df[df.PeakQ.isin(['Q4','Q5'])]
        for pname,start,end in PERIODS:
            p=z[(z.SignalDate>=start)&(z.SignalDate<=end)]
            for name,fn in defs.items():
                g=p[fn(p)]
                rows.append({'Scope':scope,'Period':pname,'Rule':name,
                             'BreadthMean':g.BreadthMean.mean() if len(g) else np.nan,'Breadth_d5':g.BreadthMean_d5.mean() if len(g) else np.nan,**stats(g)})
    return pd.DataFrame(rows)


def main():
    src=load_src();trades,_=plv2.build_trades(src);trades=trades.sort_values(['SignalDate','Ticker']).reset_index(drop=True)
    valid=trades.PeakD50_20.notna();r=trades.loc[valid,'PeakD50_20'].rank(method='first')
    trades.loc[valid,'PeakQ']=pd.qcut(r,5,labels=['Q1','Q2','Q3','Q4','Q5']).astype(str).values
    breadth=build_breadth(src);breadth.to_csv(OUT/'00_daily_breadth.csv',index=False)
    z=trades.merge(breadth,left_on='SignalDate',right_on='Date',how='left').drop(columns=['Date'])
    z.to_csv(OUT/'01_trades_with_breadth.csv',index=False)

    mats=[];edges=[]
    for level,change in [('BreadthMean','BreadthMean_d5'),('B25','dB25_5'),('B50','dB50_5'),('B100','dB100_5')]:
        for scope in ['ALL','HIGH_Q4Q5']:
            m,le,ce=matrix_oos(z,level,change,scope);mats.append(m)
            for typ,e in [('LEVEL',le),('CHANGE',ce)]:
                for i,v in enumerate(e):edges.append({'Scope':scope,'Level':level,'Change':change,'Type':typ,'Boundary':i,'Value':v})
    mats=pd.concat(mats,ignore_index=True);mats.to_csv(OUT/'02_oos_breadth_matrices.csv',index=False)
    pd.DataFrame(edges).to_csv(OUT/'03_train_edges.csv',index=False)
    nr=named_rules(z);nr.to_csv(OUT/'04_named_market_states.csv',index=False)

    lines=[]
    lines.append('PARADISE LOST — BREADTH LEVEL x 5D CHANGE')
    lines.append('Breadth is built point-in-time from the same current-Nikkei225 historical price panel; therefore survivorship bias remains. Use as internal regime test, not a universal market claim.')
    lines.append('')
    lines.append('NAMED STATES — HIGH MOUNTAIN Q4+Q5')
    lines.append(nr[nr.Scope.eq('HIGH_Q4Q5')].round(3).to_string(index=False))
    lines.append('')
    lines.append('2024-26 MATRIX — HIGH Q4Q5 / BreadthMean x 5d change')
    a=mats[(mats.Scope=='HIGH_Q4Q5')&(mats.Level=='BreadthMean')&(mats.Period=='TEST_2024_2026')]
    lines.append(a.round(3).to_string(index=False))
    lines.append('')
    lines.append('2016-20 TRAIN EDGES')
    lines.append(pd.DataFrame(edges).query("Scope=='HIGH_Q4Q5' and Level=='BreadthMean'").round(3).to_string(index=False))
    (OUT/'summary.txt').write_text('\n'.join(lines),encoding='utf-8')
    print((OUT/'summary.txt').read_text(encoding='utf-8'))

if __name__=='__main__':main()
