#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'runtime/nikkei225_yahoo_clean_2016_present.csv'
IDX=ROOT/'data/indexes/nikkei225/daily.csv'
BASE_PATH=Path(__file__).with_name('paradise_lost_v2_analysis.py')
OUT=ROOT/'analysis_outputs/paradise_lost_index_regime'
OUT.mkdir(parents=True,exist_ok=True)

spec=importlib.util.spec_from_file_location('plv2',BASE_PATH)
plv2=importlib.util.module_from_spec(spec); assert spec and spec.loader; spec.loader.exec_module(plv2)

PERIODS=[
 ('TRAIN_2016_2020',pd.Timestamp('2016-01-01'),pd.Timestamp('2020-12-31')),
 ('VALID_2021_2023',pd.Timestamp('2021-01-01'),pd.Timestamp('2023-12-31')),
 ('TEST_2024_2026',pd.Timestamp('2024-01-01'),pd.Timestamp('2026-09-18')),
]

def pf(x):
    r=pd.to_numeric(x,errors='coerce').dropna();gp=r[r>0].sum();gl=-r[r<0].sum()
    if gl==0:return float('inf') if gp>0 else np.nan
    return float(gp/gl)

def stats(g):
    if len(g)==0:return {'N':0,'MeanR':np.nan,'PF':np.nan,'WinRate':np.nan,'StopRate':np.nan,'Hit1R':np.nan,'Hit2R':np.nan}
    return {'N':len(g),'MeanR':g.R.mean(),'PF':pf(g.R),'WinRate':(g.R>0).mean(),'StopRate':g.stopped.mean(),'Hit1R':g.hit1.mean(),'Hit2R':g.hit2.mean()}

def load_src():
    s=pd.read_csv(SRC);s['Date']=pd.to_datetime(s.Date,errors='coerce')
    for c in ['Open','High','Low','Close','Volume']:
        if c in s:s[c]=pd.to_numeric(s[c],errors='coerce')
    return s.dropna(subset=['Ticker','Date','Open','High','Low','Close'])

def load_idx():
    d=pd.read_csv(IDX)
    # repo index file uses lower-case columns in the source pack.
    d.columns=[str(c).strip().lower() for c in d.columns]
    datecol='date'; closecol='close'
    d[datecol]=pd.to_datetime(d[datecol],errors='coerce');d[closecol]=pd.to_numeric(d[closecol],errors='coerce')
    d=d.dropna(subset=[datecol,closecol]).sort_values(datecol).drop_duplicates(datecol,keep='last')
    out=pd.DataFrame({'SignalDate':d[datecol],'IDXClose':d[closecol]})
    for n in (5,15,25,50,100):
        ma=d[closecol].rolling(n,min_periods=n).mean()
        out[f'IDX_MA{n}']=ma.values
        out[f'IDX_d{n}']=((d[closecol]/ma)-1.0).values*100
        out[f'IDX_MA{n}_slope5']=(ma-ma.shift(5)).values
    out['IDX_below15']=out.IDXClose<out.IDX_MA15
    out['IDX_below25']=out.IDXClose<out.IDX_MA25
    out['IDX_below50']=out.IDXClose<out.IDX_MA50
    out['IDX_15_25_bear']=(out.IDX_MA15<out.IDX_MA25)&out.IDX_below15
    out['IDX_5_15_25_bear']=(out.IDX_MA5<out.IDX_MA15)&(out.IDX_MA15<out.IDX_MA25)&(out.IDXClose<out.IDX_MA5)
    out['IDX_25_down']=out.IDX_MA25_slope5<0
    out['IDX_50_down']=out.IDX_MA50_slope5<0
    return out

def main():
    src=load_src();trades,_=plv2.build_trades(src);trades=trades.sort_values(['SignalDate','Ticker']).reset_index(drop=True)
    valid=trades.PeakD50_20.notna();r=trades.loc[valid,'PeakD50_20'].rank(method='first')
    trades.loc[valid,'PeakQ']=pd.qcut(r,5,labels=['Q1','Q2','Q3','Q4','Q5']).astype(str).values
    idx=load_idx();z=trades.merge(idx,on='SignalDate',how='left');z.to_csv(OUT/'01_trades_index_regime.csv',index=False)

    rules={
      'ALL':lambda x:pd.Series(True,index=x.index),
      'IDX_BELOW15':lambda x:x.IDX_below15.fillna(False),
      'IDX_BELOW25':lambda x:x.IDX_below25.fillna(False),
      'IDX_BELOW50':lambda x:x.IDX_below50.fillna(False),
      'IDX_15_25_BEAR':lambda x:x.IDX_15_25_bear.fillna(False),
      'IDX_5_15_25_BEAR':lambda x:x.IDX_5_15_25_bear.fillna(False),
      'IDX_BELOW25_25DOWN':lambda x:x.IDX_below25.fillna(False)&x.IDX_25_down.fillna(False),
      'IDX_BELOW25_50DOWN':lambda x:x.IDX_below25.fillna(False)&x.IDX_50_down.fillna(False),
      'IDX_BELOW25_D25_LT_M2':lambda x:x.IDX_below25.fillna(False)&(x.IDX_d25<=-2),
      'IDX_BELOW25_D25_LT_M4':lambda x:x.IDX_below25.fillna(False)&(x.IDX_d25<=-4),
    }

    rows=[]
    for scope in ['ALL','HIGH_Q4Q5','Q5']:
        base=z if scope=='ALL' else z[z.PeakQ.isin(['Q4','Q5'])] if scope=='HIGH_Q4Q5' else z[z.PeakQ.eq('Q5')]
        for pname,start,end in PERIODS:
            p=base[(base.SignalDate>=start)&(base.SignalDate<=end)]
            for name,fn in rules.items():
                g=p[fn(p)]
                rows.append({'Scope':scope,'Period':pname,'Rule':name,
                             'IDX_d25_mean':g.IDX_d25.mean() if len(g) else np.nan,
                             'IDX_d50_mean':g.IDX_d50.mean() if len(g) else np.nan,**stats(g)})
    res=pd.DataFrame(rows);res.to_csv(OUT/'02_regime_results.csv',index=False)

    lines=[]
    lines.append('PARADISE LOST — NIKKEI225 INDEX MA REGIME')
    lines.append('Same A+ v2 signals. Index filter is evaluated on signal close; no future data.')
    lines.append('')
    lines.append('HIGH Q4+Q5')
    lines.append(res[res.Scope.eq('HIGH_Q4Q5')].round(3).to_string(index=False))
    lines.append('')
    lines.append('Q5 ONLY')
    lines.append(res[res.Scope.eq('Q5')].round(3).to_string(index=False))
    (OUT/'summary.txt').write_text('\n'.join(lines),encoding='utf-8')
    print((OUT/'summary.txt').read_text(encoding='utf-8'))

if __name__=='__main__':main()
