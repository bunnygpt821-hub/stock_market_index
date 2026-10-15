#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import StringIO
from pathlib import Path
import time

import numpy as np
import pandas as pd
import requests

BASE = Path(__file__).with_name('paradise_lost_altitude_by_regime.py')
spec = importlib.util.spec_from_file_location('altreg', BASE)
mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(mod)

YH_URL='https://finance.yahoo.co.jp/quote/998405.T/history'
NY_URL='https://nikkeiyosoku.com/topix/data/'
HEADERS={'User-Agent':'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124 Safari/537.36'}


def _yahoo_page(year:int,page:int)->pd.DataFrame:
    params={'from':f'{year}0101','to':f'{year}1231','timeFrame':'d','page':str(page)}
    for attempt in range(3):
        try:
            r=requests.get(YH_URL,params=params,timeout=20,headers=HEADERS); r.raise_for_status()
            tabs=pd.read_html(StringIO(r.text))
            if not tabs: return pd.DataFrame()
            t=tabs[0].copy()
            if '日付' not in t.columns or '終値' not in t.columns: return pd.DataFrame()
            return t[['日付','終値']]
        except Exception as e:
            if attempt==2: print('WARN yahoo',year,page,repr(e))
            time.sleep(attempt+1)
    return pd.DataFrame()


def _ny_page(page:int)->pd.DataFrame:
    params={'start_date':'2023-10-01','end_date':'2026-09-18','page':str(page)}
    for attempt in range(3):
        try:
            r=requests.get(NY_URL,params=params,timeout=20,headers=HEADERS); r.raise_for_status()
            tabs=pd.read_html(StringIO(r.text))
            for t in tabs:
                if '日付' in t.columns and '終値' in t.columns:
                    return t[['日付','終値']].copy()
            return pd.DataFrame()
        except Exception as e:
            if attempt==2: print('WARN nikkeiyosoku',page,repr(e))
            time.sleep(attempt+1)
    return pd.DataFrame()


def download_topix_combined()->pd.DataFrame:
    # Yahoo Japan: stable for historical years through 2023.
    yf=[]
    tasks=[(y,p) for y in range(2015,2024) for p in range(1,14)]
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs=[ex.submit(_yahoo_page,y,p) for y,p in tasks]
        for fut in as_completed(futs):
            t=fut.result()
            if len(t): yf.append(t)
    y=pd.concat(yf,ignore_index=True)
    y['Date']=pd.to_datetime(y['日付'],format='%Y/%m/%d',errors='coerce')
    y['Close']=pd.to_numeric(y['終値'].astype(str).str.replace(',','',regex=False),errors='coerce')
    y=y[['Date','Close']].dropna().drop_duplicates('Date',keep='last')
    y['Source']='YahooJapan'

    # NikkeiYosoku: reliable for 2023-10 onward, with overlap for continuity check.
    nf=[]
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs=[ex.submit(_ny_page,p) for p in range(1,45)]
        for fut in as_completed(futs):
            t=fut.result()
            if len(t): nf.append(t)
    n=pd.concat(nf,ignore_index=True)
    n['Date']=pd.to_datetime(n['日付'],format='%Y/%m/%d',errors='coerce')
    n['Close']=pd.to_numeric(n['終値'].astype(str).str.replace(',','',regex=False),errors='coerce')
    n=n[['Date','Close']].dropna().drop_duplicates('Date',keep='last')
    n['Source']='NikkeiYosoku'

    ov=y.merge(n,on='Date',suffixes=('_y','_n'))
    if len(ov):
        ov['AbsDiff']=(ov.Close_y-ov.Close_n).abs()
        print('TOPIX overlap',len(ov),'rows; max abs diff',ov.AbsDiff.max(),'median',ov.AbsDiff.median())
        # Same index should agree to normal rounding. Refuse if the sources disagree materially.
        bad=ov[ov.AbsDiff>0.20]
        if len(bad)>max(2,int(len(ov)*0.05)):
            print(bad[['Date','Close_y','Close_n','AbsDiff']].head(20).to_string(index=False))
            raise RuntimeError(f'TOPIX source mismatch: {len(bad)}/{len(ov)} overlap rows differ >0.20')

    # Use NikkeiYosoku from 2023-10 onward, Yahoo before then.
    ypre=y[y.Date<pd.Timestamp('2023-10-01')][['Date','Close']]
    q=pd.concat([ypre,n[['Date','Close']]],ignore_index=True).sort_values('Date').drop_duplicates('Date',keep='last')
    q=q[(q.Date>=pd.Timestamp('2015-01-01'))&(q.Date<=pd.Timestamp('2026-09-18'))].copy()
    print('TOPIX combined',len(q),'rows',q.Date.min().date(),'..',q.Date.max().date())
    if len(q)<2800 or q.Date.max()<pd.Timestamp('2026-09-18'):
        raise RuntimeError('Combined TOPIX history incomplete')
    # Detect suspicious calendar gaps excluding long Japanese holiday periods.
    gaps=q.Date.sort_values().diff().dt.days
    print('largest calendar gap',int(gaps.max()))
    for nn in [50,100]:
        q[f'MA{nn}']=q.Close.rolling(nn,min_periods=nn).mean()
    q['TOPIX_d100']=(q.Close/q.MA100-1.0)*100.0
    return q[['Date','Close','MA50','MA100','TOPIX_d100']].rename(columns={
        'Close':'TOPIX','MA50':'TOPIX_MA50','MA100':'TOPIX_MA100'})


mod.download_topix=download_topix_combined
mod.main()
