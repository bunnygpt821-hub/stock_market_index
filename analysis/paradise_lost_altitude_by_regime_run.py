#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import StringIO
from pathlib import Path
import time

import pandas as pd
import requests

BASE = Path(__file__).with_name('paradise_lost_altitude_by_regime.py')
spec = importlib.util.spec_from_file_location('altreg', BASE)
mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(mod)

NY_URL='https://nikkeiyosoku.com/topix/data/'
HEADERS={'User-Agent':'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124 Safari/537.36'}


def _page(page:int)->pd.DataFrame:
    params={'start_date':'2015-01-01','end_date':'2026-09-18','page':str(page)}
    for attempt in range(4):
        try:
            r=requests.get(NY_URL,params=params,timeout=25,headers=HEADERS); r.raise_for_status()
            tabs=pd.read_html(StringIO(r.text))
            for t in tabs:
                if '日付' in t.columns and '終値' in t.columns:
                    return t[['日付','終値']].copy()
            return pd.DataFrame()
        except Exception as e:
            if attempt==3: print('WARN page',page,repr(e))
            time.sleep(1.2*(attempt+1))
    return pd.DataFrame()


def download_topix_nikkeiyosoku()->pd.DataFrame:
    frames=[]
    # 150 pages * 20 rows covers ~3000 sessions, safely beyond 2015-01 to 2026-09.
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs={ex.submit(_page,p):p for p in range(1,151)}
        for fut in as_completed(futs):
            p=futs[fut]
            t=fut.result()
            if len(t): frames.append(t)
    if not frames:
        raise RuntimeError('NikkeiYosoku TOPIX scrape returned no data')
    q=pd.concat(frames,ignore_index=True)
    q['Date']=pd.to_datetime(q['日付'],format='%Y/%m/%d',errors='coerce')
    q['Close']=pd.to_numeric(q['終値'].astype(str).str.replace(',','',regex=False),errors='coerce')
    q=q[['Date','Close']].dropna().sort_values('Date').drop_duplicates('Date',keep='last')
    q=q[(q.Date>=pd.Timestamp('2015-01-01'))&(q.Date<=pd.Timestamp('2026-09-18'))].copy()
    print('TOPIX NikkeiYosoku',len(q),'rows',q.Date.min().date(),'..',q.Date.max().date())
    if len(q)<2800 or q.Date.min()>pd.Timestamp('2015-01-10') or q.Date.max()<pd.Timestamp('2026-09-18'):
        raise RuntimeError('NikkeiYosoku TOPIX history incomplete')
    gaps=q.Date.diff().dt.days
    print('largest calendar gap',int(gaps.max()))
    for n in [50,100]:
        q[f'MA{n}']=q.Close.rolling(n,min_periods=n).mean()
    q['TOPIX_d100']=(q.Close/q.MA100-1.0)*100.0
    return q[['Date','Close','MA50','MA100','TOPIX_d100']].rename(columns={
        'Close':'TOPIX','MA50':'TOPIX_MA50','MA100':'TOPIX_MA100'})


mod.download_topix=download_topix_nikkeiyosoku
mod.main()
