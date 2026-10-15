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

BASE_URL = 'https://finance.yahoo.co.jp/quote/998405.T/history'
HEADERS = {'User-Agent':'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124 Safari/537.36'}


def _one_page(year: int, page: int) -> pd.DataFrame:
    start=f'{year}0101'
    end='20260918' if year==2026 else f'{year}1231'
    params={'from':start,'to':end,'timeFrame':'d','page':str(page)}
    last=None
    for attempt in range(5):
        try:
            r=requests.get(BASE_URL,params=params,timeout=30,headers=HEADERS)
            r.raise_for_status()
            tables=pd.read_html(StringIO(r.text))
            if not tables:
                return pd.DataFrame()
            t=tables[0].copy()
            if '日付' not in t.columns or '終値' not in t.columns:
                return pd.DataFrame()
            t['_year']=year; t['_page']=page
            return t
        except Exception as e:
            last=e; time.sleep(1.5*(attempt+1))
    print('WARN failed year/page',year,page,repr(last))
    return pd.DataFrame()


def download_topix_yahoo_japan() -> pd.DataFrame:
    frames=[]
    tasks=[(year,page) for year in range(2015,2027) for page in range(1,15)]
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs={ex.submit(_one_page,y,p):(y,p) for y,p in tasks}
        for fut in as_completed(futs):
            t=fut.result()
            if len(t): frames.append(t)
    if not frames:
        raise RuntimeError('Yahoo Japan TOPIX history scrape returned no tables')
    q=pd.concat(frames,ignore_index=True)
    q['Date']=pd.to_datetime(q['日付'],format='%Y/%m/%d',errors='coerce')
    q['Close']=pd.to_numeric(q['終値'].astype(str).str.replace(',','',regex=False),errors='coerce')
    q=q.dropna(subset=['Date','Close']).sort_values('Date').drop_duplicates('Date',keep='last')
    q=q[(q.Date>=pd.Timestamp('2015-01-01'))&(q.Date<=pd.Timestamp('2026-09-18'))].copy()
    if len(q)<2500:
        raise RuntimeError(f'TOPIX scrape too short: {len(q)} rows, {q.Date.min() if len(q) else None}..{q.Date.max() if len(q) else None}')
    for n in [50,100]:
        q[f'MA{n}']=q.Close.rolling(n,min_periods=n).mean()
    q['TOPIX_d100']=(q.Close/q.MA100-1.0)*100.0
    print('TOPIX scraped',len(q),'rows',q.Date.min().date(),'..',q.Date.max().date())
    return q[['Date','Close','MA50','MA100','TOPIX_d100']].rename(
        columns={'Close':'TOPIX','MA50':'TOPIX_MA50','MA100':'TOPIX_MA100'}
    )


mod.download_topix = download_topix_yahoo_japan
mod.main()
