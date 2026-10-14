#!/usr/bin/env python3
from __future__ import annotations

import time
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

START = pd.Timestamp('2016-01-01')
END = pd.Timestamp('2026-09-19')  # exclusive; last market day 2026-09-18
CODES = ['4151', '4502', '4503', '4506', '4507', '4519', '4523', '4568', '4578', '285A', '4062', '6479', '6501', '6503', '6504', '6506', '6526', '6645', '6701', '6702', '6723', '6724', '6752', '6753', '6758', '6762', '6770', '6841', '6857', '6861', '6902', '6920', '6954', '6963', '6971', '6976', '6981', '7735', '7751', '7752', '8035', '543A', '7201', '7202', '7203', '7211', '7261', '7267', '7269', '7270', '7272', '4543', '4902', '6146', '7731', '7733', '7741', '9432', '9433', '9434', '9984', '5831', '7186', '8304', '8306', '8308', '8309', '8316', '8331', '8354', '8411', '8253', '8591', '8697', '8601', '8604', '8630', '8725', '8750', '8766', '8795', '1332', '2002', '2269', '2282', '2501', '2502', '2503', '2801', '2802', '2871', '2914', '3086', '3092', '3099', '3382', '7453', '7532', '8233', '8252', '8267', '9843', '9983', '2413', '2432', '3659', '3697', '4307', '4324', '4385', '4661', '4689', '4704', '4751', '4755', '6098', '6178', '6532', '7974', '9602', '9735', '9766', '1605', '3401', '3402', '3861', '3405', '3407', '4004', '4005', '4021', '4042', '4043', '4061', '4063', '4183', '4188', '4208', '4452', '4901', '4911', '6988', '5019', '5020', '5101', '5108', '5201', '5214', '5233', '5301', '5332', '5333', '5401', '5406', '5411', '3436', '5706', '5711', '5713', '5714', '5801', '5802', '5803', '2768', '8001', '8002', '8015', '8031', '8053', '8058', '1721', '1801', '1802', '1803', '1808', '1812', '1925', '1928', '1963', '5631', '6103', '6113', '6273', '6301', '6302', '6305', '6326', '6361', '6367', '6471', '6472', '6473', '7004', '7011', '7013', '7012', '7832', '7911', '7912', '7951', '3289', '8801', '8802', '8804', '8830', '9001', '9005', '9007', '9008', '9009', '9020', '9021', '9022', '9064', '9147', '9101', '9104', '9107', '9201', '9202', '9501', '9502', '9503', '9531', '9532']
TICKERS = [c + '.T' for c in CODES]
assert len(TICKERS) == 225
MIN_DATE = {'543A.T': pd.Timestamp('2026-04-01'), '285A.T': pd.Timestamp('2024-12-18')}
OUT = Path('runtime/nikkei225_yahoo_clean_2016_present.csv')
OUT.parent.mkdir(parents=True, exist_ok=True)


def normalize(raw, tickers):
    frames=[]; found=set()
    if raw is None or raw.empty: return frames,found
    for ticker in tickers:
        sub=None
        if isinstance(raw.columns,pd.MultiIndex):
            if ticker in raw.columns.get_level_values(0): sub=raw[ticker].copy()
            elif ticker in raw.columns.get_level_values(-1):
                try: sub=raw.xs(ticker,axis=1,level=-1).copy()
                except Exception: pass
        elif len(tickers)==1: sub=raw.copy()
        if sub is None or sub.empty: continue
        req=['Open','High','Low','Close','Volume']
        if not all(c in sub.columns for c in req): continue
        sub=sub[req].copy().dropna(subset=['Open','High','Low','Close'])
        if sub.empty: continue
        sub=sub.reset_index(); dc='Date' if 'Date' in sub.columns else sub.columns[0]
        sub=sub.rename(columns={dc:'Date'}); sub['Date']=pd.to_datetime(sub['Date'],errors='coerce')
        try: sub['Date']=sub['Date'].dt.tz_localize(None)
        except Exception: pass
        sub['Date']=sub['Date'].dt.normalize(); sub.insert(0,'Ticker',ticker)
        lo=MIN_DATE.get(ticker,START)
        sub=sub[(sub['Date']>=lo)&(sub['Date']<END)].drop_duplicates('Date',keep='last').sort_values('Date')
        if not sub.empty:
            frames.append(sub[['Ticker','Date','Open','High','Low','Close','Volume']]); found.add(ticker)
    return frames,found


def download_batch(tickers):
    raw=yf.download(tickers=tickers,start='2016-01-01',end=END.strftime('%Y-%m-%d'),interval='1d',group_by='ticker',auto_adjust=True,repair=False,actions=False,threads=True,progress=False,keepna=False,timeout=30,multi_level_index=True)
    return normalize(raw,tickers)


def retry_one(ticker,tries=4):
    for n in range(1,tries+1):
        try:
            d=yf.Ticker(ticker).history(start='2016-01-01',end=END.strftime('%Y-%m-%d'),interval='1d',auto_adjust=True,repair=False,actions=False,keepna=False,raise_errors=False)
            if d is not None and not d.empty:
                req=['Open','High','Low','Close','Volume']
                if all(c in d.columns for c in req):
                    d=d[req].dropna(subset=['Open','High','Low','Close']).copy().reset_index()
                    d['Date']=pd.to_datetime(d['Date'],errors='coerce')
                    try: d['Date']=d['Date'].dt.tz_localize(None)
                    except Exception: pass
                    d['Date']=d['Date'].dt.normalize(); d.insert(0,'Ticker',ticker)
                    lo=MIN_DATE.get(ticker,START)
                    d=d[(d['Date']>=lo)&(d['Date']<END)].drop_duplicates('Date',keep='last').sort_values('Date')
                    if not d.empty: return d[['Ticker','Date','Open','High','Low','Close','Volume']]
        except Exception as e: print('retry',ticker,n,e)
        time.sleep(n)
    return None

frames=[]; found=set(); BATCH=15
for i in range(0,len(TICKERS),BATCH):
    batch=TICKERS[i:i+BATCH]
    print(f'batch {i+1}-{min(i+BATCH,len(TICKERS))}/225',flush=True)
    try:
        fs,ok=download_batch(batch); frames.extend(fs); found|=ok
    except Exception as e: print('batch error',e,flush=True)
    time.sleep(0.25)

missing=[t for t in TICKERS if t not in found]
print('individual retries',len(missing),flush=True)
for t in missing:
    d=retry_one(t)
    if d is not None and not d.empty: frames.append(d); found.add(t)
if not frames: raise RuntimeError('no data')
raw=pd.concat(frames,ignore_index=True).drop_duplicates(['Ticker','Date'],keep='last').sort_values(['Ticker','Date']).reset_index(drop=True)
positive_counts=raw.assign(_pos=(raw['Volume'].fillna(0)>0).astype(int)).groupby('Date')['_pos'].sum()
market_dates=set(positive_counts[positive_counts>=100].index)
d=raw[raw['Date'].isin(market_dates)].copy()
d.to_csv(OUT,index=False,encoding='utf-8-sig')
print('SUMMARY', 'tickers=',d['Ticker'].nunique(),'rows=',len(d),'first=',d['Date'].min().date(),'last=',d['Date'].max().date(),'market_days=',len(market_dates),'missing=',len(set(TICKERS)-set(d['Ticker'].unique())),flush=True)
