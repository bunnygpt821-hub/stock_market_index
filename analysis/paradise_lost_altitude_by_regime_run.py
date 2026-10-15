#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from pathlib import Path
import pandas as pd
import yfinance as yf

BASE = Path(__file__).with_name('paradise_lost_altitude_by_regime.py')
spec = importlib.util.spec_from_file_location('altreg', BASE)
mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(mod)


def download_topix_yahoo_japan() -> pd.DataFrame:
    q = yf.download('998405.T', start='2015-01-01', end='2026-10-01', progress=False, auto_adjust=False)
    if q is None or len(q) == 0:
        raise RuntimeError('TOPIX 998405.T download failed')
    if isinstance(q.columns, pd.MultiIndex):
        q.columns = [c[0] for c in q.columns]
    q = q.reset_index()
    q['Date'] = pd.to_datetime(q['Date'], errors='coerce')
    q['Close'] = pd.to_numeric(q['Close'], errors='coerce')
    q = q.dropna(subset=['Date','Close']).sort_values('Date').drop_duplicates('Date', keep='last')
    for n in [50,100]:
        q[f'MA{n}'] = q.Close.rolling(n, min_periods=n).mean()
    q['TOPIX_d100'] = (q.Close/q.MA100 - 1.0)*100.0
    return q[['Date','Close','MA50','MA100','TOPIX_d100']].rename(
        columns={'Close':'TOPIX','MA50':'TOPIX_MA50','MA100':'TOPIX_MA100'}
    )


mod.download_topix = download_topix_yahoo_japan
mod.main()
