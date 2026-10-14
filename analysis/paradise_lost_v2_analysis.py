#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import math
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'runtime/nikkei225_yahoo_clean_2016_present.csv'
OUT = ROOT / 'analysis_outputs/paradise_lost_v2'
OUT.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Pre-registered / reproducible A+ v2 setup.
# No threshold below is selected from performance results.
# -----------------------------------------------------------------------------
HIGH_ZONE_PCT = 0.005       # signal High within 0.5% of rolling 60-session High
BODY_ATR = 0.8              # bearish real body >= 0.8 * signal-day ATR20
CLOSE_POS_MAX = 0.30        # close in bottom 30% of daily range
DEDUP_SESSIONS = 10         # same-ticker accepted signals at least 10 sessions apart
HOLD_DAYS = 10              # entry day = day 1; if not stopped, day-10 close exit
STOP_ATR = 1.0
START = pd.Timestamp('2016-01-01')
END = pd.Timestamp('2026-09-18')


def pf_of(r: pd.Series) -> float:
    x = pd.to_numeric(r, errors='coerce').dropna()
    gp = x[x > 0].sum(); gl = -x[x < 0].sum()
    if gl == 0:
        return float('inf') if gp > 0 else float('nan')
    return float(gp / gl)


def stats_row(g: pd.DataFrame) -> dict:
    r = g['R'].dropna()
    return {
        'N': int(len(g)),
        'MeanR': float(r.mean()) if len(r) else np.nan,
        'MedianR': float(r.median()) if len(r) else np.nan,
        'PF': pf_of(r),
        'WinRate': float((r > 0).mean()) if len(r) else np.nan,
        'Hit1R': float(g['hit1'].mean()) if len(g) else np.nan,
        'Hit1_5R': float(g['hit1_5'].mean()) if len(g) else np.nan,
        'Hit2R': float(g['hit2'].mean()) if len(g) else np.nan,
        'MeanMFE': float(g['MFE_R'].mean()) if len(g) else np.nan,
        'MeanMAE': float(g['MAE_R'].mean()) if len(g) else np.nan,
        'StopRate': float(g['stopped'].mean()) if len(g) else np.nan,
    }


def bucket_stats(trades: pd.DataFrame, feature: str, bins, labels) -> pd.DataFrame:
    x = trades.copy()
    x['Bucket'] = pd.cut(x[feature], bins=bins, labels=labels, right=False, include_lowest=True)
    rows = []
    for lab in labels:
        g = x[x['Bucket'] == lab]
        if len(g) == 0:
            row = {'Feature': feature, 'Bucket': str(lab), 'N': 0}
        else:
            row = {'Feature': feature, 'Bucket': str(lab), **stats_row(g)}
            row['FeatureMean'] = float(g[feature].mean())
            row['FeatureMedian'] = float(g[feature].median())
        rows.append(row)
    return pd.DataFrame(rows)


def quantile_stats(trades: pd.DataFrame, feature: str, q=5) -> pd.DataFrame:
    x = trades[np.isfinite(trades[feature])].copy()
    if x.empty:
        return pd.DataFrame()
    # rank first makes equal values deterministic and prevents qcut duplicate-edge failure.
    x['_rank'] = x[feature].rank(method='first')
    x['Q'] = pd.qcut(x['_rank'], q=q, labels=[f'Q{i+1}' for i in range(q)])
    rows = []
    for lab, g in x.groupby('Q', observed=True):
        rows.append({'Feature': feature, 'Quantile': str(lab), **stats_row(g),
                     'FeatureMin': float(g[feature].min()), 'FeatureMax': float(g[feature].max()),
                     'FeatureMean': float(g[feature].mean())})
    return pd.DataFrame(rows)


def prep_ticker(g: pd.DataFrame) -> pd.DataFrame:
    d = g.sort_values('Date').copy().reset_index(drop=True)
    # Preserve exact Yahoo-adjusted source, but make intraday H/L internally consistent.
    d['High'] = d[['Open','High','Close']].max(axis=1)
    d['Low'] = d[['Open','Low','Close']].min(axis=1)
    pc = d['Close'].shift(1)
    tr = pd.concat([d['High']-d['Low'], (d['High']-pc).abs(), (d['Low']-pc).abs()], axis=1).max(axis=1)
    d['ATR20'] = tr.rolling(20, min_periods=20).mean()
    for n in (15,25,50,100):
        d[f'MA{n}'] = d['Close'].rolling(n, min_periods=n).mean()
        d[f'd{n}'] = (d['Close']/d[f'MA{n}'] - 1.0) * 100.0
    for n in (5,10,20,60):
        d[f'ret{n}'] = (d['Close']/d['Close'].shift(n) - 1.0) * 100.0
    # Peak is PRIOR 20 sessions, excluding the signal day: prior altitude before collapse.
    for n in (25,50,100):
        d[f'PeakD{n}_20'] = d[f'd{n}'].shift(1).rolling(20, min_periods=20).max()
        d[f'CollapseD{n}_20'] = d[f'PeakD{n}_20'] - d[f'd{n}']
    d['H60'] = d['High'].rolling(60, min_periods=60).max()
    d['high_zone_gap_pct'] = (d['H60'] - d['High']) / d['H60'] * 100.0
    body = d['Open'] - d['Close']
    rng = (d['High'] - d['Low']).replace(0, np.nan)
    close_pos = (d['Close'] - d['Low']) / rng
    d['setup'] = (
        (d['Date'] >= START) & (d['Date'] <= END)
        & (d['Open'] > d['Close'])
        & (body >= BODY_ATR * d['ATR20'])
        & (close_pos <= CLOSE_POS_MAX)
        & (d['High'] >= (1.0 - HIGH_ZONE_PCT) * d['H60'])
        & (d['Close'].shift(1) >= d['MA15'].shift(1))
        & (d['Close'] < d['MA15'])
    )
    d['Ticker'] = g['Ticker'].iloc[0]
    d['seq'] = np.arange(len(d), dtype=int)
    return d


def simulate_one(d: pd.DataFrame, i: int) -> dict | None:
    # Need next session entry plus ten complete holding sessions: i+1 ... i+10.
    if i + HOLD_DAYS >= len(d):
        return None
    atr = float(d.at[i, 'ATR20'])
    if not np.isfinite(atr) or atr <= 0:
        return None
    entry_i = i + 1
    exit_i = i + HOLD_DAYS
    entry = float(d.at[entry_i, 'Open'])
    if not np.isfinite(entry):
        return None
    stop = entry + STOP_ATR * atr
    window = d.loc[entry_i:exit_i]
    stopped = False
    final_r = np.nan
    out_i = exit_i
    hit = {1.0: False, 1.5: False, 2.0: False}
    alive_for_target = {1.0: True, 1.5: True, 2.0: True}
    mfe = 0.0; mae = 0.0
    for j, row in window.iterrows():
        hi = float(row['High']); lo = float(row['Low'])
        mfe = max(mfe, (entry - lo)/atr)
        mae = max(mae, (hi - entry)/atr)
        stop_today = hi >= stop
        # Conservative OHLC ordering: if stop and target occur same day, stop wins.
        for t in (1.0,1.5,2.0):
            if not hit[t] and alive_for_target[t]:
                if stop_today:
                    alive_for_target[t] = False
                elif lo <= entry - t*atr:
                    hit[t] = True
        if stop_today:
            stopped = True
            final_r = -1.0
            out_i = j
            break
    if not stopped:
        final_r = (entry - float(d.at[exit_i,'Close'])) / atr
    row = {
        'Ticker': d.at[i,'Ticker'], 'SignalDate': d.at[i,'Date'],
        'EntryDate': d.at[entry_i,'Date'], 'ExitDate': d.at[out_i,'Date'],
        'SignalClose': float(d.at[i,'Close']), 'EntryOpen': entry, 'ATR20': atr,
        'R': float(final_r), 'stopped': bool(stopped), 'MFE_R': float(mfe), 'MAE_R': float(mae),
        'hit1': hit[1.0], 'hit1_5': hit[1.5], 'hit2': hit[2.0],
        'high_zone_gap_pct': float(d.at[i,'high_zone_gap_pct']),
    }
    for c in ['d25','d50','d100','ret5','ret10','ret20','ret60',
              'PeakD25_20','PeakD50_20','PeakD100_20',
              'CollapseD25_20','CollapseD50_20','CollapseD100_20']:
        row[c] = float(d.at[i,c]) if np.isfinite(d.at[i,c]) else np.nan
    return row


def build_trades(source: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame]:
    prepped=[]; events=[]
    for ticker,g in source.groupby('Ticker',sort=False):
        d=prep_ticker(g)
        prepped.append(d)
        sig=np.flatnonzero(d['setup'].to_numpy(bool))
        last=-10**9
        for i in sig:
            # Prior convention: accept next same-ticker signal only when at least 10 sessions apart.
            if i-last < DEDUP_SESSIONS:
                continue
            x=simulate_one(d,int(i))
            if x is not None:
                events.append(x); last=int(i)
    return pd.DataFrame(events), pd.concat(prepped,ignore_index=True)


def spearman_table(trades: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows=[]
    for f in features:
        z=trades[[f,'R']].dropna()
        rho=z[f].rank().corr(z['R'].rank()) if len(z)>2 else np.nan
        # For a short, higher feature is better if it raises R; positive rho = favorable monotonicity.
        rows.append({'Feature':f,'N':len(z),'Spearman_Rho_vs_R':rho})
    return pd.DataFrame(rows)


def two_d_table(trades: pd.DataFrame, xfeature: str, yfeature: str, qx=4, qy=4) -> pd.DataFrame:
    z=trades[[xfeature,yfeature,'R','hit1','hit1_5','hit2','MFE_R','MAE_R','stopped']].dropna().copy()
    z['_xr']=z[xfeature].rank(method='first');z['_yr']=z[yfeature].rank(method='first')
    z['XQ']=pd.qcut(z['_xr'],qx,labels=[f'Q{i+1}' for i in range(qx)])
    z['YQ']=pd.qcut(z['_yr'],qy,labels=[f'Q{i+1}' for i in range(qy)])
    rows=[]
    for (xq,yq),g in z.groupby(['XQ','YQ'],observed=True):
        rows.append({'XFeature':xfeature,'YFeature':yfeature,'XQ':str(xq),'YQ':str(yq),
                     'XMean':g[xfeature].mean(),'YMean':g[yfeature].mean(),**stats_row(g)})
    return pd.DataFrame(rows)


def main():
    src=pd.read_csv(SRC)
    src['Date']=pd.to_datetime(src['Date'],errors='coerce')
    for c in ['Open','High','Low','Close','Volume']:
        src[c]=pd.to_numeric(src[c],errors='coerce')
    src=src.dropna(subset=['Ticker','Date','Open','High','Low','Close'])
    trades,_=build_trades(src)
    trades=trades.sort_values(['SignalDate','Ticker']).reset_index(drop=True)
    trades.to_csv(OUT/'v2_trades.csv',index=False)

    overall=pd.DataFrame([{'Setup':'A+ v2',**stats_row(trades)}])
    overall.to_csv(OUT/'overall.csv',index=False)

    # 1) Current MA divergences
    current_specs={
        'd25':([-np.inf,0,3,5,8,np.inf],['<0','0-3','3-5','5-8','>=8']),
        'd50':([-np.inf,0,5,10,np.inf],['<0','0-5','5-10','>=10']),
        'd100':([-np.inf,0,10,15,20,np.inf],['<0','0-10','10-15','15-20','>=20']),
    }
    cur_fixed=pd.concat([bucket_stats(trades,f,*spec) for f,spec in current_specs.items()],ignore_index=True)
    cur_q=pd.concat([quantile_stats(trades,f,5) for f in current_specs],ignore_index=True)
    cur_fixed.to_csv(OUT/'01_current_ma_fixed.csv',index=False);cur_q.to_csv(OUT/'01_current_ma_quintiles.csv',index=False)

    # 2) Prior returns
    ret_specs={
        'ret5':([-np.inf,0,3,7,12,np.inf],['<0','0-3','3-7','7-12','>=12']),
        'ret10':([-np.inf,0,5,10,20,np.inf],['<0','0-5','5-10','10-20','>=20']),
        'ret20':([-np.inf,0,10,20,35,np.inf],['<0','0-10','10-20','20-35','>=35']),
        'ret60':([-np.inf,0,20,40,70,np.inf],['<0','0-20','20-40','40-70','>=70']),
    }
    ret_fixed=pd.concat([bucket_stats(trades,f,*spec) for f,spec in ret_specs.items()],ignore_index=True)
    ret_q=pd.concat([quantile_stats(trades,f,5) for f in ret_specs],ignore_index=True)
    ret_fixed.to_csv(OUT/'02_prior_returns_fixed.csv',index=False);ret_q.to_csv(OUT/'02_prior_returns_quintiles.csv',index=False)

    # 3) Peak prior divergences (prior 20 sessions, excluding signal)
    peak_q=pd.concat([quantile_stats(trades,f,5) for f in ['PeakD25_20','PeakD50_20','PeakD100_20']],ignore_index=True)
    peak_q.to_csv(OUT/'03_peak_divergence_quintiles.csv',index=False)

    # 4) Collapse = Peak - Current
    collapse_q=pd.concat([quantile_stats(trades,f,5) for f in ['CollapseD25_20','CollapseD50_20','CollapseD100_20']],ignore_index=True)
    collapse_q.to_csv(OUT/'04_collapse_quintiles.csv',index=False)

    # Monotonicity diagnostics across all selector features.
    selector_features=list(current_specs)+list(ret_specs)+['PeakD25_20','PeakD50_20','PeakD100_20','CollapseD25_20','CollapseD50_20','CollapseD100_20']
    mono=spearman_table(trades,selector_features)
    mono.to_csv(OUT/'feature_spearman.csv',index=False)

    # 5) 2D interaction maps: run-up x collapse.
    maps=[]
    for x,y in [('ret20','CollapseD50_20'),('ret20','CollapseD100_20'),('ret60','CollapseD50_20'),('ret60','CollapseD100_20')]:
        maps.append(two_d_table(trades,x,y,4,4))
    maps=pd.concat(maps,ignore_index=True)
    maps.to_csv(OUT/'05_runup_x_collapse_2d.csv',index=False)

    # Human-readable compact summary.
    def fmt_table(df, cols):
        q=df.copy()
        for c in cols:
            if c in q.columns and pd.api.types.is_numeric_dtype(q[c]): q[c]=q[c].map(lambda v: '' if pd.isna(v) else f'{v:.3f}')
        return q.to_string(index=False)
    with (OUT/'summary.txt').open('w',encoding='utf-8') as f:
        f.write('PARADISE LOST A+ v2 — REPRODUCIBLE CANONICAL SETUP\n')
        f.write(f'Source: Yahoo auto_adjust=True, repair=False; current Nikkei225; 2016-01-04..2026-09-18.\n')
        f.write(f'Setup: High within {HIGH_ZONE_PCT*100:.1f}% of rolling60 High; bearish body >= {BODY_ATR:.1f}ATR20; close bottom {CLOSE_POS_MAX*100:.0f}%; prev Close>=MA15 and signal Close<MA15.\n')
        f.write(f'Dedup: same ticker >= {DEDUP_SESSIONS} sessions. Entry next open; stop +1ATR; un-stopped exit day-{HOLD_DAYS} close. Same-day stop+target => stop first.\n')
        f.write('NOTE: v2 is NOT claimed to be the unrecoverable old N=295 event set. It is a fixed reproducible successor.\n\n')
        f.write('OVERALL\n'+fmt_table(overall,['MeanR','MedianR','PF','WinRate','Hit1R','Hit1_5R','Hit2R','MeanMFE','MeanMAE','StopRate'])+'\n\n')
        f.write('1 CURRENT MA — FIXED BANDS\n'+fmt_table(cur_fixed,['MeanR','MedianR','PF','WinRate','Hit1R','Hit1_5R','Hit2R','MeanMFE','MeanMAE','StopRate','FeatureMean'])+'\n\n')
        f.write('1 CURRENT MA — QUINTILES\n'+fmt_table(cur_q,['MeanR','PF','WinRate','Hit1R','Hit2R','FeatureMin','FeatureMax','FeatureMean'])+'\n\n')
        f.write('2 PRIOR RETURNS — QUINTILES\n'+fmt_table(ret_q,['MeanR','PF','WinRate','Hit1R','Hit2R','FeatureMin','FeatureMax','FeatureMean'])+'\n\n')
        f.write('3 PEAK DIVERGENCE — QUINTILES\n'+fmt_table(peak_q,['MeanR','PF','WinRate','Hit1R','Hit2R','FeatureMin','FeatureMax','FeatureMean'])+'\n\n')
        f.write('4 COLLAPSE — QUINTILES\n'+fmt_table(collapse_q,['MeanR','PF','WinRate','Hit1R','Hit2R','FeatureMin','FeatureMax','FeatureMean'])+'\n\n')
        f.write('SPEARMAN FEATURE vs FINAL R\n'+fmt_table(mono,['Spearman_Rho_vs_R'])+'\n\n')
        for (x,y),g in maps.groupby(['XFeature','YFeature'],sort=False):
            f.write(f'5 2D {x} x {y}\n'+fmt_table(g[['XQ','YQ','N','XMean','YMean','MeanR','PF','WinRate','Hit1R','Hit2R']],['XMean','YMean','MeanR','PF','WinRate','Hit1R','Hit2R'])+'\n\n')
    print((OUT/'summary.txt').read_text(encoding='utf-8'))

if __name__=='__main__':
    main()
