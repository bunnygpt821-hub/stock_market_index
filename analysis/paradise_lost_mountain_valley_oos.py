#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'runtime/nikkei225_yahoo_clean_2016_present.csv'
BASE_PATH = Path(__file__).with_name('paradise_lost_v2_analysis.py')
OUT = ROOT / 'analysis_outputs/paradise_lost_mountain_valley'
OUT.mkdir(parents=True, exist_ok=True)

spec = importlib.util.spec_from_file_location('plv2', BASE_PATH)
plv2 = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(plv2)

PERIODS = [
    ('TRAIN_2016_2020', pd.Timestamp('2016-01-01'), pd.Timestamp('2020-12-31')),
    ('VALID_2021_2023', pd.Timestamp('2021-01-01'), pd.Timestamp('2023-12-31')),
    ('TEST_2024_2026', pd.Timestamp('2024-01-01'), pd.Timestamp('2026-09-18')),
]


def pf(x: pd.Series) -> float:
    r = pd.to_numeric(x, errors='coerce').dropna()
    gp = r[r > 0].sum(); gl = -r[r < 0].sum()
    if gl == 0:
        return float('inf') if gp > 0 else np.nan
    return float(gp / gl)


def load_source() -> pd.DataFrame:
    src = pd.read_csv(SRC)
    src['Date'] = pd.to_datetime(src['Date'], errors='coerce')
    for c in ['Open','High','Low','Close','Volume']:
        if c in src.columns:
            src[c] = pd.to_numeric(src[c], errors='coerce')
    return src.dropna(subset=['Ticker','Date','Open','High','Low','Close']).copy()


def enrich_forward_paths(trades: pd.DataFrame, prepped: pd.DataFrame) -> pd.DataFrame:
    out = trades.copy()
    keys = {}
    for ticker, g in prepped.groupby('Ticker', sort=False):
        d = g.sort_values('Date').reset_index(drop=True).copy()
        pos = {pd.Timestamp(dt): i for i, dt in enumerate(d['Date'])}
        keys[str(ticker)] = (d, pos)

    records = []
    for _, tr in out.iterrows():
        ticker = str(tr['Ticker'])
        sigdate = pd.Timestamp(tr['SignalDate'])
        d, pos = keys[ticker]
        i = pos.get(sigdate)
        rec = {'Ticker': ticker, 'SignalDate': sigdate}
        if i is None or i + 1 >= len(d):
            records.append(rec); continue

        entry_i = i + 1
        entry = float(d.at[entry_i, 'Open'])
        atr = float(d.at[i, 'ATR20'])
        h60 = float(d.at[i, 'H60']) if np.isfinite(d.at[i, 'H60']) else np.nan
        rec['EntryOpen_check'] = entry
        rec['H60_signal'] = h60

        for h in (5, 10, 20):
            end_i = min(entry_i + h - 1, len(d) - 1)
            w = d.loc[entry_i:end_i]
            if w.empty or not np.isfinite(entry) or entry <= 0:
                continue
            minlow = float(w['Low'].min())
            maxhigh = float(w['High'].max())
            endclose = float(w.iloc[-1]['Close'])
            rec[f'FAV_pct_{h}'] = (entry - minlow) / entry * 100.0
            rec[f'ADV_pct_{h}'] = (maxhigh - entry) / entry * 100.0
            rec[f'CloseShort_pct_{h}'] = (entry - endclose) / entry * 100.0
            if np.isfinite(atr) and atr > 0:
                rec[f'FAV_ATR_{h}'] = (entry - minlow) / atr
                rec[f'ADV_ATR_{h}'] = (maxhigh - entry) / atr
            if np.isfinite(h60) and h60 > 0:
                rec[f'PeakToLow_pct_{h}'] = (h60 - minlow) / h60 * 100.0

        records.append(rec)

    path = pd.DataFrame(records)
    out = out.merge(path, on=['Ticker','SignalDate'], how='left')
    return out


def quintile_depth_table(df: pd.DataFrame, feature: str) -> pd.DataFrame:
    z = df[np.isfinite(df[feature])].copy()
    z['_rank'] = z[feature].rank(method='first')
    z['Q'] = pd.qcut(z['_rank'], 5, labels=[f'Q{i}' for i in range(1,6)])
    rows=[]
    for q,g in z.groupby('Q', observed=True):
        row = {
            'Feature': feature, 'Q': str(q), 'N': len(g),
            'FeatureMin': g[feature].min(), 'FeatureMax': g[feature].max(), 'FeatureMean': g[feature].mean(),
        }
        for h in (5,10,20):
            for col in [f'FAV_pct_{h}', f'FAV_ATR_{h}', f'ADV_pct_{h}', f'ADV_ATR_{h}', f'PeakToLow_pct_{h}', f'CloseShort_pct_{h}']:
                if col in g:
                    row[f'{col}_mean'] = g[col].mean()
                    row[f'{col}_median'] = g[col].median()
        rows.append(row)
    return pd.DataFrame(rows)


def corr_table(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows=[]
    targets=[]
    for h in (5,10,20):
        targets += [f'FAV_pct_{h}', f'FAV_ATR_{h}', f'ADV_pct_{h}', f'ADV_ATR_{h}', f'PeakToLow_pct_{h}', f'CloseShort_pct_{h}']
    for f in features:
        for t in targets:
            if t not in df.columns:
                continue
            z = df[[f,t]].replace([np.inf,-np.inf],np.nan).dropna()
            rho = z[f].rank().corr(z[t].rank()) if len(z) >= 8 else np.nan
            rows.append({'Feature':f,'Target':t,'N':len(z),'Spearman':rho})
    return pd.DataFrame(rows)


def period_stats(g: pd.DataFrame) -> dict:
    return {
        'N': len(g),
        'MeanR': g['R'].mean() if len(g) else np.nan,
        'PF': pf(g['R']) if len(g) else np.nan,
        'WinRate': (g['R']>0).mean() if len(g) else np.nan,
        'Hit1R': g['hit1'].mean() if len(g) else np.nan,
        'Hit2R': g['hit2'].mean() if len(g) else np.nan,
        'StopRate': g['stopped'].mean() if len(g) else np.nan,
        'MeanMFE_R': g['MFE_R'].mean() if len(g) else np.nan,
        'MeanMAE_R': g['MAE_R'].mean() if len(g) else np.nan,
    }


def train_edges(train: pd.DataFrame, feature: str, q=4) -> list[float]:
    vals = train[feature].dropna().to_numpy(float)
    return list(np.quantile(vals, np.linspace(0,1,q+1)))


def assign_by_edges(s: pd.Series, edges: list[float]) -> pd.Series:
    e = np.array(edges, dtype=float)
    e[0] = -np.inf; e[-1] = np.inf
    # Avoid duplicate edges in pathological cases.
    for i in range(1,len(e)):
        if e[i] <= e[i-1]:
            e[i] = np.nextafter(e[i-1], np.inf)
    return pd.cut(s, bins=e, labels=[f'Q{i}' for i in range(1,len(e))], right=True, include_lowest=True)


def oos_cells(df: pd.DataFrame, x='ret60', y='CollapseD50_20') -> tuple[pd.DataFrame,pd.DataFrame]:
    train = df[(df['SignalDate'] >= PERIODS[0][1]) & (df['SignalDate'] <= PERIODS[0][2])].dropna(subset=[x,y]).copy()
    xedges = train_edges(train,x,4); yedges=train_edges(train,y,4)
    rows=[]
    for pname,start,end in PERIODS:
        g = df[(df['SignalDate']>=start)&(df['SignalDate']<=end)].dropna(subset=[x,y]).copy()
        g['XQ'] = assign_by_edges(g[x],xedges); g['YQ'] = assign_by_edges(g[y],yedges)
        for xq in [f'Q{i}' for i in range(1,5)]:
            for yq in [f'Q{i}' for i in range(1,5)]:
                c = g[(g['XQ']==xq)&(g['YQ']==yq)]
                rows.append({'Period':pname,'X':x,'Y':y,'XQ':xq,'YQ':yq,**period_stats(c),
                             'XMean':c[x].mean() if len(c) else np.nan,'YMean':c[y].mean() if len(c) else np.nan})
    cells=pd.DataFrame(rows)

    # Pre-declared "moderate" cells, not selected by test performance.
    moderate=[]
    for pname,_,_ in PERIODS:
        p=cells[cells['Period']==pname]
        for rule,cellspec in {
            'Q2xQ2':['Q2/Q2'],
            'Q2xQ3':['Q2/Q3'],
            'Q3xQ2':['Q3/Q2'],
            'Q3xQ3':['Q3/Q3'],
            'MODERATE_4CELLS':['Q2/Q2','Q2/Q3','Q3/Q2','Q3/Q3'],
        }.items():
            sub=p[(p['XQ'].astype(str)+'/'+p['YQ'].astype(str)).isin(cellspec)]
            # Cell-level aggregation cannot recover pooled PF, so rebuild from underlying trades below.
            moderate.append({'Period':pname,'Rule':rule,'Cells':';'.join(cellspec)})
    rules=pd.DataFrame(moderate)

    pooled=[]
    for pname,start,end in PERIODS:
        g = df[(df['SignalDate']>=start)&(df['SignalDate']<=end)].dropna(subset=[x,y]).copy()
        g['XQ']=assign_by_edges(g[x],xedges);g['YQ']=assign_by_edges(g[y],yedges)
        rule_defs={
            'Q2xQ2':[('Q2','Q2')], 'Q2xQ3':[('Q2','Q3')],
            'Q3xQ2':[('Q3','Q2')], 'Q3xQ3':[('Q3','Q3')],
            'MODERATE_4CELLS':[('Q2','Q2'),('Q2','Q3'),('Q3','Q2'),('Q3','Q3')],
        }
        for rule,pairs in rule_defs.items():
            mask=pd.Series(False,index=g.index)
            for a,b in pairs: mask |= ((g['XQ']==a)&(g['YQ']==b))
            c=g[mask]
            pooled.append({'Period':pname,'Rule':rule,**period_stats(c),
                           'Ret60Mean':c[x].mean() if len(c) else np.nan,
                           'Collapse50Mean':c[y].mean() if len(c) else np.nan})
    pooled=pd.DataFrame(pooled)
    edges=pd.DataFrame({
        'Feature':[x]*5+[y]*5,
        'Boundary':['min','Q25','Q50','Q75','max']*2,
        'Value':xedges+yedges,
    })
    return cells, pooled, edges


def main():
    src=load_source()
    trades, prepped = plv2.build_trades(src)
    trades=trades.sort_values(['SignalDate','Ticker']).reset_index(drop=True)
    z=enrich_forward_paths(trades,prepped)
    z.to_csv(OUT/'mountain_valley_trades.csv',index=False)

    mountain_features=['ret20','ret60','PeakD25_20','PeakD50_20','PeakD100_20','d50','d100']
    qt=pd.concat([quintile_depth_table(z,f) for f in mountain_features],ignore_index=True)
    qt.to_csv(OUT/'01_mountain_vs_future_depth_quintiles.csv',index=False)

    ct=corr_table(z,mountain_features)
    ct.to_csv(OUT/'02_mountain_vs_future_depth_spearman.csv',index=False)

    cells,pooled,edges=oos_cells(z,'ret60','CollapseD50_20')
    edges.to_csv(OUT/'03_oos_train_edges.csv',index=False)
    cells.to_csv(OUT/'04_oos_cells_ret60_x_collapse50.csv',index=False)
    pooled.to_csv(OUT/'05_oos_moderate_rules.csv',index=False)

    # Also ret20 x CollapseD50 as a second, pre-declared sensitivity check.
    cells20,pooled20,edges20=oos_cells(z,'ret20','CollapseD50_20')
    edges20.to_csv(OUT/'06_oos_train_edges_ret20.csv',index=False)
    cells20.to_csv(OUT/'07_oos_cells_ret20_x_collapse50.csv',index=False)
    pooled20.to_csv(OUT/'08_oos_moderate_rules_ret20.csv',index=False)

    lines=[]
    lines.append('PARADISE LOST — MOUNTAIN HIGH / VALLEY DEEP + OOS')
    lines.append('A+ v2 fixed setup; forward drawdowns ignore the 1ATR stop so this tests price-path depth separately from tradable PF.')
    lines.append(f'N={len(z)} from {z.SignalDate.min().date()} to {z.SignalDate.max().date()}')
    lines.append('')
    lines.append('SPEARMAN — mountain feature vs future path')
    key_targets=['FAV_pct_5','FAV_pct_10','FAV_pct_20','FAV_ATR_10','FAV_ATR_20','ADV_pct_10','ADV_pct_20','PeakToLow_pct_20']
    show=ct[ct['Target'].isin(key_targets)].copy()
    lines.append(show.pivot(index='Feature',columns='Target',values='Spearman').round(3).to_string())
    lines.append('')
    lines.append('QUINTILES — RET60')
    cols=['Feature','Q','N','FeatureMean','FAV_pct_10_mean','FAV_pct_20_mean','FAV_ATR_20_mean','ADV_pct_10_mean','ADV_pct_20_mean','PeakToLow_pct_20_mean']
    lines.append(qt[qt.Feature=='ret60'][cols].round(3).to_string(index=False))
    lines.append('')
    lines.append('QUINTILES — PeakD50_20')
    lines.append(qt[qt.Feature=='PeakD50_20'][cols].round(3).to_string(index=False))
    lines.append('')
    lines.append('OOS TRAIN EDGES — ret60 x CollapseD50_20')
    lines.append(edges.round(4).to_string(index=False))
    lines.append('')
    lines.append('OOS MODERATE RULES — ret60 x CollapseD50_20')
    lines.append(pooled.round(3).to_string(index=False))
    lines.append('')
    lines.append('OOS MODERATE RULES — ret20 x CollapseD50_20')
    lines.append(pooled20.round(3).to_string(index=False))
    lines.append('')
    lines.append('Interpretation guide: positive Spearman(feature,FAV_pct) supports mountain-high -> deeper future decline in raw percent. If FAV_pct rises but FAV_ATR does not, much of the effect is volatility scaling. If ADV_pct also rises, big mountains create deeper valleys but are harder to short with a tight stop.')
    (OUT/'summary.txt').write_text('\n'.join(lines),encoding='utf-8')
    print((OUT/'summary.txt').read_text(encoding='utf-8'))

if __name__=='__main__':
    main()
