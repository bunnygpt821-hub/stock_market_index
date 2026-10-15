#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'runtime/nikkei225_yahoo_clean_2016_present.csv'
TOPS = ROOT / 'analysis_outputs/paradise_lost_true_tops/01_cycle_top_events.csv'
OUT = ROOT / 'analysis_outputs/paradise_lost_altitude_by_regime'
OUT.mkdir(parents=True, exist_ok=True)

ALT_FEATURES = [
    'd25','d50','d100','ret20','ret60',
    'GainFromLow60_pct','GainFromLow120_pct','GainFromLow120_ATR'
]
CORE = ['d50','d100','ret60','GainFromLow120_pct']
REGIME_ORDER = [
    'Broad Expansion','Expansion Overheat','Thin-Rotation',
    'Rollover Early','Fast Damage','Risk-Off','Other/Transition'
]


def load_src() -> pd.DataFrame:
    s = pd.read_csv(SRC)
    s['Date'] = pd.to_datetime(s['Date'], errors='coerce')
    for c in ['Open','High','Low','Close','Volume']:
        s[c] = pd.to_numeric(s[c], errors='coerce')
    return s.dropna(subset=['Ticker','Date','Close']).copy()


def build_breadth(src: pd.DataFrame) -> pd.DataFrame:
    frames=[]
    for ticker,g in src.groupby('Ticker', sort=False):
        d = g.sort_values('Date').drop_duplicates('Date', keep='last').copy()
        for n in [25,50,100]:
            d[f'MA{n}'] = d.Close.rolling(n, min_periods=n).mean()
        d['above25'] = d.Close > d.MA25
        d['ma50gt100'] = d.MA50 > d.MA100
        d['full_up'] = (d.Close > d.MA25) & (d.MA25 > d.MA50) & (d.MA50 > d.MA100)
        valid = d[['MA25','MA50','MA100']].notna().all(axis=1)
        d.loc[~valid, ['above25','ma50gt100','full_up']] = np.nan
        frames.append(d[['Date','Ticker','above25','ma50gt100','full_up']])
    x = pd.concat(frames, ignore_index=True)
    rows=[]
    for dt,g in x.groupby('Date'):
        row={'Date':dt,'UniverseN':g.Ticker.nunique()}
        for c,out in [('above25','Above25'),('ma50gt100','MA50gt100'),('full_up','FullUp')]:
            v=g[c].dropna()
            row[out] = 100.0*pd.to_numeric(v, errors='coerce').mean() if len(v) else np.nan
        row['PropagationGap'] = row['MA50gt100'] - row['Above25'] if np.isfinite(row['MA50gt100']) and np.isfinite(row['Above25']) else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values('Date').reset_index(drop=True)


def download_topix() -> pd.DataFrame:
    q = yf.download('^TOPX', start='2015-01-01', end='2026-10-01', progress=False, auto_adjust=False)
    if q is None or len(q)==0:
        raise RuntimeError('TOPIX ^TOPX download failed; refusing to substitute an ETF because regime thresholds depend on the index level/MA state.')
    if isinstance(q.columns, pd.MultiIndex):
        q.columns=[c[0] for c in q.columns]
    q=q.reset_index(); q['Date']=pd.to_datetime(q['Date'], errors='coerce'); q['Close']=pd.to_numeric(q['Close'], errors='coerce')
    q=q.dropna(subset=['Date','Close']).sort_values('Date').drop_duplicates('Date', keep='last')
    for n in [50,100]:
        q[f'MA{n}']=q.Close.rolling(n,min_periods=n).mean()
    q['TOPIX_d100']=(q.Close/q.MA100-1.0)*100.0
    return q[['Date','Close','MA50','MA100','TOPIX_d100']].rename(columns={'Close':'TOPIX','MA50':'TOPIX_MA50','MA100':'TOPIX_MA100'})


def classify(row) -> str:
    a25=row.Above25; fu=row.FullUp; gap=row.PropagationGap
    tp=row.TOPIX; m50=row.TOPIX_MA50; m100=row.TOPIX_MA100; d100=row.TOPIX_d100
    if any(pd.isna(v) for v in [a25,fu,gap,tp,m50,m100,d100]):
        return 'Other/Transition'
    # Historical precedence preserved: Risk-Off -> Fast Damage -> Rollover -> Overheat -> Broad -> Thin -> Other.
    if tp < m100:
        return 'Risk-Off'
    if a25 < 40 and tp < m50:
        return 'Fast Damage'
    if gap >= 25 and a25 < 50:
        return 'Rollover Early'
    broad = (a25 >= 60) and (fu >= 40) and (gap < 15) and (tp > m50)
    if broad and d100 >= 10:
        return 'Expansion Overheat'
    if broad:
        return 'Broad Expansion'
    if (a25 >= 50) and (fu < 35) and (tp > m100):
        return 'Thin-Rotation'
    return 'Other/Transition'


def tier_series(s: pd.Series):
    s=pd.to_numeric(s, errors='coerce')
    p75=s.quantile(.75); p95=s.quantile(.95); p995=s.quantile(.995)
    tier=pd.Series(index=s.index, dtype='object')
    tier.loc[s < p75]='Normal Top (<P75)'
    tier.loc[(s >= p75)&(s < p95)]='High Mountain (P75-95)'
    tier.loc[(s >= p95)&(s < p995)]='Seven Summits (P95-99.5)'
    tier.loc[s >= p995]='Everest (>=P99.5)'
    return tier, {'P75':p75,'P95':p95,'P99.5':p995}


def main():
    src=load_src()
    breadth=build_breadth(src)
    topix=download_topix()
    regime=breadth.merge(topix,on='Date',how='left')
    regime['Regime']=regime.apply(classify,axis=1)
    regime.to_csv(OUT/'00_daily_regimes.csv',index=False)

    tops=pd.read_csv(TOPS)
    tops['PeakDate']=pd.to_datetime(tops['PeakDate'], errors='coerce')
    true=tops[tops['True120'].astype(str).str.lower().isin(['true','1'])].copy()
    z=true.merge(regime,left_on='PeakDate',right_on='Date',how='left').drop(columns=['Date'])

    # Percentile tiers for each altitude metric, defined on the full True120 top distribution.
    thresholds=[]; tier_rows=[]
    for f in ALT_FEATURES:
        z[f'{f}_pct_rank']=pd.to_numeric(z[f],errors='coerce').rank(pct=True,method='average')
        z[f'{f}_tier'], th=tier_series(z[f])
        thresholds.append({'Feature':f,**th})
        for tier,g in z.groupby(f'{f}_tier',dropna=False):
            if pd.isna(tier): continue
            total=len(g)
            for reg in REGIME_ORDER:
                n=int((g.Regime==reg).sum())
                tier_rows.append({'Feature':f,'Tier':tier,'Regime':reg,'N':n,'TierN':total,
                                  'ShareWithinTier':n/total if total else np.nan})
    pd.DataFrame(thresholds).to_csv(OUT/'01_altitude_thresholds.csv',index=False)
    pd.DataFrame(tier_rows).to_csv(OUT/'02_tier_by_regime.csv',index=False)

    # Composite altitude: average percentile across d50,d100,ret60,GainFromLow120_pct.
    z['CompositeRaw']=z[[f'{f}_pct_rank' for f in CORE]].mean(axis=1)
    z['CompositePctRank']=z.CompositeRaw.rank(pct=True,method='average')
    comp_tier,comp_th=tier_series(z.CompositeRaw)
    z['CompositeTier']=comp_tier

    # ANY-core extreme flags: useful for "any Everest-like metric".
    for q,label in [(0.95,'AnyCoreP95'),(0.995,'AnyCoreP995')]:
        flags=[]
        for f in CORE:
            flags.append(z[f'{f}_pct_rank']>=q)
        z[label]=pd.concat(flags,axis=1).any(axis=1)

    z.to_csv(OUT/'03_true120_tops_with_regime.csv',index=False)

    # Regime summary, including maxima: directly tests whether Thin/Damage ever reaches monster altitude.
    summary=[]
    for reg in REGIME_ORDER:
        g=z[z.Regime==reg]
        rec={'Regime':reg,'N':len(g),'ShareOfAllTops':len(g)/len(z) if len(z) else np.nan,
             'CompositeMedianPctRank':g.CompositePctRank.median() if len(g) else np.nan,
             'SevenPlusCompositeN':int(g.CompositeTier.isin(['Seven Summits (P95-99.5)','Everest (>=P99.5)']).sum()) if len(g) else 0,
             'EverestCompositeN':int((g.CompositeTier=='Everest (>=P99.5)').sum()) if len(g) else 0,
             'AnyCoreP95_N':int(g.AnyCoreP95.sum()) if len(g) else 0,
             'AnyCoreP995_N':int(g.AnyCoreP995.sum()) if len(g) else 0}
        for f in ALT_FEATURES:
            rec[f'{f}_median']=g[f].median() if len(g) else np.nan
            rec[f'{f}_P95']=g[f].quantile(.95) if len(g)>=5 else np.nan
            rec[f'{f}_max']=g[f].max() if len(g) else np.nan
        summary.append(rec)
    summ=pd.DataFrame(summary)
    summ.to_csv(OUT/'04_regime_altitude_summary.csv',index=False)

    # Composite tier distribution by regime.
    comp=[]
    for tier,g in z.groupby('CompositeTier'):
        for reg in REGIME_ORDER:
            n=int((g.Regime==reg).sum()); comp.append({'Tier':tier,'Regime':reg,'N':n,'TierN':len(g),'ShareWithinTier':n/len(g) if len(g) else np.nan})
    comp=pd.DataFrame(comp); comp.to_csv(OUT/'05_composite_tier_by_regime.csv',index=False)

    # Within-regime composition.
    within=[]
    for reg,g in z.groupby('Regime'):
        for tier,n in g.CompositeTier.value_counts(dropna=False).items():
            within.append({'Regime':reg,'Tier':tier,'N':int(n),'RegimeN':len(g),'ShareWithinRegime':n/len(g)})
    pd.DataFrame(within).to_csv(OUT/'06_composite_within_regime.csv',index=False)

    # Extreme event list.
    extreme=z[(z.CompositePctRank>=.95)|z.AnyCoreP995].copy().sort_values(['CompositePctRank','GainFromLow120_pct'],ascending=False)
    extreme.to_csv(OUT/'07_extreme_events.csv',index=False)

    # Cross-tab for concise summary.
    pivot=comp.pivot(index='Tier',columns='Regime',values='N').fillna(0).astype(int)
    lines=[]
    lines.append('PARADISE LOST — TRUE120 TOP ALTITUDE x MARKET REGIME')
    lines.append('Regime is point-in-time at PeakDate. Top labels are ex-post True120, so this studies where actual cycle tops formed, not a live entry rule.')
    lines.append('Breadth uses the current Nikkei225 panel historically, so survivorship bias remains.')
    lines.append('Regime rules: Risk-Off TOPIX<100MA; Fast Damage Above25<40 & TOPIX<50MA; Rollover Gap>=25 & Above25<50; Overheat=Broad + TOPIX/100MA>=+10%; Broad Above25>=60 & FullUp>=40 & Gap<15 & TOPIX>50MA; Thin Above25>=50 & FullUp<35 & TOPIX>100MA; precedence in that order.')
    lines.append(f'True120 tops={len(z)}; unmatched regime={int(z.Regime.isna().sum())}')
    lines.append('')
    lines.append('ALTITUDE THRESHOLDS')
    lines.append(pd.DataFrame(thresholds).round(3).to_string(index=False))
    lines.append('')
    lines.append('COMPOSITE TIER x REGIME — COUNTS')
    lines.append(pivot.to_string())
    lines.append('')
    lines.append('REGIME SUMMARY — CORE MEDIAN / P95 / MAX + EXTREME COUNTS')
    cols=['Regime','N','ShareOfAllTops','CompositeMedianPctRank','SevenPlusCompositeN','EverestCompositeN','AnyCoreP95_N','AnyCoreP995_N',
          'd50_median','d50_P95','d50_max','d100_median','d100_P95','d100_max','ret60_median','ret60_P95','ret60_max','GainFromLow120_pct_median','GainFromLow120_pct_P95','GainFromLow120_pct_max']
    lines.append(summ[cols].round(3).to_string(index=False))
    lines.append('')
    lines.append('COMPOSITE TIER DISTRIBUTION — SHARE WITHIN TIER')
    lines.append(comp.round(3).to_string(index=False))
    lines.append('')
    lines.append('TOP 30 EXTREMES')
    showcols=['Ticker','PeakDate','Regime','d25','d50','d100','ret20','ret60','GainFromLow120_pct','GainFromLow120_ATR','CompositePctRank','CompositeTier','FutureDD120_pct']
    lines.append(extreme[showcols].head(30).round(3).to_string(index=False))
    (OUT/'summary.txt').write_text('\n'.join(lines),encoding='utf-8')
    print((OUT/'summary.txt').read_text(encoding='utf-8'))


if __name__=='__main__':
    main()
