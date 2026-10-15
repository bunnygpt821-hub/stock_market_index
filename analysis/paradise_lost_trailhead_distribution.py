#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'runtime/nikkei225_yahoo_clean_2016_present.csv'
TOPS = ROOT / 'analysis_outputs/paradise_lost_altitude_by_regime/03_true120_tops_with_regime.csv'
REG = ROOT / 'analysis_outputs/paradise_lost_altitude_by_regime/00_daily_regimes.csv'
OUT = ROOT / 'analysis_outputs/paradise_lost_trailhead_distribution'
OUT.mkdir(parents=True, exist_ok=True)

TIERS = [
    'Normal Top (<P75)',
    'High Mountain (P75-95)',
    'Seven Summits (P95-99.5)',
    'Everest (>=P99.5)',
]
REGIMES = [
    'Broad Expansion','Expansion Overheat','Thin-Rotation',
    'Rollover Early','Fast Damage','Risk-Off','Other/Transition'
]


def prep(g: pd.DataFrame) -> pd.DataFrame:
    d = g.sort_values('Date').drop_duplicates('Date', keep='last').reset_index(drop=True).copy()
    for c in ['Open','High','Low','Close','Volume']:
        d[c] = pd.to_numeric(d[c], errors='coerce')
    pc = d.Close.shift(1)
    tr = pd.concat([(d.High-d.Low),(d.High-pc).abs(),(d.Low-pc).abs()],axis=1).max(axis=1)
    d['ATR20'] = tr.rolling(20,min_periods=20).mean()
    for n in [25,50,100]:
        d[f'MA{n}'] = d.Close.rolling(n,min_periods=n).mean()
        d[f'Close_d{n}'] = (d.Close/d[f'MA{n}']-1)*100
        d[f'Low_d{n}'] = (d.Low/d[f'MA{n}']-1)*100
    for n in [20,60]:
        d[f'ret{n}'] = (d.Close/d.Close.shift(n)-1)*100
    return d


def main():
    src = pd.read_csv(SRC)
    src['Date'] = pd.to_datetime(src['Date'], errors='coerce')
    tops = pd.read_csv(TOPS)
    tops['PeakDate'] = pd.to_datetime(tops['PeakDate'], errors='coerce')
    reg = pd.read_csv(REG)
    reg['Date'] = pd.to_datetime(reg['Date'], errors='coerce')
    regcols = ['Date','Regime','Above25','FullUp','PropagationGap','TOPIX','TOPIX_MA50','TOPIX_MA100','TOPIX_d100']
    reg = reg[[c for c in regcols if c in reg.columns]].drop_duplicates('Date')

    top_lookup = {(str(r.Ticker), pd.Timestamp(r.PeakDate)): r for _,r in tops.iterrows()}
    rows=[]
    for ticker,g in src.groupby('Ticker', sort=False):
        tkr=str(ticker)
        relevant=tops[tops.Ticker.astype(str).eq(tkr)]
        if relevant.empty: continue
        d=prep(g)
        idx_by_date={pd.Timestamp(dt):i for i,dt in enumerate(d.Date)}
        for _,tr in relevant.iterrows():
            peakdate=pd.Timestamp(tr.PeakDate)
            i=idx_by_date.get(peakdate)
            if i is None or i<120: continue
            lo_slice=d.loc[i-120:i-1,'Low']
            if lo_slice.dropna().empty: continue
            b=int(lo_slice.idxmin())
            base=d.loc[b]
            peak=d.loc[i]
            rec={
                'Ticker':tkr,
                'TrailheadDate':pd.Timestamp(base.Date),
                'PeakDate':peakdate,
                'ClimbSessions':i-b,
                'TrailheadLow':float(base.Low),
                'TrailheadClose':float(base.Close),
                'PeakHigh':float(peak.High),
                'ClimbPct':(float(peak.High)/float(base.Low)-1)*100,
                'CompositeTier':tr.CompositeTier,
                'CompositePctRank':tr.CompositePctRank,
                'Peak_d50':tr.d50,'Peak_d100':tr.d100,'Peak_ret60':tr.ret60,
                'Peak_GainFromLow120_pct':tr.GainFromLow120_pct,
                'FutureDD120_pct':tr.FutureDD120_pct,
            }
            for c in ['Close_d25','Close_d50','Close_d100','Low_d25','Low_d50','Low_d100','ret20','ret60','ATR20']:
                rec['Trail_'+c]=base[c]
            rows.append(rec)

    z=pd.DataFrame(rows)
    z=z.merge(reg.add_prefix('TrailReg_'),left_on='TrailheadDate',right_on='TrailReg_Date',how='left')
    z=z.drop(columns=['TrailReg_Date'],errors='ignore')
    z.to_csv(OUT/'01_trailhead_events.csv',index=False)

    # Percentile distribution of trailhead altitude by eventual summit tier.
    feats=['Trail_Close_d25','Trail_Close_d50','Trail_Close_d100',
           'Trail_Low_d25','Trail_Low_d50','Trail_Low_d100',
           'Trail_ret20','Trail_ret60','ClimbSessions','ClimbPct']
    rows=[]
    for tier in TIERS:
        g=z[z.CompositeTier.eq(tier)]
        for f in feats:
            s=pd.to_numeric(g[f],errors='coerce').replace([np.inf,-np.inf],np.nan).dropna()
            if s.empty: continue
            rows.append({
                'Tier':tier,'Feature':f,'N':len(s),'Mean':s.mean(),
                'P10':s.quantile(.10),'P25':s.quantile(.25),'P50':s.quantile(.50),
                'P75':s.quantile(.75),'P90':s.quantile(.90),'P95':s.quantile(.95),
                'Min':s.min(),'Max':s.max(),
            })
    dist=pd.DataFrame(rows); dist.to_csv(OUT/'02_trailhead_altitude_percentiles.csv',index=False)

    # Trailhead market regime distribution within each eventual summit tier.
    rr=[]
    for tier in TIERS:
        g=z[z.CompositeTier.eq(tier)]
        for rg in REGIMES:
            n=int((g.TrailReg_Regime==rg).sum())
            rr.append({'Tier':tier,'TrailheadRegime':rg,'N':n,'TierN':len(g),'ShareWithinTier':n/len(g) if len(g) else np.nan})
    rr=pd.DataFrame(rr); rr.to_csv(OUT/'03_trailhead_regime_by_summit_tier.csv',index=False)

    # Among True120 tops originating in each trailhead regime, what eventual height did they reach?
    rc=[]
    for rg in REGIMES:
        g=z[z.TrailReg_Regime.eq(rg)]
        for tier in TIERS:
            n=int((g.CompositeTier==tier).sum())
            rc.append({'TrailheadRegime':rg,'Tier':tier,'N':n,'RegimeN':len(g),'ShareWithinRegime':n/len(g) if len(g) else np.nan})
    rc=pd.DataFrame(rc); rc.to_csv(OUT/'04_summit_tier_within_trailhead_regime.csv',index=False)

    # Quartiles of trailhead altitude vs final summit tier.
    qb=[]
    for f in ['Trail_Close_d25','Trail_Close_d50','Trail_Close_d100','Trail_Low_d50','Trail_Low_d100','Trail_ret60']:
        valid=z[[f,'CompositeTier']].dropna().copy()
        try:
            valid['Q']=pd.qcut(valid[f],4,duplicates='drop')
        except Exception:
            continue
        for q,g in valid.groupby('Q',observed=True):
            for tier in TIERS:
                n=int((g.CompositeTier==tier).sum())
                qb.append({'Feature':f,'Quartile':str(q),'Tier':tier,'N':n,'QuartileN':len(g),'Share':n/len(g)})
    qb=pd.DataFrame(qb); qb.to_csv(OUT/'05_trailhead_quartile_vs_summit_tier.csv',index=False)

    # Spearman: does starting higher/lower relate to eventual summit altitude?
    cors=[]
    for f in ['Trail_Close_d25','Trail_Close_d50','Trail_Close_d100','Trail_Low_d25','Trail_Low_d50','Trail_Low_d100','Trail_ret20','Trail_ret60','ClimbSessions']:
        x=z[[f,'CompositePctRank','Peak_GainFromLow120_pct']].replace([np.inf,-np.inf],np.nan).dropna()
        if len(x)<10: continue
        cors.append({'Feature':f,'N':len(x),
                     'Spearman_vs_CompositePctRank':x[f].corr(x.CompositePctRank,method='spearman'),
                     'Spearman_vs_ClimbPct':x[f].corr(z.loc[x.index,'ClimbPct'],method='spearman')})
    cors=pd.DataFrame(cors); cors.to_csv(OUT/'06_trailhead_correlations.csv',index=False)

    # Extreme event trailheads.
    extreme=z[z.CompositeTier.isin(['Seven Summits (P95-99.5)','Everest (>=P99.5)'])].sort_values('CompositePctRank',ascending=False)
    extreme.to_csv(OUT/'07_extreme_trailheads.csv',index=False)

    lines=[]
    lines.append('PARADISE LOST — TRAILHEAD ALTITUDE DISTRIBUTION')
    lines.append('Trailhead = lowest Low in the 120 trading sessions immediately before PeakDate, exactly the valley underlying GainFromLow120 at the summit.')
    lines.append('This is ex-post descriptive labeling, not a live start signal. Current-Nikkei225 survivorship bias remains.')
    lines.append(f'True120 tops matched={len(z)} / source tops={len(tops)}')
    lines.append('')
    lines.append('TRAILHEAD ALTITUDE BY EVENTUAL SUMMIT TIER — MEDIANS')
    show=dist[dist.Feature.isin(['Trail_Close_d25','Trail_Close_d50','Trail_Close_d100','Trail_Low_d50','Trail_Low_d100','Trail_ret20','Trail_ret60','ClimbSessions','ClimbPct'])]
    med=show.pivot(index='Tier',columns='Feature',values='P50').reindex(TIERS)
    lines.append(med.round(3).to_string())
    lines.append('')
    lines.append('TRAILHEAD ALTITUDE BY EVENTUAL SUMMIT TIER — P10/P25/P50/P75/P90')
    lines.append(show[['Tier','Feature','N','P10','P25','P50','P75','P90']].round(3).to_string(index=False))
    lines.append('')
    lines.append('TRAILHEAD REGIME DISTRIBUTION WITHIN EVENTUAL SUMMIT TIER')
    p=rr.pivot(index='Tier',columns='TrailheadRegime',values='ShareWithinTier').reindex(TIERS)
    lines.append(p.round(3).to_string())
    lines.append('')
    lines.append('EVENTUAL SUMMIT TIER WITHIN TRAILHEAD REGIME')
    p2=rc.pivot(index='TrailheadRegime',columns='Tier',values='ShareWithinRegime').reindex(REGIMES)
    lines.append(p2.round(3).to_string())
    lines.append('')
    lines.append('TRAILHEAD ALTITUDE CORRELATIONS')
    lines.append(cors.round(3).to_string(index=False))
    lines.append('')
    lines.append('SEVEN SUMMITS + EVEREST TRAILHEADS')
    cols=['Ticker','TrailheadDate','PeakDate','ClimbSessions','TrailReg_Regime',
          'Trail_Close_d25','Trail_Close_d50','Trail_Close_d100','Trail_Low_d50','Trail_Low_d100',
          'Trail_ret20','Trail_ret60','ClimbPct','CompositePctRank','CompositeTier','FutureDD120_pct']
    lines.append(extreme[cols].round(3).to_string(index=False))
    (OUT/'summary.txt').write_text('\n'.join(lines),encoding='utf-8')
    print((OUT/'summary.txt').read_text(encoding='utf-8'))

if __name__=='__main__':
    main()
