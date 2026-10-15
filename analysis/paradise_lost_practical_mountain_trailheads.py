#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TRAIL = ROOT / 'analysis_outputs/paradise_lost_trailhead_distribution/01_trailhead_events.csv'
TOPS = ROOT / 'analysis_outputs/paradise_lost_altitude_by_regime/03_true120_tops_with_regime.csv'
OUT = ROOT / 'analysis_outputs/paradise_lost_practical_mountains'
OUT.mkdir(parents=True, exist_ok=True)

BANDS = [
    ('P00-50 Base', 0.00, 0.50),
    ('P50-75 Mid Mountain', 0.50, 0.75),
    ('P75-90 Practical High', 0.75, 0.90),
    ('P90-95 Upper High', 0.90, 0.95),
    ('P95-99.5 Seven', 0.95, 0.995),
    ('P99.5-100 Everest', 0.995, 1.000001),
]
REGIMES = ['Broad Expansion','Expansion Overheat','Thin-Rotation','Rollover Early','Fast Damage','Risk-Off','Other/Transition']
FULL_YEARS = list(range(2016, 2026))


def band_name(x: float) -> str:
    if not np.isfinite(x): return 'NA'
    for name,lo,hi in BANDS:
        if lo <= x < hi: return name
    return 'NA'


def qstats(s: pd.Series) -> dict:
    x=pd.to_numeric(s,errors='coerce').replace([np.inf,-np.inf],np.nan).dropna()
    if x.empty:
        return {'Nvalid':0,'P10':np.nan,'P25':np.nan,'P50':np.nan,'P75':np.nan,'P90':np.nan,'Mean':np.nan}
    return {'Nvalid':len(x),'P10':x.quantile(.10),'P25':x.quantile(.25),'P50':x.quantile(.50),'P75':x.quantile(.75),'P90':x.quantile(.90),'Mean':x.mean()}


def main():
    tr=pd.read_csv(TRAIL)
    tr['PeakDate']=pd.to_datetime(tr['PeakDate'],errors='coerce')
    tr['TrailheadDate']=pd.to_datetime(tr['TrailheadDate'],errors='coerce')
    tops=pd.read_csv(TOPS)
    tops['PeakDate']=pd.to_datetime(tops['PeakDate'],errors='coerce')

    # Bring in the full future-drawdown and summit-altitude fields.
    keep=['Ticker','PeakDate','FutureDD20_pct','FutureDD60_pct','FutureDD120_pct','d25','d50','d100','ret20','ret60','GainFromLow120_pct','GainFromLow120_ATR']
    keep=[c for c in keep if c in tops.columns]
    z=tr.merge(tops[keep],on=['Ticker','PeakDate'],how='left',suffixes=('','_top'))
    z['Band']=z['CompositePctRank'].apply(band_name)
    z['PeakYear']=z.PeakDate.dt.year
    z.to_csv(OUT/'01_events_with_practical_bands.csv',index=False)

    # Frequency/depth summary. Full-year frequency uses 2016-2025; 2026 reported separately.
    freq=[]
    annual=[]
    for name,_,_ in BANDS:
        g=z[z.Band.eq(name)].copy()
        counts=g.groupby('PeakYear').size()
        full_counts=pd.Series({y:int(counts.get(y,0)) for y in FULL_YEARS})
        rec={
            'Band':name,'N_total':len(g),
            'FullYearMeanN':full_counts.mean(),'FullYearMedianN':full_counts.median(),
            'FullYearMinN':full_counts.min(),'FullYearMaxN':full_counts.max(),
            'YearsZero':int((full_counts==0).sum()),
            'YearsGE5':int((full_counts>=5).sum()),'YearsGE10':int((full_counts>=10).sum()),
            'N_2026_YTD':int(counts.get(2026,0)),
            'ClimbPctMedian':g.ClimbPct.median(),'ClimbSessionsMedian':g.ClimbSessions.median(),
            'PeakD50Median':g.d50.median() if 'd50' in g else np.nan,
            'PeakD100Median':g.d100.median() if 'd100' in g else np.nan,
            'PeakRet60Median':g.ret60.median() if 'ret60' in g else np.nan,
            'PeakGain120Median':g.GainFromLow120_pct.median() if 'GainFromLow120_pct' in g else np.nan,
        }
        for h in [20,60,120]:
            c=f'FutureDD{h}_pct'
            if c in g:
                rec[f'DD{h}_Median']=g[c].median(); rec[f'DD{h}_Mean']=g[c].mean(); rec[f'DD{h}_P25']=g[c].quantile(.25)
        # Descriptive opportunity mass only: not a trading expectancy.
        if 'FutureDD60_pct' in g:
            rec['FreqDepth60'] = full_counts.mean() * abs(g.FutureDD60_pct.median())
        if 'FutureDD120_pct' in g:
            rec['FreqDepth120'] = full_counts.mean() * abs(g.FutureDD120_pct.median())
        freq.append(rec)
        for y in range(2016,2027):
            annual.append({'Band':name,'Year':y,'N':int(counts.get(y,0))})
    freq=pd.DataFrame(freq); freq.to_csv(OUT/'02_frequency_depth_summary.csv',index=False)
    annual=pd.DataFrame(annual); annual.to_csv(OUT/'03_annual_counts.csv',index=False)

    # Trailhead altitude distributions by practical band.
    trail_feats=['Trail_Close_d25','Trail_Close_d50','Trail_Close_d100','Trail_Low_d50','Trail_Low_d100','Trail_ret20','Trail_ret60','ClimbSessions','ClimbPct']
    dist=[]
    for name,_,_ in BANDS:
        g=z[z.Band.eq(name)]
        for f in trail_feats:
            if f not in g: continue
            rec={'Band':name,'Feature':f,**qstats(g[f])}
            dist.append(rec)
    dist=pd.DataFrame(dist); dist.to_csv(OUT/'04_trailhead_altitude_distribution.csv',index=False)

    # Trailhead regime distribution by practical band.
    rr=[]
    for name,_,_ in BANDS:
        g=z[z.Band.eq(name)]
        for rg in REGIMES:
            n=int((g.TrailReg_Regime==rg).sum())
            rr.append({'Band':name,'TrailheadRegime':rg,'N':n,'BandN':len(g),'Share':n/len(g) if len(g) else np.nan})
    rr=pd.DataFrame(rr); rr.to_csv(OUT/'05_trailhead_regime_distribution.csv',index=False)

    # Detailed context for the likely practical zone, but without pre-choosing the winner.
    practical=z[z.CompositePctRank.between(.50,.95,inclusive='left')].copy()
    practical.to_csv(OUT/'06_p50_p95_events.csv',index=False)

    lines=[]
    lines.append('PARADISE LOST — PRACTICAL MOUNTAIN FREQUENCY x TRAILHEAD')
    lines.append('Bands are based on CompositePctRank of ex-post True120 cycle tops. This is descriptive opportunity supply, not live PF or expectancy.')
    lines.append('Full-year frequency uses 2016-2025; 2026 is YTD through source endpoint.')
    lines.append('FreqDepth = mean annual event count × absolute median future drawdown (%); use only as a rough frequency-depth proxy, not a trading metric.')
    lines.append('')
    lines.append('FREQUENCY / DEPTH')
    cols=['Band','N_total','FullYearMeanN','FullYearMedianN','FullYearMinN','FullYearMaxN','YearsZero','YearsGE5','YearsGE10','N_2026_YTD',
          'ClimbPctMedian','ClimbSessionsMedian','PeakD50Median','PeakD100Median','PeakRet60Median','PeakGain120Median',
          'DD20_Median','DD60_Median','DD120_Median','FreqDepth60','FreqDepth120']
    cols=[c for c in cols if c in freq.columns]
    lines.append(freq[cols].round(3).to_string(index=False))
    lines.append('')
    lines.append('ANNUAL COUNTS')
    piv=annual.pivot(index='Year',columns='Band',values='N')
    lines.append(piv.to_string())
    lines.append('')
    lines.append('TRAILHEAD ALTITUDE — MEDIANS')
    med=dist.pivot(index='Band',columns='Feature',values='P50').reindex([x[0] for x in BANDS])
    lines.append(med.round(3).to_string())
    lines.append('')
    lines.append('TRAILHEAD ALTITUDE — P25/P50/P75')
    show=dist[dist.Feature.isin(['Trail_Close_d25','Trail_Close_d50','Trail_Close_d100','Trail_Low_d50','Trail_Low_d100','Trail_ret20','Trail_ret60'])]
    lines.append(show[['Band','Feature','Nvalid','P25','P50','P75']].round(3).to_string(index=False))
    lines.append('')
    lines.append('TRAILHEAD REGIME SHARES')
    rp=rr.pivot(index='Band',columns='TrailheadRegime',values='Share').reindex([x[0] for x in BANDS])
    lines.append(rp.round(3).to_string())
    (OUT/'summary.txt').write_text('\n'.join(lines),encoding='utf-8')
    print((OUT/'summary.txt').read_text(encoding='utf-8'))

if __name__=='__main__':
    main()
