#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'analysis_outputs/paradise_lost_altitude_by_regime'
TOPS=BASE/'03_true120_tops_with_regime.csv'
REG=BASE/'00_daily_regimes.csv'
OUT=ROOT/'analysis_outputs/paradise_lost_altitude_regime_path'
OUT.mkdir(parents=True,exist_ok=True)

OFFSETS=[-40,-20,-10,0,10,20]
TIERS=['Normal Top (<P75)','High Mountain (P75-95)','Seven Summits (P95-99.5)','Everest (>=P99.5)']
REGIMES=['Broad Expansion','Expansion Overheat','Thin-Rotation','Rollover Early','Fast Damage','Risk-Off','Other/Transition']


def main():
    tops=pd.read_csv(TOPS); tops['PeakDate']=pd.to_datetime(tops.PeakDate,errors='coerce')
    reg=pd.read_csv(REG); reg['Date']=pd.to_datetime(reg.Date,errors='coerce'); reg=reg.sort_values('Date').reset_index(drop=True)
    date_to_i={pd.Timestamp(d):i for i,d in enumerate(reg.Date)}
    rows=[]
    for _,r in tops.iterrows():
        i=date_to_i.get(pd.Timestamp(r.PeakDate))
        if i is None: continue
        for off in OFFSETS:
            j=i+off
            if j<0 or j>=len(reg): continue
            rr=reg.iloc[j]
            rows.append({
                'Ticker':r.Ticker,'PeakDate':r.PeakDate,'Tier':r.CompositeTier,
                'CompositePctRank':r.CompositePctRank,'Offset':off,
                'Date':rr.Date,'Regime':rr.Regime,'Above25':rr.Above25,'FullUp':rr.FullUp,
                'PropagationGap':rr.PropagationGap,'TOPIX_d100':rr.TOPIX_d100,
                'd50':r.d50,'d100':r.d100,'ret60':r.ret60,'GainFromLow120_pct':r.GainFromLow120_pct,
                'FutureDD120_pct':r.FutureDD120_pct,
            })
    z=pd.DataFrame(rows); z.to_csv(OUT/'01_paths_long.csv',index=False)

    # Distribution by tier and offset.
    dist=[]
    for tier in TIERS:
        for off in OFFSETS:
            g=z[(z.Tier==tier)&(z.Offset==off)]
            for rg in REGIMES:
                n=int((g.Regime==rg).sum())
                dist.append({'Tier':tier,'Offset':off,'Regime':rg,'N':n,'TierOffsetN':len(g),'Share':n/len(g) if len(g) else np.nan})
    dist=pd.DataFrame(dist); dist.to_csv(OUT/'02_regime_distribution.csv',index=False)

    # Individual Everest/Seven event paths.
    ext=z[z.Tier.isin(['Seven Summits (P95-99.5)','Everest (>=P99.5)'])].copy()
    wide=ext.pivot_table(index=['Ticker','PeakDate','Tier','CompositePctRank'],columns='Offset',values='Regime',aggfunc='first').reset_index()
    wide.to_csv(OUT/'03_extreme_paths_wide.csv',index=False)

    # Transition from -20->0 and -10->0, plus post-peak 0->+10/+20.
    trans=[]
    for tier in TIERS:
        for a,b in [(-40,-20),(-20,-10),(-20,0),(-10,0),(0,10),(0,20)]:
            x=z[(z.Tier==tier)&(z.Offset.isin([a,b]))].pivot_table(index=['Ticker','PeakDate'],columns='Offset',values='Regime',aggfunc='first').dropna()
            if a not in x.columns or b not in x.columns: continue
            vc=(x[a].astype(str)+' -> '+x[b].astype(str)).value_counts()
            for path,n in vc.items():
                trans.append({'Tier':tier,'FromOffset':a,'ToOffset':b,'Transition':path,'N':int(n),'PairN':len(x),'Share':n/len(x)})
    tr=pd.DataFrame(trans); tr.to_csv(OUT/'04_transitions.csv',index=False)

    # Broad-ish growth vs fragile regimes around peak.
    families={
        'GROWTH':['Broad Expansion','Expansion Overheat'],
        'THIN':['Thin-Rotation'],
        'FRAGILE':['Rollover Early','Fast Damage','Risk-Off'],
        'OTHER':['Other/Transition'],
    }
    fam=[]
    for tier in TIERS:
        for off in OFFSETS:
            g=z[(z.Tier==tier)&(z.Offset==off)]
            for famname,rgs in families.items():
                n=int(g.Regime.isin(rgs).sum())
                fam.append({'Tier':tier,'Offset':off,'Family':famname,'N':n,'Total':len(g),'Share':n/len(g) if len(g) else np.nan})
    fam=pd.DataFrame(fam); fam.to_csv(OUT/'05_family_path.csv',index=False)

    lines=[]
    lines.append('PARADISE LOST — ALTITUDE REGIME PATH')
    lines.append('Offsets are trading sessions relative to True120 cycle-top PeakDate. Peak labeling is ex-post; regimes are point-in-time.')
    lines.append('')
    for tier in ['Everest (>=P99.5)','Seven Summits (P95-99.5)','High Mountain (P75-95)','Normal Top (<P75)']:
        lines.append(tier)
        p=fam[fam.Tier==tier].pivot(index='Offset',columns='Family',values='Share').reindex(OFFSETS)
        lines.append(p.round(3).to_string())
        lines.append('REGIME COUNTS/SHARES')
        d=dist[dist.Tier==tier].pivot(index='Offset',columns='Regime',values='Share').reindex(OFFSETS)
        lines.append(d.round(3).to_string())
        lines.append('')
    lines.append('EXTREME EVENT PATHS')
    lines.append(wide.sort_values(['Tier','CompositePctRank'],ascending=[True,False]).to_string(index=False))
    lines.append('')
    lines.append('TOP TRANSITIONS FOR EXTREMES')
    t=tr[tr.Tier.isin(['Seven Summits (P95-99.5)','Everest (>=P99.5)'])].sort_values(['Tier','FromOffset','ToOffset','N'],ascending=[True,True,True,False])
    lines.append(t.groupby(['Tier','FromOffset','ToOffset']).head(8).round(3).to_string(index=False))
    (OUT/'summary.txt').write_text('\n'.join(lines),encoding='utf-8')
    print((OUT/'summary.txt').read_text(encoding='utf-8'))

if __name__=='__main__': main()
