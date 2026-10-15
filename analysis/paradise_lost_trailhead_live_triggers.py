#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'runtime/nikkei225_yahoo_clean_2016_present.csv'
TRAIL = ROOT / 'analysis_outputs/paradise_lost_trailhead_distribution/01_trailhead_events.csv'
OUT = ROOT / 'analysis_outputs/paradise_lost_trailhead_live_triggers'
OUT.mkdir(parents=True, exist_ok=True)

TRIGGERS = [
    'RECLAIM25',
    'RECLAIM50',
    'MA25_SLOPE5_UP',
    'CROSS25_50',
    'SANGO3',
    'SANGO5',
    'RECLAIM25_SANGO3',
]

BANDS = [
    ('P50-75 Mid Mountain', 0.50, 0.75),
    ('P75-90 Practical High', 0.75, 0.90),
    ('P50-90 Practical Core', 0.50, 0.90),
]


def prep(g: pd.DataFrame) -> pd.DataFrame:
    d = g.sort_values('Date').drop_duplicates('Date', keep='last').reset_index(drop=True).copy()
    for c in ['Open','High','Low','Close','Volume']:
        d[c] = pd.to_numeric(d[c], errors='coerce')
    for n in [25,50,100]:
        d[f'MA{n}'] = d.Close.rolling(n, min_periods=n).mean()
    d['MA25_slope5'] = d.MA25 - d.MA25.shift(5)
    d['PrevHigh3'] = d.High.shift(1).rolling(3, min_periods=3).max()
    d['PrevHigh5'] = d.High.shift(1).rolling(5, min_periods=5).max()
    return d


def trigger_mask(d: pd.DataFrame, name: str) -> pd.Series:
    if name == 'RECLAIM25':
        return (d.Close > d.MA25) & (d.Close.shift(1) <= d.MA25.shift(1))
    if name == 'RECLAIM50':
        return (d.Close > d.MA50) & (d.Close.shift(1) <= d.MA50.shift(1))
    if name == 'MA25_SLOPE5_UP':
        return (d.MA25_slope5 > 0) & (d.MA25_slope5.shift(1) <= 0)
    if name == 'CROSS25_50':
        return (d.MA25 > d.MA50) & (d.MA25.shift(1) <= d.MA50.shift(1))
    if name == 'SANGO3':
        return d.High > d.PrevHigh3
    if name == 'SANGO5':
        return d.High > d.PrevHigh5
    if name == 'RECLAIM25_SANGO3':
        return (d.Close > d.MA25) & (d.High > d.PrevHigh3)
    raise ValueError(name)


def first_trigger(d: pd.DataFrame, start_i: int, end_i: int, name: str):
    m = trigger_mask(d, name)
    idx = d.index[(d.index >= start_i) & (d.index <= end_i) & m.fillna(False)]
    return int(idx[0]) if len(idx) else None


def band_name(rank: float) -> str:
    if pd.isna(rank): return 'UNKNOWN'
    if rank < .50: return 'P00-50 Base'
    if rank < .75: return 'P50-75 Mid Mountain'
    if rank < .90: return 'P75-90 Practical High'
    if rank < .95: return 'P90-95 Upper High'
    if rank < .995: return 'P95-99.5 Seven'
    return 'P99.5-100 Everest'


def known_mountain_panel(src: pd.DataFrame, trail: pd.DataFrame):
    rows=[]
    maps={}
    for ticker,g in src.groupby('Ticker', sort=False):
        d=prep(g)
        maps[str(ticker)] = (d, {pd.Timestamp(x):i for i,x in enumerate(d.Date)})

    for _,r in trail.iterrows():
        rank=float(r.CompositePctRank)
        if not (0.50 <= rank < 0.90):
            continue
        ticker=str(r.Ticker)
        if ticker not in maps: continue
        d,pos=maps[ticker]
        bi=pos.get(pd.Timestamp(r.TrailheadDate)); pi=pos.get(pd.Timestamp(r.PeakDate))
        if bi is None or pi is None or pi <= bi+1: continue
        low=float(r.TrailheadLow); peak=float(r.PeakHigh)
        for trig in TRIGGERS:
            si=first_trigger(d, bi+1, pi-1, trig)
            if si is None or si+1 > pi: continue
            ei=si+1
            entry=float(d.at[ei,'Open'])
            if not np.isfinite(entry) or entry <= 0: continue
            remaining_pct=(peak/entry-1)*100.0
            gained_pct=(entry/low-1)*100.0
            total_move=peak-low
            remaining_frac=(peak-entry)/total_move if total_move>0 else np.nan
            window=d.loc[si:pi]
            retest2=(window.Low.min() <= low*1.02) if len(window) else False
            rows.append({
                'Ticker':ticker,'TrailheadDate':pd.Timestamp(r.TrailheadDate),'PeakDate':pd.Timestamp(r.PeakDate),
                'Band':band_name(rank),'CompositePctRank':rank,'Trigger':trig,
                'SignalDate':pd.Timestamp(d.at[si,'Date']),'EntryDate':pd.Timestamp(d.at[ei,'Date']),
                'DelaySignal':si-bi,'DelayEntry':ei-bi,'DaysEntryToPeak':pi-ei,
                'TrailheadLow':low,'EntryOpen':entry,'PeakHigh':peak,
                'GainUsedPct':gained_pct,'RemainingUpsidePct':remaining_pct,
                'RemainingClimbFraction':remaining_frac,'RetestWithin2PctAfterSignal':bool(retest2),
                'Signal_d25':(float(d.at[si,'Close'])/float(d.at[si,'MA25'])-1)*100 if np.isfinite(d.at[si,'MA25']) else np.nan,
                'Signal_d50':(float(d.at[si,'Close'])/float(d.at[si,'MA50'])-1)*100 if np.isfinite(d.at[si,'MA50']) else np.nan,
            })
    z=pd.DataFrame(rows)
    return z


def summarize_known(z: pd.DataFrame, trail: pd.DataFrame):
    rows=[]
    for bname,lo,hi in BANDS:
        base=trail[(trail.CompositePctRank>=lo)&(trail.CompositePctRank<hi)]
        for trig in TRIGGERS:
            g=z[(z.CompositePctRank>=lo)&(z.CompositePctRank<hi)&(z.Trigger==trig)]
            rows.append({
                'Band':bname,'Trigger':trig,'BaseN':len(base),'TriggeredN':len(g),
                'DetectionRate':len(g)/len(base) if len(base) else np.nan,
                'DelayEntryMedian':g.DelayEntry.median() if len(g) else np.nan,
                'GainUsedPctMedian':g.GainUsedPct.median() if len(g) else np.nan,
                'RemainingUpsidePctMedian':g.RemainingUpsidePct.median() if len(g) else np.nan,
                'RemainingClimbFractionMedian':g.RemainingClimbFraction.median() if len(g) else np.nan,
                'DaysEntryToPeakMedian':g.DaysEntryToPeak.median() if len(g) else np.nan,
                'Retest2PctRate':g.RetestWithin2PctAfterSignal.mean() if len(g) else np.nan,
                'SignalD50Median':g.Signal_d50.median() if len(g) else np.nan,
            })
    return pd.DataFrame(rows)


def prospective_candidates(src: pd.DataFrame):
    """Live-ish bottom-candidate universe.
    Candidate low must be the lowest low of the preceding 10 sessions and survive
    three subsequent sessions without an undercut. It becomes known at i+3.
    After accepting a candidate, suppress new candidates for 20 sessions.
    Outcome is purely forward: max gain from candidate low over next 120 sessions.
    """
    rows=[]
    for ticker,g in src.groupby('Ticker', sort=False):
        d=prep(g)
        if len(d)<180: continue
        prior10=d.Low.shift(1).rolling(10,min_periods=10).min()
        raw=(d.Low <= prior10)
        last_confirm=-999
        for i in range(110,len(d)-125):
            if not bool(raw.iloc[i]): continue
            confirm=i+3
            if confirm-last_confirm < 20: continue
            if float(d.loc[i+1:confirm,'Low'].min()) < float(d.at[i,'Low']):
                continue
            last_confirm=confirm
            low=float(d.at[i,'Low'])
            fwd=d.loc[i+1:i+120]
            maxhigh=float(fwd.High.max())
            gain120=(maxhigh/low-1)*100.0
            # Max drawdown after confirmation before a strong move develops.
            dd20=(float(d.loc[confirm:min(confirm+19,len(d)-1),'Low'].min())/low-1)*100.0
            for trig in TRIGGERS:
                si=first_trigger(d, confirm, min(confirm+40,len(d)-2), trig)
                if si is None: continue
                rows.append({
                    'Ticker':str(ticker),'BottomDate':pd.Timestamp(d.at[i,'Date']),
                    'ConfirmDate':pd.Timestamp(d.at[confirm,'Date']),'Trigger':trig,
                    'SignalDate':pd.Timestamp(d.at[si,'Date']),'DelayFromBottom':si-i,
                    'ForwardGain120Pct':gain120,'GainGE40':gain120>=40.0,'GainGE55':gain120>=55.0,
                    'BottomUndercut20Pct':dd20,
                })
    return pd.DataFrame(rows)


def summarize_candidates(c: pd.DataFrame):
    rows=[]
    # reconstruct denominator as unique bottom candidates represented by any trigger
    # and also provide per-trigger absolute counts, so trigger scarcity is visible.
    union=c[['Ticker','BottomDate']].drop_duplicates()
    for trig in TRIGGERS:
        g=c[c.Trigger==trig]
        rows.append({
            'Trigger':trig,'TriggeredCandidates':g[['Ticker','BottomDate']].drop_duplicates().shape[0],
            'UnionCandidateN':len(union),
            'TriggerCoverageVsUnion':g[['Ticker','BottomDate']].drop_duplicates().shape[0]/len(union) if len(union) else np.nan,
            'Gain40Precision':g.GainGE40.mean() if len(g) else np.nan,
            'Gain55Precision':g.GainGE55.mean() if len(g) else np.nan,
            'ForwardGain120Median':g.ForwardGain120Pct.median() if len(g) else np.nan,
            'ForwardGain120P75':g.ForwardGain120Pct.quantile(.75) if len(g) else np.nan,
            'DelayMedian':g.DelayFromBottom.median() if len(g) else np.nan,
            'Undercut20MedianPct':g.BottomUndercut20Pct.median() if len(g) else np.nan,
        })
    return pd.DataFrame(rows)


def main():
    src=pd.read_csv(SRC)
    src['Date']=pd.to_datetime(src.Date,errors='coerce')
    trail=pd.read_csv(TRAIL)
    trail['TrailheadDate']=pd.to_datetime(trail.TrailheadDate,errors='coerce')
    trail['PeakDate']=pd.to_datetime(trail.PeakDate,errors='coerce')
    trail['CompositePctRank']=pd.to_numeric(trail.CompositePctRank,errors='coerce')

    known=known_mountain_panel(src,trail)
    known.to_csv(OUT/'01_known_practical_mountain_triggers.csv',index=False)
    ks=summarize_known(known,trail)
    ks.to_csv(OUT/'02_known_trigger_summary.csv',index=False)

    cand=prospective_candidates(src)
    cand.to_csv(OUT/'03_prospective_bottom_candidates.csv',index=False)
    cs=summarize_candidates(cand)
    cs.to_csv(OUT/'04_prospective_trigger_summary.csv',index=False)

    lines=[]
    lines.append('PARADISE LOST — PRACTICAL MOUNTAIN LIVE TRAILHEAD TRIGGERS')
    lines.append('Known-mountain panel: ex-post P50-90 True120 mountains only; compares how early each point-in-time trigger recognizes the climb.')
    lines.append('Entry convention for remaining-upside metrics = next session open after trigger signal.')
    lines.append('Prospective candidate panel: 10-session low that is not undercut for the next 3 sessions; candidate becomes known on day +3; same ticker candidates suppressed for 20 sessions. Trigger searched through +40 sessions. Outcome = max High over +120 sessions from candidate low. This panel does NOT require a future True120 top.')
    lines.append('SANGO3/5 = current High exceeds prior 3/5-session High. RECLAIM25_SANGO3 = Close>25MA and SANGO3 on same signal day.')
    lines.append('')
    lines.append('KNOWN PRACTICAL MOUNTAINS — DETECTION / TIMING / REMAINING UPSIDE')
    lines.append(ks.round(3).to_string(index=False))
    lines.append('')
    lines.append('PROSPECTIVE BOTTOM CANDIDATES — FALSE-POSITIVE / GROWTH FILTER')
    lines.append(cs.round(3).to_string(index=False))
    (OUT/'summary.txt').write_text('\n'.join(lines),encoding='utf-8')
    print((OUT/'summary.txt').read_text(encoding='utf-8'))

if __name__=='__main__':
    main()
