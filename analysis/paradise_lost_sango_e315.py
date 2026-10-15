#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'runtime/nikkei225_yahoo_clean_2016_present.csv'
TRAIL = ROOT / 'analysis_outputs/paradise_lost_trailhead_distribution/01_trailhead_events.csv'
OUT = ROOT / 'analysis_outputs/paradise_lost_sango_e315'
OUT.mkdir(parents=True, exist_ok=True)

BASE_TRIGGERS = ['SANGO3','SANGO5','RECLAIM25']
STRATEGIES = ['SANGO3','SANGO5','SANGO3_E315','SANGO5_E315','RECLAIM25_E315']
PERIODS = [
    ('TRAIN_2016_2020', pd.Timestamp('2016-01-01'), pd.Timestamp('2020-12-31')),
    ('VALID_2021_2023', pd.Timestamp('2021-01-01'), pd.Timestamp('2023-12-31')),
    ('TEST_2024_2026', pd.Timestamp('2024-01-01'), pd.Timestamp('2026-09-18')),
]
BANDS = [
    ('P50-75 Mid Mountain', .50, .75),
    ('P75-90 Practical High', .75, .90),
    ('P50-90 Practical Core', .50, .90),
]


def prep(g: pd.DataFrame) -> pd.DataFrame:
    d = g.sort_values('Date').drop_duplicates('Date', keep='last').reset_index(drop=True).copy()
    for c in ['Open','High','Low','Close','Volume']:
        d[c] = pd.to_numeric(d[c], errors='coerce')
    pc = d.Close.shift(1)
    tr = pd.concat([(d.High-d.Low),(d.High-pc).abs(),(d.Low-pc).abs()], axis=1).max(axis=1)
    d['ATR20'] = tr.rolling(20, min_periods=20).mean()
    for n in [15,25,50,100]:
        d[f'MA{n}'] = d.Close.rolling(n, min_periods=n).mean()
    d['MA15_up1'] = d.MA15 > d.MA15.shift(1)
    d['PrevHigh3'] = d.High.shift(1).rolling(3, min_periods=3).max()
    d['PrevHigh5'] = d.High.shift(1).rolling(5, min_periods=5).max()
    return d


def trigger_mask(d: pd.DataFrame, name: str) -> pd.Series:
    if name == 'SANGO3':
        return d.High > d.PrevHigh3
    if name == 'SANGO5':
        return d.High > d.PrevHigh5
    if name == 'RECLAIM25':
        return (d.Close > d.MA25) & (d.Close.shift(1) <= d.MA25.shift(1))
    raise ValueError(name)


def first_trigger(d: pd.DataFrame, start_i: int, end_i: int, name: str):
    m = trigger_mask(d, name).fillna(False)
    idx = d.index[(d.index >= start_i) & (d.index <= end_i) & m]
    return int(idx[0]) if len(idx) else None


def strategy_base(strategy: str) -> str:
    if strategy.startswith('SANGO3'): return 'SANGO3'
    if strategy.startswith('SANGO5'): return 'SANGO5'
    if strategy.startswith('RECLAIM25'): return 'RECLAIM25'
    raise ValueError(strategy)


def is_gated(strategy: str) -> bool:
    return strategy.endswith('_E315')


def e315_state(d: pd.DataFrame, signal_i: int):
    """E+315 definition used here:
    signal -> next-session open entry. On entry+3 sessions, Close > entry price
    AND MA15 is higher than the prior session. Initial stop is 1 ATR20 from signal.
    If the stop is touched before/on the gate day, the position did not survive to E+315.
    """
    entry_i = signal_i + 1
    gate_i = entry_i + 3
    if gate_i >= len(d): return None
    atr = float(d.at[signal_i,'ATR20']); entry = float(d.at[entry_i,'Open'])
    if not np.isfinite(atr) or atr <= 0 or not np.isfinite(entry) or entry <= 0: return None
    stop = entry - atr
    survived = True
    for j in range(entry_i, gate_i+1):
        if float(d.at[j,'Low']) <= stop:
            survived = False
            break
    price_ok = float(d.at[gate_i,'Close']) > entry
    ma_ok = bool(d.at[gate_i,'MA15_up1']) if pd.notna(d.at[gate_i,'MA15_up1']) else False
    passed = survived and price_ok and ma_ok
    return {
        'Entry_i':entry_i,'Gate_i':gate_i,'ATR':atr,'Entry':entry,
        'Stop':stop,'SurvivedToGate':survived,'PriceOK':price_ok,'MA15Up':ma_ok,'E315Pass':passed,
    }


def simulate(d: pd.DataFrame, signal_i: int, hold: int, gated: bool):
    st = e315_state(d, signal_i)
    if st is None: return None
    ei=st['Entry_i']; gi=st['Gate_i']; atr=st['ATR']; entry=st['Entry']; stop=st['Stop']
    end=min(ei+hold-1, len(d)-1)
    stopped=False; gate_fail=False; r=np.nan; exit_i=end
    # Stop has precedence within a day.
    for j in range(ei, end+1):
        if float(d.at[j,'Low']) <= stop:
            stopped=True; r=-1.0; exit_i=j; break
        if gated and j == gi and not st['E315Pass']:
            gate_fail=True; r=(float(d.at[j,'Close'])-entry)/atr; exit_i=j; break
    if not stopped and not gate_fail:
        r=(float(d.at[end,'Close'])-entry)/atr
        exit_i=end
    return {
        'R':float(r),'Stopped':stopped,'GateFailExit':gate_fail,'E315Pass':bool(st['E315Pass']),
        'SurvivedToGate':bool(st['SurvivedToGate']),'Entry_i':ei,'Gate_i':gi,'Exit_i':exit_i,
        'Entry':entry,'ATR':atr,
    }


def pf(s: pd.Series):
    x=pd.to_numeric(s,errors='coerce').dropna(); gp=x[x>0].sum(); gl=-x[x<0].sum()
    if gl==0: return float('inf') if gp>0 else np.nan
    return float(gp/gl)


def seq_dd(s: pd.Series):
    x=pd.to_numeric(s,errors='coerce').fillna(0.0).to_numpy(float)
    if len(x)==0: return np.nan
    eq=np.cumsum(x); peaks=np.maximum.accumulate(np.r_[0.0,eq])[:-1]
    dd=eq-peaks
    return float(-dd.min())


def known_panel(src: pd.DataFrame, trail: pd.DataFrame):
    maps={}
    for ticker,g in src.groupby('Ticker',sort=False):
        d=prep(g); maps[str(ticker)]=(d,{pd.Timestamp(x):i for i,x in enumerate(d.Date)})
    rows=[]
    for _,r in trail.iterrows():
        rank=float(r.CompositePctRank)
        if not (.50 <= rank < .90): continue
        ticker=str(r.Ticker)
        if ticker not in maps: continue
        d,pos=maps[ticker]; bi=pos.get(pd.Timestamp(r.TrailheadDate)); pi=pos.get(pd.Timestamp(r.PeakDate))
        if bi is None or pi is None or pi <= bi+5: continue
        low=float(r.TrailheadLow); peak=float(r.PeakHigh)
        for strategy in STRATEGIES:
            base=strategy_base(strategy)
            si=first_trigger(d,bi+1,pi-1,base)
            if si is None: continue
            st=e315_state(d,si)
            if st is None: continue
            captured = True if not is_gated(strategy) else bool(st['E315Pass'] and st['Gate_i'] < pi)
            action_i=st['Entry_i'] if not is_gated(strategy) else st['Gate_i']+1
            if action_i >= len(d) or action_i > pi: action_i=None
            if action_i is not None:
                action=float(d.at[action_i,'Open'])
                rem=(peak/action-1)*100.0
                frac=(peak-action)/(peak-low) if peak>low else np.nan
                delay=action_i-bi
                days=pi-action_i
            else:
                action=rem=frac=delay=days=np.nan
            rows.append({
                'Ticker':ticker,'TrailheadDate':pd.Timestamp(r.TrailheadDate),'PeakDate':pd.Timestamp(r.PeakDate),
                'CompositePctRank':rank,'Strategy':strategy,'BaseTrigger':base,'Captured':captured,
                'SignalDate':pd.Timestamp(d.at[si,'Date']),'EntryDate':pd.Timestamp(d.at[st['Entry_i'],'Date']),
                'GateDate':pd.Timestamp(d.at[st['Gate_i'],'Date']),'E315Pass':bool(st['E315Pass']),
                'SurvivedToGate':bool(st['SurvivedToGate']),'InitialDelay':st['Entry_i']-bi,
                'ActionDelay':delay,'ActionOpen':action,'RemainingUpsidePct':rem,'RemainingClimbFraction':frac,
                'DaysActionToPeak':days,'PeakHigh':peak,'TrailheadLow':low,
            })
    return pd.DataFrame(rows)


def known_summary(z: pd.DataFrame, trail: pd.DataFrame):
    rows=[]
    for bname,lo,hi in BANDS:
        base=trail[(trail.CompositePctRank>=lo)&(trail.CompositePctRank<hi)]
        for strategy in STRATEGIES:
            g=z[(z.CompositePctRank>=lo)&(z.CompositePctRank<hi)&(z.Strategy==strategy)]
            cap=g[g.Captured]
            rows.append({
                'Band':bname,'Strategy':strategy,'BaseN':len(base),'TriggerN':len(g),
                'CaptureN':len(cap),'CaptureRate':len(cap)/len(base) if len(base) else np.nan,
                'E315PassGivenTrigger':g.E315Pass.mean() if len(g) else np.nan,
                'InitialDelayMed':g.InitialDelay.median() if len(g) else np.nan,
                'ActionDelayMed':cap.ActionDelay.median() if len(cap) else np.nan,
                'RemainingUpsideMed':cap.RemainingUpsidePct.median() if len(cap) else np.nan,
                'RemainingFracMed':cap.RemainingClimbFraction.median() if len(cap) else np.nan,
                'DaysActionToPeakMed':cap.DaysActionToPeak.median() if len(cap) else np.nan,
            })
    return pd.DataFrame(rows)


def prospective(src: pd.DataFrame):
    rows=[]
    for ticker,g in src.groupby('Ticker',sort=False):
        d=prep(g)
        if len(d)<190: continue
        prior10=d.Low.shift(1).rolling(10,min_periods=10).min(); raw=d.Low<=prior10
        last_confirm=-999
        for i in range(110,len(d)-130):
            if not bool(raw.iloc[i]): continue
            confirm=i+3
            if confirm-last_confirm<20: continue
            if float(d.loc[i+1:confirm,'Low'].min()) < float(d.at[i,'Low']): continue
            last_confirm=confirm
            bottom=float(d.at[i,'Low']); fwd=d.loc[i+1:i+120]
            maxhigh=float(fwd.High.max()); gain=(maxhigh/bottom-1)*100.0
            for strategy in STRATEGIES:
                base=strategy_base(strategy); si=first_trigger(d,confirm,min(confirm+40,len(d)-65),base)
                if si is None: continue
                st=e315_state(d,si)
                if st is None: continue
                precision_include = True if not is_gated(strategy) else bool(st['E315Pass'])
                # Remaining opportunity from the point the strategy is actionable.
                action_i=st['Entry_i'] if not is_gated(strategy) else st['Gate_i']+1
                if action_i >= len(d): continue
                max120=float(d.loc[action_i:min(action_i+119,len(d)-1),'High'].max())
                action_open=float(d.at[action_i,'Open'])
                action_fwd=(max120/action_open-1)*100.0 if action_open>0 else np.nan
                rec={
                    'Ticker':str(ticker),'BottomDate':pd.Timestamp(d.at[i,'Date']),'ConfirmDate':pd.Timestamp(d.at[confirm,'Date']),
                    'Strategy':strategy,'BaseTrigger':base,'SignalDate':pd.Timestamp(d.at[si,'Date']),
                    'EntryDate':pd.Timestamp(d.at[st['Entry_i'],'Date']),'GateDate':pd.Timestamp(d.at[st['Gate_i'],'Date']),
                    'E315Pass':bool(st['E315Pass']),'PrecisionIncluded':precision_include,
                    'BottomGain120Pct':gain,'GainGE40':gain>=40,'GainGE55':gain>=55,
                    'ActionForwardGain120Pct':action_fwd,'DelaySignal':si-i,'DelayAction':action_i-i,
                }
                for hold in [20,60]:
                    sim=simulate(d,si,hold,is_gated(strategy))
                    if sim is None: continue
                    rec[f'R{hold}']=sim['R']; rec[f'Stop{hold}']=sim['Stopped']; rec[f'GateFail{hold}']=sim['GateFailExit']
                rows.append(rec)
    return pd.DataFrame(rows)


def prospective_precision(z: pd.DataFrame):
    rows=[]
    for strategy in STRATEGIES:
        g=z[z.Strategy==strategy]
        q=g[g.PrecisionIncluded]
        rows.append({
            'Strategy':strategy,'TriggeredN':len(g),'IncludedAfterGateN':len(q),
            'GatePassRate':g.E315Pass.mean() if len(g) else np.nan,
            'Gain40Precision':q.GainGE40.mean() if len(q) else np.nan,
            'Gain55Precision':q.GainGE55.mean() if len(q) else np.nan,
            'BottomGain120Med':q.BottomGain120Pct.median() if len(q) else np.nan,
            'ActionForwardGain120Med':q.ActionForwardGain120Pct.median() if len(q) else np.nan,
            'DelayActionMed':q.DelayAction.median() if len(q) else np.nan,
        })
    return pd.DataFrame(rows)


def trade_summary(z: pd.DataFrame):
    rows=[]
    for hold in [20,60]:
        rcol=f'R{hold}'; scol=f'Stop{hold}'; gcol=f'GateFail{hold}'
        for strategy in STRATEGIES:
            base=z[z.Strategy==strategy].dropna(subset=[rcol]).sort_values('EntryDate')
            for pname,start,end in [('ALL',pd.Timestamp('2016-01-01'),pd.Timestamp('2026-09-18'))]+PERIODS:
                g=base[(base.EntryDate>=start)&(base.EntryDate<=end)]
                rows.append({
                    'Hold':hold,'Strategy':strategy,'Period':pname,'N':len(g),
                    'MeanR':g[rcol].mean() if len(g) else np.nan,'MedianR':g[rcol].median() if len(g) else np.nan,
                    'PF':pf(g[rcol]) if len(g) else np.nan,'WinRate':(g[rcol]>0).mean() if len(g) else np.nan,
                    'StopRate':g[scol].mean() if len(g) else np.nan,'GateFailExitRate':g[gcol].mean() if len(g) else np.nan,
                    'SumR':g[rcol].sum() if len(g) else np.nan,'SeqMaxDD_R':seq_dd(g[rcol]) if len(g) else np.nan,
                })
    return pd.DataFrame(rows)


def main():
    src=pd.read_csv(SRC); src['Date']=pd.to_datetime(src.Date,errors='coerce')
    trail=pd.read_csv(TRAIL); trail['TrailheadDate']=pd.to_datetime(trail.TrailheadDate,errors='coerce'); trail['PeakDate']=pd.to_datetime(trail.PeakDate,errors='coerce'); trail['CompositePctRank']=pd.to_numeric(trail.CompositePctRank,errors='coerce')

    known=known_panel(src,trail); known.to_csv(OUT/'01_known_mountain_e315.csv',index=False)
    ks=known_summary(known,trail); ks.to_csv(OUT/'02_known_summary.csv',index=False)

    pros=prospective(src); pros.to_csv(OUT/'03_prospective_e315.csv',index=False)
    ps=prospective_precision(pros); ps.to_csv(OUT/'04_precision_summary.csv',index=False)
    ts=trade_summary(pros); ts.to_csv(OUT/'05_trade_summary.csv',index=False)

    lines=[]
    lines.append('PARADISE LOST — SANGO x E+315')
    lines.append('E+315 used here: entry = next open after trigger; on entry+3 sessions, Close > entry price AND MA15 > prior-day MA15. A 1ATR20 initial stop from trigger day must survive through the gate day.')
    lines.append('Gated strategy management: enter on the original trigger. If stopped before/on E+3 => -1R. If E+315 fails without stop => exit E+3 close. If it passes => continue to the fixed 20/60-session evaluation horizon with the same 1ATR stop.')
    lines.append('R uses 1 ATR initial stop = 1R. SeqMaxDD_R is trade-sequence drawdown, not a concurrency-aware portfolio backtest.')
    lines.append('Known-mountain panel is ex-post descriptive. Prospective bottom panel does not require a future True120 top and is used for precision/PF/OOS tests.')
    lines.append('')
    lines.append('KNOWN P50-90 MOUNTAINS — CAPTURE / REMAINING MOUNTAIN')
    lines.append(ks.round(3).to_string(index=False))
    lines.append('')
    lines.append('PROSPECTIVE BOTTOMS — MOUNTAIN PRECISION')
    lines.append(ps.round(3).to_string(index=False))
    lines.append('')
    lines.append('PROSPECTIVE TRADE RESULTS — 1ATR STOP')
    lines.append(ts.round(3).to_string(index=False))
    (OUT/'summary.txt').write_text('\n'.join(lines),encoding='utf-8')
    print((OUT/'summary.txt').read_text(encoding='utf-8'))

if __name__=='__main__':
    main()
