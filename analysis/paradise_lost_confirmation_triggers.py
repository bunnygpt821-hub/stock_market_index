#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'runtime/nikkei225_yahoo_clean_2016_present.csv'
BASE_PATH = Path(__file__).with_name('paradise_lost_v2_analysis.py')
OUT = ROOT / 'analysis_outputs/paradise_lost_confirmation'
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

TRIGGERS = [
    'BASE',
    'SIGLOW_CLOSE',
    'LOW3_CLOSE',
    'MA25_CROSS',
    'MA50_CROSS',
    'RETEST15_FAIL',
    'MA25_AND_LOW3',
]


def pf(x: pd.Series) -> float:
    r = pd.to_numeric(x, errors='coerce').dropna()
    gp = r[r > 0].sum(); gl = -r[r < 0].sum()
    if gl == 0:
        return float('inf') if gp > 0 else np.nan
    return float(gp / gl)


def stats(g: pd.DataFrame) -> dict:
    if g.empty:
        return {'N':0,'MeanR':np.nan,'PF':np.nan,'WinRate':np.nan,'StopRate':np.nan,'Hit1R':np.nan,'Hit2R':np.nan,'MeanMFE':np.nan,'MeanMAE':np.nan,'MedianDelay':np.nan}
    return {
        'N':len(g), 'MeanR':g['R'].mean(), 'PF':pf(g['R']),
        'WinRate':(g['R']>0).mean(), 'StopRate':g['stopped'].mean(),
        'Hit1R':g['hit1'].mean(), 'Hit2R':g['hit2'].mean(),
        'MeanMFE':g['MFE_R'].mean(), 'MeanMAE':g['MAE_R'].mean(),
        'MedianDelay':g['Delay'].median(),
    }


def load_source() -> pd.DataFrame:
    src=pd.read_csv(SRC)
    src['Date']=pd.to_datetime(src['Date'],errors='coerce')
    for c in ['Open','High','Low','Close','Volume']:
        if c in src.columns: src[c]=pd.to_numeric(src[c],errors='coerce')
    return src.dropna(subset=['Ticker','Date','Open','High','Low','Close']).copy()


def find_trigger(d: pd.DataFrame, anchor_i: int, trigger: str) -> int | None:
    if trigger == 'BASE':
        return anchor_i

    max10=min(anchor_i+10,len(d)-2)
    max15=min(anchor_i+15,len(d)-2)
    sig_low=float(d.at[anchor_i,'Low'])

    if trigger == 'SIGLOW_CLOSE':
        for j in range(anchor_i+1,max10+1):
            if float(d.at[j,'Close']) < sig_low:
                return j
        return None

    if trigger == 'LOW3_CLOSE':
        for j in range(anchor_i+1,max10+1):
            if j < 3: continue
            prior3=float(d.loc[j-3:j-1,'Low'].min())
            if float(d.at[j,'Close']) < prior3:
                return j
        return None

    if trigger == 'MA25_CROSS':
        for j in range(anchor_i+1,max15+1):
            if not (np.isfinite(d.at[j,'MA25']) and np.isfinite(d.at[j-1,'MA25'])): continue
            if float(d.at[j-1,'Close']) >= float(d.at[j-1,'MA25']) and float(d.at[j,'Close']) < float(d.at[j,'MA25']):
                return j
        return None

    if trigger == 'MA50_CROSS':
        for j in range(anchor_i+1,max15+1):
            if not (np.isfinite(d.at[j,'MA50']) and np.isfinite(d.at[j-1,'MA50'])): continue
            if float(d.at[j-1,'Close']) >= float(d.at[j-1,'MA50']) and float(d.at[j,'Close']) < float(d.at[j,'MA50']):
                return j
        return None

    if trigger == 'RETEST15_FAIL':
        reclaimed=False
        for j in range(anchor_i+1,max15+1):
            if not np.isfinite(d.at[j,'MA15']): continue
            if float(d.at[j,'Close']) >= float(d.at[j,'MA15']):
                reclaimed=True
                continue
            if reclaimed and j>0 and np.isfinite(d.at[j-1,'MA15']) and float(d.at[j-1,'Close']) >= float(d.at[j-1,'MA15']):
                return j
        return None

    if trigger == 'MA25_AND_LOW3':
        for j in range(anchor_i+1,max15+1):
            if j < 3 or not np.isfinite(d.at[j,'MA25']): continue
            prior3=float(d.loc[j-3:j-1,'Low'].min())
            if float(d.at[j,'Close']) < float(d.at[j,'MA25']) and float(d.at[j,'Close']) < prior3:
                return j
        return None

    raise ValueError(trigger)


def simulate(d: pd.DataFrame, trigger_i: int, hold: int, stop_atr: float) -> dict | None:
    entry_i=trigger_i+1
    exit_i=entry_i+hold-1
    if exit_i >= len(d): return None
    atr=float(d.at[trigger_i,'ATR20'])
    if not np.isfinite(atr) or atr<=0: return None
    entry=float(d.at[entry_i,'Open'])
    if not np.isfinite(entry): return None
    stop=entry+stop_atr*atr
    stopped=False; out_i=exit_i; r=np.nan
    hit1=False; hit2=False; mfe=0.0; mae=0.0
    for j in range(entry_i,exit_i+1):
        hi=float(d.at[j,'High']); lo=float(d.at[j,'Low'])
        mfe=max(mfe,(entry-lo)/atr); mae=max(mae,(hi-entry)/atr)
        stop_today=hi>=stop
        if not hit1 and not stop_today and lo<=entry-1.0*atr: hit1=True
        if not hit2 and not stop_today and lo<=entry-2.0*atr: hit2=True
        if stop_today:
            stopped=True; out_i=j; r=-stop_atr; break
    if not stopped:
        r=(entry-float(d.at[exit_i,'Close']))/atr
    return {
        'TriggerDate':d.at[trigger_i,'Date'], 'EntryDate':d.at[entry_i,'Date'], 'ExitDate':d.at[out_i,'Date'],
        'R':float(r), 'stopped':stopped, 'hit1':hit1, 'hit2':hit2,
        'MFE_R':float(mfe), 'MAE_R':float(mae), 'EntryOpen':entry, 'ATR20_trigger':atr,
    }


def main():
    src=load_source()
    anchors, prepped=plv2.build_trades(src)
    anchors=anchors.sort_values(['SignalDate','Ticker']).reset_index(drop=True)
    # Height classification is frozen at original A+ signal, before any delayed trigger.
    valid=anchors['PeakD50_20'].notna()
    ranks=anchors.loc[valid,'PeakD50_20'].rank(method='first')
    anchors.loc[valid,'PeakQ']=pd.qcut(ranks,5,labels=['Q1','Q2','Q3','Q4','Q5']).astype(str).values

    maps={}
    for ticker,g in prepped.groupby('Ticker',sort=False):
        d=g.sort_values('Date').reset_index(drop=True).copy()
        maps[str(ticker)]=(d,{pd.Timestamp(x):i for i,x in enumerate(d['Date'])})

    rows=[]
    for _,a in anchors.iterrows():
        ticker=str(a['Ticker']); sig=pd.Timestamp(a['SignalDate'])
        d,pos=maps[ticker]; ai=pos.get(sig)
        if ai is None: continue
        for trigger in TRIGGERS:
            ti=find_trigger(d,ai,trigger)
            if ti is None: continue
            delay=ti-ai
            for hold in (10,20):
                for stop_atr in (1.0,1.5):
                    s=simulate(d,ti,hold,stop_atr)
                    if s is None: continue
                    rows.append({
                        'Ticker':ticker,'SignalDate':sig,'PeakQ':a.get('PeakQ',np.nan),
                        'PeakD50_20':a['PeakD50_20'],'ret60':a['ret60'],'CollapseD50_20':a['CollapseD50_20'],
                        'Trigger':trigger,'Delay':delay,'Hold':hold,'StopATR':stop_atr,**s,
                    })
    trades=pd.DataFrame(rows)
    trades.to_csv(OUT/'confirmation_trades.csv',index=False)

    summary=[]
    for hold in (10,20):
        for stop_atr in (1.0,1.5):
            x=trades[(trades.Hold==hold)&(trades.StopATR==stop_atr)]
            for trig in TRIGGERS:
                t=x[x.Trigger==trig]
                for group,mask in [
                    ('ALL',pd.Series(True,index=t.index)),
                    ('Q1_Q3',t.PeakQ.isin(['Q1','Q2','Q3'])),
                    ('Q4_Q5',t.PeakQ.isin(['Q4','Q5'])),
                    ('Q5',t.PeakQ.eq('Q5')),
                ]:
                    g=t[mask]
                    summary.append({'Hold':hold,'StopATR':stop_atr,'Trigger':trig,'HeightGroup':group,**stats(g)})
    sm=pd.DataFrame(summary)
    sm.to_csv(OUT/'01_trigger_summary.csv',index=False)

    # OOS for high mountains Q4/Q5; trigger definitions are fixed ex ante.
    oos=[]
    x=trades[(trades.Hold==10)&(trades.StopATR==1.0)&(trades.PeakQ.isin(['Q4','Q5']))]
    for pname,start,end in PERIODS:
        p=x[(x.SignalDate>=start)&(x.SignalDate<=end)]
        for trig in TRIGGERS:
            g=p[p.Trigger==trig]
            oos.append({'Period':pname,'Trigger':trig,**stats(g)})
    oos=pd.DataFrame(oos)
    oos.to_csv(OUT/'02_oos_highmountain_q4q5.csv',index=False)

    # Trigger conversion: how many anchors actually produce a later confirmation.
    conv=[]
    highanchors=anchors[anchors.PeakQ.isin(['Q4','Q5'])]
    for trig in TRIGGERS:
        tt=trades[(trades.Hold==10)&(trades.StopATR==1.0)&(trades.Trigger==trig)&(trades.PeakQ.isin(['Q4','Q5']))]
        unique=tt[['Ticker','SignalDate']].drop_duplicates()
        conv.append({'Trigger':trig,'HighAnchors':len(highanchors),'Confirmed':len(unique),'Conversion':len(unique)/len(highanchors) if len(highanchors) else np.nan,
                     'MedianDelay':tt.Delay.median() if len(tt) else np.nan})
    conv=pd.DataFrame(conv)
    conv.to_csv(OUT/'03_trigger_conversion.csv',index=False)

    lines=[]
    lines.append('PARADISE LOST — CONFIRMATION TRIGGER TEST')
    lines.append('Height = original A+ signal PeakD50_20 quintile. Entry is delayed until confirmation; ATR and stop are reset on confirmation day.')
    lines.append('Primary comparison: Hold10 / Stop1ATR. Q4+Q5 = high mountain watchlist.')
    lines.append('')
    lines.append('CONVERSION — HIGH MOUNTAIN Q4+Q5')
    lines.append(conv.round(3).to_string(index=False))
    lines.append('')
    lines.append('PRIMARY — HOLD10 STOP1ATR')
    prim=sm[(sm.Hold==10)&(sm.StopATR==1.0)&(sm.HeightGroup.isin(['Q4_Q5','Q5']))]
    lines.append(prim.round(3).to_string(index=False))
    lines.append('')
    lines.append('SENSITIVITY — HIGH MOUNTAIN Q4+Q5')
    sens=sm[sm.HeightGroup.eq('Q4_Q5')]
    lines.append(sens.round(3).to_string(index=False))
    lines.append('')
    lines.append('OOS — HIGH MOUNTAIN Q4+Q5, HOLD10 STOP1ATR')
    lines.append(oos.round(3).to_string(index=False))
    (OUT/'summary.txt').write_text('\n'.join(lines),encoding='utf-8')
    print((OUT/'summary.txt').read_text(encoding='utf-8'))

if __name__=='__main__':
    main()
