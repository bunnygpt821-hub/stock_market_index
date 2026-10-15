#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'runtime/nikkei225_yahoo_clean_2016_present.csv'
TOPS = ROOT / 'analysis_outputs/paradise_lost_true_tops/01_cycle_top_events.csv'
OUT = ROOT / 'analysis_outputs/paradise_lost_topology'
OUT.mkdir(parents=True, exist_ok=True)

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


def load_source():
    s = pd.read_csv(SRC)
    s['Date'] = pd.to_datetime(s['Date'], errors='coerce')
    for c in ['Open','High','Low','Close','Volume']:
        s[c] = pd.to_numeric(s[c], errors='coerce')
    return s.dropna(subset=['Ticker','Date','Open','High','Low','Close']).copy()


def prepare_one(g: pd.DataFrame):
    d = g.sort_values('Date').drop_duplicates('Date', keep='last').reset_index(drop=True).copy()
    pc = d.Close.shift(1)
    tr = pd.concat([(d.High-d.Low),(d.High-pc).abs(),(d.Low-pc).abs()], axis=1).max(axis=1)
    d['ATR20'] = tr.rolling(20,min_periods=20).mean()
    return d


def local_high(d: pd.DataFrame, i: int, k: int=2) -> bool:
    if i-k < 0 or i+k >= len(d): return False
    h = float(d.at[i,'High'])
    return h >= float(d.loc[i-k:i-1,'High'].max()) and h > float(d.loc[i+1:i+k,'High'].max())


def prior_left_shoulder(d: pd.DataFrame, a_i: int):
    """Latest meaningful pivot high before A in the prior 60 sessions.
    Require at least 3 sessions separation from A and at least one >=2% or >=1ATR dip
    between P and A, so P is a genuine prior swing high rather than A's own plateau.
    """
    start = max(5, a_i-60)
    atr = float(d.at[a_i,'ATR20'])
    cands=[]
    for i in range(start, a_i-2):
        if not local_high(d,i,2): continue
        if i+1 >= a_i: continue
        p = float(d.at[i,'High'])
        low = float(d.loc[i+1:a_i-1,'Low'].min())
        dip_pct = (low/p - 1.0)*100.0
        dip_atr = (p-low)/atr if np.isfinite(atr) and atr>0 else np.nan
        if dip_pct <= -2.0 or (np.isfinite(dip_atr) and dip_atr >= 1.0):
            cands.append(i)
    return cands[-1] if cands else None


def download_index():
    q = yf.download('^N225', start='2015-01-01', end='2026-10-01', progress=False, auto_adjust=False)
    if q is None or len(q)==0:
        return pd.DataFrame(columns=['Date'])
    if isinstance(q.columns, pd.MultiIndex):
        q.columns = [c[0] for c in q.columns]
    q = q.reset_index()
    q['Date'] = pd.to_datetime(q['Date'], errors='coerce')
    q['Close'] = pd.to_numeric(q['Close'], errors='coerce')
    q = q.dropna(subset=['Date','Close']).sort_values('Date')
    out = pd.DataFrame({'ConfirmDate':q.Date,'IDXClose':q.Close})
    for n in [5,15,25,50]:
        ma = q.Close.rolling(n,min_periods=n).mean()
        out[f'IDX_MA{n}'] = ma.values
        out[f'IDX_d{n}'] = ((q.Close/ma)-1.0).values*100.0
        out[f'IDX_MA{n}_slope5'] = (ma-ma.shift(5)).values
    out['IDX_below25'] = out.IDXClose < out.IDX_MA25
    out['IDX_below50'] = out.IDXClose < out.IDX_MA50
    out['IDX_25down'] = out.IDX_MA25_slope5 < 0
    out['IDX_bearstack'] = (out.IDX_MA5<out.IDX_MA15)&(out.IDX_MA15<out.IDX_MA25)&(out.IDXClose<out.IDX_MA5)
    return out


def simulate_from_confirm(d: pd.DataFrame, confirm_i: int, hold: int=20, stop_atr: float=1.0):
    entry_i = confirm_i + 1
    exit_i = entry_i + hold - 1
    if entry_i >= len(d) or exit_i >= len(d): return None
    atr = float(d.at[confirm_i,'ATR20']); entry = float(d.at[entry_i,'Open'])
    if not np.isfinite(atr) or atr<=0 or not np.isfinite(entry) or entry<=0: return None
    stop = entry + stop_atr*atr
    hit1=hit2=False; day1=day2=np.nan; stopped=False; r=np.nan; mfe=mae=0.0
    for k,j in enumerate(range(entry_i,exit_i+1),start=1):
        hi=float(d.at[j,'High']); lo=float(d.at[j,'Low'])
        mfe=max(mfe,(entry-lo)/atr); mae=max(mae,(hi-entry)/atr)
        stop_today = hi >= stop
        if not stop_today:
            if not hit1 and lo <= entry-atr: hit1=True; day1=k
            if not hit2 and lo <= entry-2*atr: hit2=True; day2=k
        if stop_today:
            stopped=True; r=-stop_atr; break
    if not stopped:
        r=(entry-float(d.at[exit_i,'Close']))/atr
    rec={'R':float(r),'stopped':stopped,'hit1':hit1,'hit2':hit2,'DayTo1ATR':day1,'DayTo2ATR':day2,'MFE_R':mfe,'MAE_R':mae}
    for h in [5,10,20]:
        e=min(entry_i+h-1,len(d)-1); w=d.loc[entry_i:e]
        rec[f'FAV{h}_pct']=(entry-float(w.Low.min()))/entry*100.0
        rec[f'ADV{h}_pct']=(float(w.High.max())-entry)/entry*100.0
    return rec


def stats(g):
    if len(g)==0:
        return {'N':0,'MeanR':np.nan,'PF':np.nan,'WinRate':np.nan,'StopRate':np.nan,'Hit1R':np.nan,'Hit2R':np.nan,'FAV10':np.nan,'FAV20':np.nan,'ADV10':np.nan,'Day1_med':np.nan,'Day2_med':np.nan}
    return {
        'N':len(g),'MeanR':g.R.mean(),'PF':pf(g.R),'WinRate':(g.R>0).mean(),'StopRate':g.stopped.mean(),
        'Hit1R':g.hit1.mean(),'Hit2R':g.hit2.mean(),'FAV10':g.FAV10_pct.mean(),'FAV20':g.FAV20_pct.mean(),
        'ADV10':g.ADV10_pct.mean(),'Day1_med':g.DayTo1ATR.median(),'Day2_med':g.DayTo2ATR.median()
    }


def main():
    src=load_source(); tops=pd.read_csv(TOPS)
    for c in ['PeakDate','PullbackDate','LowerHighDate']:
        tops[c]=pd.to_datetime(tops[c],errors='coerce')
    tops=tops[tops.True120.astype(bool)].copy()

    maps={}
    for ticker,g in src.groupby('Ticker',sort=False):
        d=prepare_one(g); maps[str(ticker)] = (d,{pd.Timestamp(x):i for i,x in enumerate(d.Date)})

    rows=[]
    for _,t in tops.iterrows():
        ticker=str(t.Ticker)
        if ticker not in maps: continue
        d,pos=maps[ticker]
        a_i=pos.get(pd.Timestamp(t.PeakDate)); b_i=pos.get(pd.Timestamp(t.LowerHighDate))
        if a_i is None or b_i is None or b_i+2>=len(d): continue
        p_i=prior_left_shoulder(d,a_i)
        if p_i is None: continue
        A=float(d.at[a_i,'High']); B=float(d.at[b_i,'High']); P=float(d.at[p_i,'High'])
        if not (P < A and B < A): continue
        confirm_i=b_i+1  # pivot B is only known after the following session
        sim=simulate_from_confirm(d,confirm_i,20,1.0)
        if sim is None: continue
        # Neckline proxy = lower of the two valleys P->A and A->B.
        valley1=float(d.loc[p_i+1:a_i-1,'Low'].min()) if a_i-p_i>1 else np.nan
        valley2=float(d.loc[a_i+1:b_i-1,'Low'].min()) if b_i-a_i>1 else np.nan
        neck=max(valley1,valley2) if np.isfinite(valley1) and np.isfinite(valley2) else np.nan
        rec={
            'Ticker':ticker,'PeakDate':t.PeakDate,'PDate':d.at[p_i,'Date'],'BDate':t.LowerHighDate,
            'ConfirmDate':d.at[confirm_i,'Date'],'A':A,'P':P,'B':B,
            'BGapPct':(B/A-1)*100.0,'PGapPct':(P/A-1)*100.0,
            'ShoulderDiffPct':abs(B-P)/A*100.0,
            'HeadProminencePct':(A-max(P,B))/A*100.0,
            'PtoB_pct':(B/P-1)*100.0,
            'Neckline':neck,
            'BAboveNeckPct':(B/neck-1)*100.0 if np.isfinite(neck) and neck>0 else np.nan,
            'Peak_d25':t.d25,'Peak_d50':t.d50,'Peak_d100':t.d100,'Peak_ret20':t.ret20,'Peak_ret60':t.ret60,
            'PeakGain120':t.GainFromLow120_pct,
            **sim,
        }
        rows.append(rec)
    z=pd.DataFrame(rows)
    idx=download_index(); z=z.merge(idx,on='ConfirmDate',how='left')
    z['IDX_WEAK25']=z.IDX_below25.fillna(False)
    z['IDX_WEAK25DOWN']=z.IDX_below25.fillna(False)&z.IDX_25down.fillna(False)
    z.to_csv(OUT/'01_topology_events.csv',index=False)

    # B/A gap buckets: double-top closeness directly.
    gap_abs=-z.BGapPct
    gap_bins=[0,1,3,5,8,12,20,999]
    z['BGapBin']=pd.cut(gap_abs,bins=gap_bins,right=False,include_lowest=True)
    rows=[]
    for period,start,end in PERIODS:
        p=z[(z.PeakDate>=start)&(z.PeakDate<=end)]
        for b,g in p.groupby('BGapBin',observed=True):
            rows.append({'Period':period,'Pattern':'BGAP','Bucket':str(b),'BGapMean':(-g.BGapPct).mean(),'ShoulderDiffMean':g.ShoulderDiffPct.mean(),**stats(g)})
    gapres=pd.DataFrame(rows); gapres.to_csv(OUT/'02_bgap_buckets.csv',index=False)

    # H&S grid: right/left shoulders close, both clearly below head.
    hrows=[]
    for sym in [1,2,3,5]:
        for minhead in [1,3,5,8]:
            hs=(z.ShoulderDiffPct<=sym)&((-z.BGapPct)>=minhead)&((-z.PGapPct)>=minhead)
            for idxrule,imask in [('ANY',pd.Series(True,index=z.index)),('IDX25_WEAK',z.IDX_WEAK25),('IDX25_WEAK_DOWN',z.IDX_WEAK25DOWN)]:
                for period,start,end in PERIODS:
                    g=z[hs & imask & (z.PeakDate>=start)&(z.PeakDate<=end)]
                    hrows.append({'SymPct':sym,'MinHeadPct':minhead,'IndexRule':idxrule,'Period':period,
                                  'BGapMean':(-g.BGapPct).mean() if len(g) else np.nan,'PGapMean':(-g.PGapPct).mean() if len(g) else np.nan,
                                  'ShoulderDiffMean':g.ShoulderDiffPct.mean() if len(g) else np.nan,**stats(g)})
    hsres=pd.DataFrame(hrows); hsres.to_csv(OUT/'03_head_shoulders_grid.csv',index=False)

    # Simple named patterns, fixed before looking at results.
    patterns={
      'DOUBLE_TOP_0_1':(-z.BGapPct<1),
      'DOUBLE_TOP_0_3':(-z.BGapPct<3),
      'DOUBLE_TOP_1_3':((-z.BGapPct>=1)&(-z.BGapPct<3)),
      'DOUBLE_TOP_3_5':((-z.BGapPct>=3)&(-z.BGapPct<5)),
      'HS_TIGHT_2':((z.ShoulderDiffPct<=2)&(-z.BGapPct>=3)&(-z.PGapPct>=3)),
      'HS_MEDIUM_3':((z.ShoulderDiffPct<=3)&(-z.BGapPct>=3)&(-z.PGapPct>=3)),
      'HS_DEEP_3':((z.ShoulderDiffPct<=3)&(-z.BGapPct>=5)&(-z.PGapPct>=5)),
      'WEAK_REBOUND_8_12':((-z.BGapPct>=8)&(-z.BGapPct<12)),
      'WEAK_REBOUND_12P':(-z.BGapPct>=12),
    }
    nrows=[]
    for name,mask in patterns.items():
        for idxrule,imask in [('ANY',pd.Series(True,index=z.index)),('IDX25_WEAK',z.IDX_WEAK25),('IDX25_WEAK_DOWN',z.IDX_WEAK25DOWN)]:
            for period,start,end in PERIODS:
                g=z[mask & imask & (z.PeakDate>=start)&(z.PeakDate<=end)]
                nrows.append({'Pattern':name,'IndexRule':idxrule,'Period':period,
                              'BGapMean':(-g.BGapPct).mean() if len(g) else np.nan,'ShoulderDiffMean':g.ShoulderDiffPct.mean() if len(g) else np.nan,
                              'PeakD50Mean':g.Peak_d50.mean() if len(g) else np.nan,'PeakRet60Mean':g.Peak_ret60.mean() if len(g) else np.nan,
                              **stats(g)})
    named=pd.DataFrame(nrows); named.to_csv(OUT/'04_named_patterns_oos.csv',index=False)

    # All-period shape distribution and relation to subsequent speed.
    allrows=[]
    for name,mask in patterns.items():
        g=z[mask]
        allrows.append({'Pattern':name,'BGapMean':(-g.BGapPct).mean() if len(g) else np.nan,'PGapMean':(-g.PGapPct).mean() if len(g) else np.nan,
                        'ShoulderDiffMean':g.ShoulderDiffPct.mean() if len(g) else np.nan,'PeakD50Mean':g.Peak_d50.mean() if len(g) else np.nan,
                        'PeakRet60Mean':g.Peak_ret60.mean() if len(g) else np.nan,**stats(g)})
    allres=pd.DataFrame(allrows); allres.to_csv(OUT/'05_named_patterns_all.csv',index=False)

    lines=[]
    lines.append('PARADISE LOST — DOUBLE TOP vs HEAD & SHOULDERS TOPOLOGY')
    lines.append('Universe = ex-post True120 cycle tops. A=cycle high, B=first meaningful lower rebound high, P=previous meaningful swing high before A.')
    lines.append('B is recognized only after its next session; simulated short enters the following open. Hold20, 1ATR stop. No lookahead entry.')
    lines.append(f'N_topology={len(z)}')
    lines.append('')
    lines.append('ALL-PERIOD NAMED PATTERNS')
    lines.append(allres.round(3).to_string(index=False))
    lines.append('')
    lines.append('B/A GAP BUCKETS BY PERIOD')
    lines.append(gapres.round(3).to_string(index=False))
    lines.append('')
    lines.append('NAMED PATTERNS OOS')
    lines.append(named.round(3).to_string(index=False))
    lines.append('')
    lines.append('H&S GRID — best cells require caution; review N and OOS stability, not max PF.')
    lines.append(hsres.round(3).to_string(index=False))
    (OUT/'summary.txt').write_text('\n'.join(lines),encoding='utf-8')
    print((OUT/'summary.txt').read_text(encoding='utf-8'))

if __name__=='__main__':
    main()
