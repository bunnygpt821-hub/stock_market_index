#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'runtime/nikkei225_yahoo_clean_2016_present.csv'
OUT = ROOT / 'analysis_outputs/paradise_lost_true_tops'
OUT.mkdir(parents=True, exist_ok=True)

START = pd.Timestamp('2016-01-01')
END = pd.Timestamp('2026-09-18')


def load_src():
    s = pd.read_csv(SRC)
    s['Date'] = pd.to_datetime(s['Date'], errors='coerce')
    for c in ['Open','High','Low','Close','Volume']:
        s[c] = pd.to_numeric(s[c], errors='coerce')
    return s.dropna(subset=['Ticker','Date','Open','High','Low','Close']).copy()


def prep(g: pd.DataFrame) -> pd.DataFrame:
    d = g.sort_values('Date').drop_duplicates('Date', keep='last').reset_index(drop=True).copy()
    pc = d['Close'].shift(1)
    tr = pd.concat([(d.High-d.Low),(d.High-pc).abs(),(d.Low-pc).abs()],axis=1).max(axis=1)
    d['ATR20'] = tr.rolling(20,min_periods=20).mean()
    for n in [15,25,50,100]:
        d[f'MA{n}'] = d.Close.rolling(n,min_periods=n).mean()
        d[f'd{n}'] = (d.Close/d[f'MA{n}']-1)*100
    for n in [5,10,20,60]:
        d[f'ret{n}'] = (d.Close/d.Close.shift(n)-1)*100
    for w in [20,60,120]:
        low = d.Low.shift(1).rolling(w,min_periods=max(10,w//2)).min()
        d[f'PrevLow{w}'] = low
        d[f'GainFromLow{w}_pct'] = (d.High/low-1)*100
        d[f'GainFromLow{w}_ATR'] = (d.High-low)/d.ATR20
    return d


def local_peak(d: pd.DataFrame, i: int, k: int=3) -> bool:
    if i-k < 0 or i+k >= len(d): return False
    h = float(d.at[i,'High'])
    return h >= float(d.loc[i-k:i-1,'High'].max()) and h > float(d.loc[i+1:i+k,'High'].max())


def first_pullback_and_lower_high(d: pd.DataFrame, i: int):
    """Find first meaningful pullback and first meaningful rebound high after it.
    Ex-post structural labeling only; not an entry rule.
    Pullback: >=3% OR >=1.5 ATR from peak within 20 sessions.
    Rebound: >=2% OR >=1 ATR from pullback low, with pivot high within next 20 sessions.
    """
    peak = float(d.at[i,'High']); atr = float(d.at[i,'ATR20'])
    if not np.isfinite(atr) or atr <= 0: return None
    end1 = min(i+20, len(d)-2)
    pull_i = None
    running_low = np.inf
    for j in range(i+1,end1+1):
        running_low = min(running_low, float(d.at[j,'Low']))
        dd_pct = (running_low/peak-1)*100
        dd_atr = (peak-running_low)/atr
        if dd_pct <= -3.0 or dd_atr >= 1.5:
            # use actual lowest low up to this first threshold crossing window
            pull_i = int(d.loc[i+1:j,'Low'].idxmin())
            break
    if pull_i is None: return None

    low = float(d.at[pull_i,'Low'])
    end2 = min(pull_i+20, len(d)-2)
    for h in range(pull_i+2,end2+1):
        if h+1 >= len(d): break
        hh = float(d.at[h,'High'])
        pivot = hh >= float(d.at[h-1,'High']) and hh > float(d.at[h+1,'High'])
        if not pivot: continue
        rebound_pct = (hh/low-1)*100
        rebound_atr = (hh-low)/atr
        if rebound_pct < 2.0 and rebound_atr < 1.0: continue
        return pull_i, h
    return None


def true_for_horizon(d: pd.DataFrame, peak_i: int, lh_i: int, horizon: int) -> bool:
    peak = float(d.at[peak_i,'High'])
    end = min(peak_i+horizon, len(d)-1)
    # lower-high must be below original peak
    if float(d.at[lh_i,'High']) >= peak: return False
    # no later high exceeds original peak during horizon
    return float(d.loc[peak_i+1:end,'High'].max()) < peak


def future_drawdowns(d: pd.DataFrame, i: int):
    peak = float(d.at[i,'High'])
    out={}
    for h in [20,60,120]:
        end=min(i+h,len(d)-1)
        if end<=i: continue
        lo=float(d.loc[i+1:end,'Low'].min())
        out[f'FutureDD{h}_pct']=(lo/peak-1)*100
    return out


def collect():
    src=load_src(); rows=[]
    for ticker,g in src.groupby('Ticker',sort=False):
        d=prep(g)
        for i in range(120,len(d)-125):
            dt=pd.Timestamp(d.at[i,'Date'])
            if dt<START or dt>END: continue
            if not local_peak(d,i,3): continue
            # candidate peak should be a recent high, not a random minor swing
            prev60=float(d.loc[max(0,i-59):i,'High'].max())
            if float(d.at[i,'High']) < prev60*0.995: continue
            plh=first_pullback_and_lower_high(d,i)
            if plh is None: continue
            pull_i,lh_i=plh
            if float(d.at[lh_i,'High']) >= float(d.at[i,'High']): continue
            rec={
                'Ticker':str(ticker),'PeakDate':dt,'PeakHigh':float(d.at[i,'High']),
                'PullbackDate':d.at[pull_i,'Date'],'LowerHighDate':d.at[lh_i,'Date'],
                'PullbackPct':(float(d.at[pull_i,'Low'])/float(d.at[i,'High'])-1)*100,
                'LowerHighGapPct':(float(d.at[lh_i,'High'])/float(d.at[i,'High'])-1)*100,
                'DaysToPullback':pull_i-i,'DaysToLowerHigh':lh_i-i,
                'True60':true_for_horizon(d,i,lh_i,60),
                'True120':true_for_horizon(d,i,lh_i,120),
            }
            for c in ['d25','d50','d100','ret5','ret10','ret20','ret60',
                      'GainFromLow20_pct','GainFromLow60_pct','GainFromLow120_pct',
                      'GainFromLow20_ATR','GainFromLow60_ATR','GainFromLow120_ATR']:
                rec[c]=d.at[i,c]
            rec.update(future_drawdowns(d,i))
            rows.append(rec)
    z=pd.DataFrame(rows)
    if z.empty: return z
    # De-duplicate same ticker peaks occurring within 20 sessions: keep higher peak.
    z=z.sort_values(['Ticker','PeakDate','PeakHigh']).reset_index(drop=True)
    keep=[]
    for ticker,g in z.groupby('Ticker',sort=False):
        gg=g.sort_values('PeakDate')
        cluster=[]; last=None
        for idx,row in gg.iterrows():
            if last is None or (row.PeakDate-last).days>35:
                if cluster:
                    keep.append(max(cluster,key=lambda x:z.loc[x,'PeakHigh']))
                cluster=[idx]
            else:
                cluster.append(idx)
            last=row.PeakDate
        if cluster: keep.append(max(cluster,key=lambda x:z.loc[x,'PeakHigh']))
    return z.loc[keep].sort_values('PeakDate').reset_index(drop=True)


def dist_table(z: pd.DataFrame, flag: str):
    x=z[z[flag]].copy()
    feats=['d25','d50','d100','ret5','ret10','ret20','ret60',
           'GainFromLow20_pct','GainFromLow60_pct','GainFromLow120_pct',
           'GainFromLow20_ATR','GainFromLow60_ATR','GainFromLow120_ATR']
    qs=[0,0.05,0.10,0.25,0.50,0.75,0.90,0.95,0.99,0.995,1.0]
    rows=[]
    for f in feats:
        s=pd.to_numeric(x[f],errors='coerce').replace([np.inf,-np.inf],np.nan).dropna()
        if s.empty: continue
        q=s.quantile(qs)
        rec={'Definition':flag,'Feature':f,'N':len(s),'Mean':s.mean(),'Std':s.std()}
        for qq,val in zip(qs,q.values):
            rec[f'P{qq*100:g}']=val
        rows.append(rec)
    return pd.DataFrame(rows)


def bins(z: pd.DataFrame, flag: str):
    x=z[z[flag]].copy(); rows=[]
    specs={
      'd25':[-999,0,3,5,8,12,20,999],
      'd50':[-999,0,5,10,15,20,30,50,999],
      'd100':[-999,0,10,15,20,30,40,60,999],
      'ret20':[-999,0,5,10,20,30,50,80,999],
      'ret60':[-999,0,10,20,30,50,80,120,999],
      'GainFromLow60_pct':[-999,0,10,20,30,50,80,120,200,999],
      'GainFromLow120_pct':[-999,0,20,30,50,80,120,200,300,999],
    }
    for f,edges in specs.items():
        s=pd.to_numeric(x[f],errors='coerce')
        cats=pd.cut(s,bins=edges,right=False)
        for b,g in x.groupby(cats,observed=True):
            rows.append({'Definition':flag,'Feature':f,'Bin':str(b),'N':len(g),
                         'Share':len(g)/len(x) if len(x) else np.nan,
                         'MedianFutureDD60':g.FutureDD60_pct.median(),
                         'MedianFutureDD120':g.FutureDD120_pct.median()})
    return pd.DataFrame(rows)


def top_events(z: pd.DataFrame, flag: str):
    x=z[z[flag]].copy(); feats=['d25','d50','d100','ret20','ret60','GainFromLow60_pct','GainFromLow120_pct','GainFromLow120_ATR']
    outs=[]
    for f in feats:
        t=x.nlargest(15,f).copy(); t.insert(0,'RankFeature',f)
        outs.append(t)
    return pd.concat(outs,ignore_index=True) if outs else pd.DataFrame()


def main():
    z=collect(); z.to_csv(OUT/'01_cycle_top_events.csv',index=False)
    d60=dist_table(z,'True60'); d120=dist_table(z,'True120'); dist=pd.concat([d60,d120],ignore_index=True)
    dist.to_csv(OUT/'02_distribution_percentiles.csv',index=False)
    b=pd.concat([bins(z,'True60'),bins(z,'True120')],ignore_index=True); b.to_csv(OUT/'03_distribution_bins.csv',index=False)
    te=pd.concat([top_events(z,'True60'),top_events(z,'True120')],ignore_index=True); te.to_csv(OUT/'04_extreme_top_events.csv',index=False)

    lines=[]
    lines.append('PARADISE LOST — TRUE CYCLE TOP ALTITUDE DISTRIBUTION')
    lines.append('Cycle top structure: recent 60d high pivot -> meaningful pullback -> meaningful lower high -> original peak not exceeded for 60/120 sessions.')
    lines.append('This is ex-post top labeling for distribution study, NOT a live entry rule. Current-Nikkei225 survivorship bias remains.')
    lines.append(f'Candidates={len(z)} True60={int(z.True60.sum()) if len(z) else 0} True120={int(z.True120.sum()) if len(z) else 0}')
    lines.append('')
    for flag in ['True60','True120']:
        lines.append(flag+' — KEY PERCENTILES')
        show=dist[(dist.Definition==flag)&dist.Feature.isin(['d25','d50','d100','ret20','ret60','GainFromLow60_pct','GainFromLow120_pct','GainFromLow120_ATR'])]
        cols=['Feature','N','Mean','P50','P75','P90','P95','P99','P99.5','P100']
        lines.append(show[cols].round(3).to_string(index=False))
        lines.append('')
        lines.append(flag+' — BINS')
        lines.append(b[b.Definition.eq(flag)].round(3).to_string(index=False))
        lines.append('')
    (OUT/'summary.txt').write_text('\n'.join(lines),encoding='utf-8')
    print((OUT/'summary.txt').read_text(encoding='utf-8'))

if __name__=='__main__': main()
