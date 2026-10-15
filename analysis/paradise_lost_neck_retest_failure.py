#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'runtime/nikkei225_yahoo_clean_2016_present.csv'
LIVE_PATH = Path(__file__).with_name('paradise_lost_live_topology.py')
OUT = ROOT / 'analysis_outputs/paradise_lost_neck_retest_failure'
OUT.mkdir(parents=True, exist_ok=True)

spec = importlib.util.spec_from_file_location('live', LIVE_PATH)
live = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(live)

PERIODS = live.PERIODS


def pf(x):
    r = pd.to_numeric(x, errors='coerce').dropna()
    gp = r[r > 0].sum(); gl = -r[r < 0].sum()
    if gl == 0:
        return float('inf') if gp > 0 else np.nan
    return float(gp / gl)


def stats(g):
    if len(g) == 0:
        return {'N':0,'MeanR':np.nan,'PF':np.nan,'WinRate':np.nan,'StopRate':np.nan,
                'Hit1R':np.nan,'Hit2R':np.nan,'Day1':np.nan,'Day2':np.nan,
                'MFE':np.nan,'MAE':np.nan}
    return {
        'N':len(g),'MeanR':g.R.mean(),'PF':pf(g.R),'WinRate':(g.R>0).mean(),
        'StopRate':g.stopped.mean(),'Hit1R':g.hit1.mean(),'Hit2R':g.hit2.mean(),
        'Day1':g.DayTo1ATR.median(),'Day2':g.DayTo2ATR.median(),
        'MFE':g.MFE_R.mean(),'MAE':g.MAE_R.mean(),
    }


def download_idx():
    q = yf.download('^N225', start='2015-01-01', end='2026-10-01', progress=False, auto_adjust=False)
    if isinstance(q.columns, pd.MultiIndex):
        q.columns = [c[0] for c in q.columns]
    q = q.reset_index(); q['Date'] = pd.to_datetime(q.Date, errors='coerce'); q['Close'] = pd.to_numeric(q.Close, errors='coerce')
    q = q.dropna(subset=['Date','Close']).sort_values('Date')
    out = pd.DataFrame({'TriggerDate':q.Date,'IDXClose':q.Close})
    for n in [5,15,25,50]:
        ma = q.Close.rolling(n, min_periods=n).mean()
        out[f'IDX_MA{n}'] = ma.values
        out[f'IDX_d{n}'] = ((q.Close/ma)-1.0).values*100.0
        out[f'IDX_MA{n}_slope5'] = (ma-ma.shift(5)).values
    out['IDX25WEAK'] = out.IDXClose < out.IDX_MA25
    out['IDX25DOWN'] = out.IDX_MA25_slope5 < 0
    out['IDXBEARSTACK'] = (out.IDX_MA5 < out.IDX_MA15) & (out.IDX_MA15 < out.IDX_MA25) & (out.IDXClose < out.IDX_MA5)
    return out


def neck_value(neck_type, j, n2, n2_price, n1=None, n1_price=None):
    if neck_type == 'DOUBLE_NECK':
        return float(n2_price)
    if neck_type == 'HS_LINE':
        if n1 is None or n1_price is None or n2 == n1:
            return np.nan
        slope = (float(n2_price)-float(n1_price))/float(n2-n1)
        return float(n2_price) + slope*float(j-n2)
    raise ValueError(neck_type)


def find_break(d, b_confirm, neck_type, n2, n2_price, n1=None, n1_price=None, max_days=20):
    end = min(b_confirm + max_days, len(d)-3)
    for j in range(b_confirm+1, end+1):
        nv = neck_value(neck_type,j,n2,n2_price,n1,n1_price)
        if np.isfinite(nv) and float(d.at[j,'Close']) < nv:
            return j
    return None


def find_retest_trigger(d, break_i, neck_type, n2, n2_price, n1=None, n1_price=None,
                        mode='TOUCH_FAIL_05ATR', max_days=12):
    end = min(break_i + max_days, len(d)-3)
    atr_break = float(d.at[break_i,'ATR20'])
    if not np.isfinite(atr_break) or atr_break <= 0:
        return None

    if mode.startswith('TOUCH_FAIL_'):
        mult = {'TOUCH_FAIL_025ATR':0.25,'TOUCH_FAIL_05ATR':0.5,'TOUCH_FAIL_10ATR':1.0}[mode]
        for j in range(break_i+1,end+1):
            nv = neck_value(neck_type,j,n2,n2_price,n1,n1_price)
            if not np.isfinite(nv): continue
            hi=float(d.at[j,'High']); cl=float(d.at[j,'Close']); op=float(d.at[j,'Open'])
            # Price retests to within tolerance from below (or pokes through), but closes back under neckline.
            touched = hi >= nv - mult*atr_break
            failed = cl < nv
            bearish = cl < op
            if touched and failed and bearish:
                return j
        return None

    if mode == 'RECLAIM_FAIL':
        reclaimed = None
        for j in range(break_i+1,end+1):
            nv = neck_value(neck_type,j,n2,n2_price,n1,n1_price)
            if not np.isfinite(nv): continue
            cl=float(d.at[j,'Close'])
            if reclaimed is None:
                if cl >= nv:
                    reclaimed=j
            else:
                if cl < nv:
                    return j
        return None

    if mode == 'NEAR_CLOSE_FAIL':
        # Close comes within 0.5 ATR below neckline and then turns lower while remaining below.
        near = None
        for j in range(break_i+1,end+1):
            nv = neck_value(neck_type,j,n2,n2_price,n1,n1_price)
            if not np.isfinite(nv): continue
            cl=float(d.at[j,'Close'])
            if near is None:
                if 0 <= nv-cl <= 0.5*atr_break:
                    near=j
            elif j > near and cl < float(d.at[j-1,'Close']) and cl < nv:
                return j
        return None

    raise ValueError(mode)


def simulate(d, signal_i, hold=20, stop_atr=1.0):
    ei=signal_i+1; xi=ei+hold-1
    if xi>=len(d): return None
    atr=float(d.at[signal_i,'ATR20']); entry=float(d.at[ei,'Open'])
    if not np.isfinite(atr) or atr<=0 or not np.isfinite(entry): return None
    stop=entry+stop_atr*atr; hit1=hit2=False; d1=d2=np.nan; mfe=mae=0.; stopped=False; r=np.nan
    for k,j in enumerate(range(ei,xi+1),start=1):
        hi=float(d.at[j,'High']); lo=float(d.at[j,'Low'])
        mfe=max(mfe,(entry-lo)/atr); mae=max(mae,(hi-entry)/atr)
        st=hi>=stop
        if not st:
            if not hit1 and lo<=entry-atr: hit1=True; d1=k
            if not hit2 and lo<=entry-2*atr: hit2=True; d2=k
        if st:
            stopped=True; r=-stop_atr; break
    if not stopped:
        r=(entry-float(d.at[xi,'Close']))/atr
    return {'R':float(r),'stopped':stopped,'hit1':hit1,'hit2':hit2,'DayTo1ATR':d1,'DayTo2ATR':d2,'MFE_R':mfe,'MAE_R':mae}


def build_setups():
    src=live.load_src(); setups=[]; maps={}
    for ticker,g in src.groupby('Ticker',sort=False):
        d=live.prep(g); maps[str(ticker)]=d
        for a in range(120,len(d)-65):
            adate=pd.Timestamp(d.at[a,'Date'])
            if adate<pd.Timestamp('2016-01-01') or adate>pd.Timestamp('2026-09-18'): continue
            if not live.local_high(d,a,3): continue
            h60=float(d.loc[max(0,a-59):a,'High'].max()); A=float(d.at[a,'High'])
            if A < h60*0.995: continue
            pb=live.find_pullback_and_b(d,a)
            if pb is None: continue
            pull,b=pb; b_confirm=b+1
            B=float(d.at[b,'High'])
            # Right valley / double-top neckline is the true low between A and B, all known by B confirmation.
            if b-a <= 1: continue
            n2=int(d.loc[a+1:b-1,'Low'].idxmin()); n2_price=float(d.at[n2,'Low'])
            p_i=live.find_left_shoulder(d,a)
            n1=n1_price=P=np.nan
            if p_i is not None and a-p_i>1:
                P=float(d.at[p_i,'High'])
                n1=int(d.loc[p_i+1:a-1,'Low'].idxmin()); n1_price=float(d.at[n1,'Low'])
            rec={'Ticker':str(ticker),'a':a,'b':b,'b_confirm':b_confirm,'ADate':adate,
                 'BDate':pd.Timestamp(d.at[b,'Date']),'A':A,'B':B,'P':P,
                 'BGapAbs':(A-B)/A*100.0,'PeakD25':d.at[a,'d25'],'PeakD50':d.at[a,'d50'],
                 'PeakD100':d.at[a,'d100'],'PeakRet20':d.at[a,'ret20'],'PeakRet60':d.at[a,'ret60'],
                 'n2':n2,'n2_price':n2_price,'n1':n1,'n1_price':n1_price}
            setups.append(rec)
    s=pd.DataFrame(setups)
    if s.empty: return s,maps
    # Same de-dup convention as prospective live topology.
    s=s.sort_values(['Ticker','ADate','BDate']).reset_index(drop=True); keep=[]
    for ticker,g in s.groupby('Ticker',sort=False):
        last=None
        for idx,r in g.iterrows():
            if last is None or (r.ADate-last).days>35:
                keep.append(idx); last=r.ADate
    return s.loc[keep].copy(),maps


def main():
    setups,maps=build_setups(); rows=[]
    modes=['NECK_BREAK','TOUCH_FAIL_025ATR','TOUCH_FAIL_05ATR','TOUCH_FAIL_10ATR','RECLAIM_FAIL','NEAR_CLOSE_FAIL']
    for _,s in setups.iterrows():
        d=maps[str(s.Ticker)]
        for neck_type in ['DOUBLE_NECK','HS_LINE']:
            if neck_type=='HS_LINE' and not (np.isfinite(s.n1) and np.isfinite(s.n1_price)):
                continue
            br=find_break(d,int(s.b_confirm),neck_type,int(s.n2),float(s.n2_price),
                          int(s.n1) if np.isfinite(s.n1) else None,
                          float(s.n1_price) if np.isfinite(s.n1_price) else None,20)
            if br is None: continue
            for mode in modes:
                sig = br if mode=='NECK_BREAK' else find_retest_trigger(
                    d,br,neck_type,int(s.n2),float(s.n2_price),
                    int(s.n1) if np.isfinite(s.n1) else None,
                    float(s.n1_price) if np.isfinite(s.n1_price) else None,
                    mode,12)
                if sig is None: continue
                for hold in [10,20]:
                    sim=simulate(d,sig,hold,1.0)
                    if sim is None: continue
                    rows.append({
                        'Ticker':s.Ticker,'ADate':s.ADate,'BDate':s.BDate,'TriggerDate':pd.Timestamp(d.at[sig,'Date']),
                        'NeckType':neck_type,'Trigger':mode,'Hold':hold,
                        'BGapAbs':s.BGapAbs,'PeakD50':s.PeakD50,'PeakD100':s.PeakD100,'PeakRet60':s.PeakRet60,
                        'DaysBtoBreak':br-int(s.b),'DaysBreakToTrigger':sig-br,'DaysBtoTrigger':sig-int(s.b),
                        **sim
                    })
    z=pd.DataFrame(rows)
    idx=download_idx(); z=z.merge(idx,on='TriggerDate',how='left')
    z['IDX_WEAK25']=z.IDX25WEAK.fillna(False); z['IDX_WEAK25DOWN']=z.IDX25WEAK.fillna(False)&z.IDX25DOWN.fillna(False)
    z['IDX_BEARSTACK']=z.IDXBEARSTACK.fillna(False)
    z.to_csv(OUT/'01_retest_events.csv',index=False)

    # Conversion counts from unique A-B setups.
    conv=[]; denom={}
    for label,mask in {
        'DOUBLE_0_1':setups.BGapAbs<1,
        'DOUBLE_1_3':(setups.BGapAbs>=1)&(setups.BGapAbs<3),
        'DOUBLE_0_3':setups.BGapAbs<3,
        'DOUBLE_3_5':(setups.BGapAbs>=3)&(setups.BGapAbs<5),
    }.items():
        denom[label]=int(mask.sum())
        for neck in ['DOUBLE_NECK','HS_LINE']:
            for trig in modes:
                q=z[(z.Hold==20)&(z.NeckType==neck)&(z.Trigger==trig)]
                if label=='DOUBLE_0_1': q=q[q.BGapAbs<1]
                elif label=='DOUBLE_1_3': q=q[(q.BGapAbs>=1)&(q.BGapAbs<3)]
                elif label=='DOUBLE_0_3': q=q[q.BGapAbs<3]
                else: q=q[(q.BGapAbs>=3)&(q.BGapAbs<5)]
                n=q[['Ticker','ADate']].drop_duplicates().shape[0]
                conv.append({'Pattern':label,'NeckType':neck,'Trigger':trig,'BaseSetups':denom[label],
                             'Triggered':n,'Conversion':n/denom[label] if denom[label] else np.nan,
                             'MedianBreakToTrigger':q.DaysBreakToTrigger.median() if n else np.nan})
    conv=pd.DataFrame(conv); conv.to_csv(OUT/'02_conversion.csv',index=False)

    pattern_masks={
        'DOUBLE_0_1':lambda x:x.BGapAbs<1,
        'DOUBLE_1_3':lambda x:(x.BGapAbs>=1)&(x.BGapAbs<3),
        'DOUBLE_0_3':lambda x:x.BGapAbs<3,
        'DOUBLE_3_5':lambda x:(x.BGapAbs>=3)&(x.BGapAbs<5),
    }
    rows=[]
    for hold in [10,20]:
        zz=z[z.Hold==hold]
        for pattern,fn in pattern_masks.items():
            pm=fn(zz)
            for neck in ['DOUBLE_NECK','HS_LINE']:
                for trig in modes:
                    base=zz[pm&(zz.NeckType==neck)&(zz.Trigger==trig)]
                    for idxrule,im in [
                        ('ANY',pd.Series(True,index=base.index)),
                        ('IDX25_WEAK',base.IDX_WEAK25),
                        ('IDX25_WEAK_DOWN',base.IDX_WEAK25DOWN),
                        ('IDX_BEARSTACK',base.IDX_BEARSTACK),
                    ]:
                        for pname,start,end in PERIODS:
                            g=base[im&(base.ADate>=start)&(base.ADate<=end)]
                            rows.append({'Hold':hold,'Pattern':pattern,'NeckType':neck,'Trigger':trig,'IndexRule':idxrule,'Period':pname,
                                         'BGapMean':g.BGapAbs.mean() if len(g) else np.nan,
                                         'PeakD50Mean':g.PeakD50.mean() if len(g) else np.nan,
                                         'PeakRet60Mean':g.PeakRet60.mean() if len(g) else np.nan,
                                         'DelayBMean':g.DaysBtoTrigger.mean() if len(g) else np.nan,
                                         **stats(g)})
    res=pd.DataFrame(rows); res.to_csv(OUT/'03_oos_results.csv',index=False)

    # A compact primary table: horizontal double-top neckline, 20d hold.
    primary=res[(res.Hold==20)&(res.NeckType=='DOUBLE_NECK')&
                (res.Pattern.isin(['DOUBLE_0_1','DOUBLE_1_3','DOUBLE_0_3']))&
                (res.IndexRule.isin(['ANY','IDX25_WEAK','IDX25_WEAK_DOWN']))].copy()
    primary.to_csv(OUT/'04_primary.csv',index=False)

    lines=[]
    lines.append('PARADISE LOST — NECKLINE BREAK → RETEST → FAILURE')
    lines.append('Prospective only. No future-top filter. A=trailing-60d-high pivot; B=first meaningful lower rebound high; N=valley between A and B. Entry is next open after trigger close. 1ATR stop.')
    lines.append('Touch-fail means price retests to within 0.25/0.5/1.0 ATR of neckline from below (or pokes through), but closes bearish and below neckline. Reclaim-fail requires a close back above neckline then a later close below it.')
    lines.append(f'Base setups={len(setups)} event rows={len(z)}')
    lines.append('')
    lines.append('CONVERSION')
    lines.append(conv.round(3).to_string(index=False))
    lines.append('')
    lines.append('PRIMARY — DOUBLE NECK, HOLD20')
    lines.append(primary.round(3).to_string(index=False))
    (OUT/'summary.txt').write_text('\n'.join(lines),encoding='utf-8')
    print((OUT/'summary.txt').read_text(encoding='utf-8'))

if __name__=='__main__':
    main()
