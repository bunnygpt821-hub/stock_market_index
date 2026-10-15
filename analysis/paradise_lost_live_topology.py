#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'runtime/nikkei225_yahoo_clean_2016_present.csv'
OUT=ROOT/'analysis_outputs/paradise_lost_live_topology'
OUT.mkdir(parents=True,exist_ok=True)

PERIODS=[
 ('TRAIN_2016_2020',pd.Timestamp('2016-01-01'),pd.Timestamp('2020-12-31')),
 ('VALID_2021_2023',pd.Timestamp('2021-01-01'),pd.Timestamp('2023-12-31')),
 ('TEST_2024_2026',pd.Timestamp('2024-01-01'),pd.Timestamp('2026-09-18')),
]


def pf(x):
    r=pd.to_numeric(x,errors='coerce').dropna();gp=r[r>0].sum();gl=-r[r<0].sum()
    if gl==0:return float('inf') if gp>0 else np.nan
    return float(gp/gl)

def load_src():
    s=pd.read_csv(SRC);s['Date']=pd.to_datetime(s.Date,errors='coerce')
    for c in ['Open','High','Low','Close','Volume']:s[c]=pd.to_numeric(s[c],errors='coerce')
    return s.dropna(subset=['Ticker','Date','Open','High','Low','Close']).copy()

def prep(g):
    d=g.sort_values('Date').drop_duplicates('Date',keep='last').reset_index(drop=True).copy()
    pc=d.Close.shift(1);tr=pd.concat([(d.High-d.Low),(d.High-pc).abs(),(d.Low-pc).abs()],axis=1).max(axis=1)
    d['ATR20']=tr.rolling(20,min_periods=20).mean()
    for n in [25,50,100]:
        d[f'MA{n}']=d.Close.rolling(n,min_periods=n).mean();d[f'd{n}']=(d.Close/d[f'MA{n}']-1)*100
    d['ret20']=(d.Close/d.Close.shift(20)-1)*100;d['ret60']=(d.Close/d.Close.shift(60)-1)*100
    return d

def local_high(d,i,k=2):
    if i-k<0 or i+k>=len(d):return False
    h=float(d.at[i,'High'])
    return h>=float(d.loc[i-k:i-1,'High'].max()) and h>float(d.loc[i+1:i+k,'High'].max())

def local_low(d,i,k=1):
    if i-k<0 or i+k>=len(d):return False
    lo=float(d.at[i,'Low'])
    return lo<=float(d.loc[i-k:i-1,'Low'].min()) and lo<float(d.loc[i+1:i+k,'Low'].min())

def find_left_shoulder(d,a_i):
    atr=float(d.at[a_i,'ATR20'])
    out=[]
    for i in range(max(5,a_i-60),a_i-2):
        if not local_high(d,i,2):continue
        p=float(d.at[i,'High']);low=float(d.loc[i+1:a_i-1,'Low'].min())
        dip_pct=(low/p-1)*100;dip_atr=(p-low)/atr if np.isfinite(atr) and atr>0 else np.nan
        if dip_pct<=-2 or (np.isfinite(dip_atr) and dip_atr>=1):out.append(i)
    return out[-1] if out else None

def find_pullback_and_b(d,a_i):
    A=float(d.at[a_i,'High']);atr=float(d.at[a_i,'ATR20'])
    if not np.isfinite(atr) or atr<=0:return None
    end1=min(a_i+20,len(d)-3);pull=None
    for j in range(a_i+1,end1+1):
        lo=float(d.loc[a_i+1:j,'Low'].min());dd=(lo/A-1)*100;dda=(A-lo)/atr
        if dd<=-3 or dda>=1.5:
            pull=int(d.loc[a_i+1:j,'Low'].idxmin());break
    if pull is None:return None
    low=float(d.at[pull,'Low']);end2=min(pull+20,len(d)-3)
    for b in range(pull+2,end2+1):
        if not local_high(d,b,1):continue
        B=float(d.at[b,'High']);reb=(B/low-1)*100;reba=(B-low)/atr
        if reb<2 and reba<1:continue
        if B>=A:return None
        return pull,b
    return None

def simulate(d,confirm_i,hold=20,stop_atr=1.0):
    ei=confirm_i+1;xi=ei+hold-1
    if xi>=len(d):return None
    atr=float(d.at[confirm_i,'ATR20']);entry=float(d.at[ei,'Open'])
    if not np.isfinite(atr) or atr<=0:return None
    stop=entry+stop_atr*atr;hit1=hit2=False;d1=d2=np.nan;mfe=mae=0.;stopped=False;r=np.nan
    for k,j in enumerate(range(ei,xi+1),start=1):
        hi=float(d.at[j,'High']);lo=float(d.at[j,'Low']);mfe=max(mfe,(entry-lo)/atr);mae=max(mae,(hi-entry)/atr)
        st=hi>=stop
        if not st:
            if not hit1 and lo<=entry-atr:hit1=True;d1=k
            if not hit2 and lo<=entry-2*atr:hit2=True;d2=k
        if st:stopped=True;r=-stop_atr;break
    if not stopped:r=(entry-float(d.at[xi,'Close']))/atr
    return {'R':r,'stopped':stopped,'hit1':hit1,'hit2':hit2,'DayTo1ATR':d1,'DayTo2ATR':d2,'MFE_R':mfe,'MAE_R':mae}

def download_idx():
    q=yf.download('^N225',start='2015-01-01',end='2026-10-01',progress=False,auto_adjust=False)
    if isinstance(q.columns,pd.MultiIndex):q.columns=[c[0] for c in q.columns]
    q=q.reset_index();q['Date']=pd.to_datetime(q.Date,errors='coerce');q['Close']=pd.to_numeric(q.Close,errors='coerce');q=q.dropna(subset=['Date','Close']).sort_values('Date')
    o=pd.DataFrame({'ConfirmDate':q.Date,'IDXClose':q.Close})
    for n in [5,15,25,50]:
        ma=q.Close.rolling(n,min_periods=n).mean();o[f'IDX_MA{n}']=ma.values;o[f'IDX_d{n}']=((q.Close/ma)-1)*100;o[f'IDX_MA{n}_slope5']=(ma-ma.shift(5)).values
    o['IDX25WEAK']=o.IDXClose<o.IDX_MA25;o['IDX25DOWN']=o.IDX_MA25_slope5<0;o['IDXBEARSTACK']=(o.IDX_MA5<o.IDX_MA15)&(o.IDX_MA15<o.IDX_MA25)&(o.IDXClose<o.IDX_MA5)
    return o

def stats(g):
    if len(g)==0:return {'N':0,'MeanR':np.nan,'PF':np.nan,'WinRate':np.nan,'StopRate':np.nan,'Hit1R':np.nan,'Hit2R':np.nan,'Day1':np.nan,'Day2':np.nan}
    return {'N':len(g),'MeanR':g.R.mean(),'PF':pf(g.R),'WinRate':(g.R>0).mean(),'StopRate':g.stopped.mean(),'Hit1R':g.hit1.mean(),'Hit2R':g.hit2.mean(),'Day1':g.DayTo1ATR.median(),'Day2':g.DayTo2ATR.median()}

def main():
    src=load_src();rows=[]
    for ticker,g in src.groupby('Ticker',sort=False):
        d=prep(g)
        for a in range(120,len(d)-45):
            dt=pd.Timestamp(d.at[a,'Date'])
            if dt<pd.Timestamp('2016-01-01') or dt>pd.Timestamp('2026-09-18'):continue
            if not local_high(d,a,3):continue
            # A must be a genuine recent high visible at the time: within 0.5% of trailing 60d high.
            h60=float(d.loc[max(0,a-59):a,'High'].max())
            A=float(d.at[a,'High'])
            if A<h60*0.995:continue
            pb=find_pullback_and_b(d,a)
            if pb is None:continue
            pull,b=pb;confirm=b+1
            P_i=find_left_shoulder(d,a)
            P=float(d.at[P_i,'High']) if P_i is not None else np.nan
            B=float(d.at[b,'High'])
            sim=simulate(d,confirm,20,1.0)
            if sim is None:continue
            rec={'Ticker':str(ticker),'ADate':dt,'BDate':d.at[b,'Date'],'ConfirmDate':d.at[confirm,'Date'],'A':A,'B':B,'P':P,
                 'BGapPct':(B/A-1)*100,'PGapPct':(P/A-1)*100 if np.isfinite(P) else np.nan,'ShoulderDiffPct':abs(B-P)/A*100 if np.isfinite(P) else np.nan,
                 'PeakD25':d.at[a,'d25'],'PeakD50':d.at[a,'d50'],'PeakD100':d.at[a,'d100'],'PeakRet20':d.at[a,'ret20'],'PeakRet60':d.at[a,'ret60'],**sim}
            rows.append(rec)
    z=pd.DataFrame(rows)
    # de-dup overlapping A candidates per ticker: keep first setup, then require 20 business-session separation approximated by dates >35 calendar days
    z=z.sort_values(['Ticker','ADate','ConfirmDate']).reset_index(drop=True)
    keep=[]
    for ticker,g in z.groupby('Ticker',sort=False):
        last=None
        for idx,r in g.iterrows():
            if last is None or (r.ADate-last).days>35:
                keep.append(idx);last=r.ADate
    z=z.loc[keep].copy()
    z=z.merge(download_idx(),on='ConfirmDate',how='left');z['IDX_WEAK25']=z.IDX25WEAK.fillna(False);z['IDX_WEAK25DOWN']=z.IDX25WEAK.fillna(False)&z.IDX25DOWN.fillna(False)
    z.to_csv(OUT/'01_live_topology_events.csv',index=False)

    gap=-z.BGapPct;z['BGapBin']=pd.cut(gap,[0,1,3,5,8,12,20,999],right=False,include_lowest=True)
    out=[]
    for pname,start,end in PERIODS:
        p=z[(z.ADate>=start)&(z.ADate<=end)]
        for b,g in p.groupby('BGapBin',observed=True):out.append({'Period':pname,'Bucket':str(b),'BGapMean':(-g.BGapPct).mean(),**stats(g)})
    bg=pd.DataFrame(out);bg.to_csv(OUT/'02_bgap_oos.csv',index=False)

    pats={
      'DOUBLE_0_1':gap<1,'DOUBLE_0_3':gap<3,'DOUBLE_1_3':(gap>=1)&(gap<3),'DOUBLE_3_5':(gap>=3)&(gap<5),
      'HS_TIGHT_2':z.ShoulderDiffPct.le(2)&(-z.BGapPct).ge(3)&(-z.PGapPct).ge(3),
      'HS_MEDIUM_3':z.ShoulderDiffPct.le(3)&(-z.BGapPct).ge(3)&(-z.PGapPct).ge(3),
      'HS_DEEP_3':z.ShoulderDiffPct.le(3)&(-z.BGapPct).ge(5)&(-z.PGapPct).ge(5),
      'WEAK_5_8':(gap>=5)&(gap<8),'WEAK_8_12':(gap>=8)&(gap<12),'WEAK_12P':gap>=12,
    }
    rr=[]
    for name,pm in pats.items():
        for idxname,im in [('ANY',pd.Series(True,index=z.index)),('IDX25_WEAK',z.IDX_WEAK25),('IDX25_WEAK_DOWN',z.IDX_WEAK25DOWN)]:
            for pname,start,end in PERIODS:
                g=z[pm&im&(z.ADate>=start)&(z.ADate<=end)]
                rr.append({'Pattern':name,'IndexRule':idxname,'Period':pname,'BGapMean':(-g.BGapPct).mean() if len(g) else np.nan,'PeakD50Mean':g.PeakD50.mean() if len(g) else np.nan,'PeakRet60Mean':g.PeakRet60.mean() if len(g) else np.nan,**stats(g)})
    res=pd.DataFrame(rr);res.to_csv(OUT/'03_named_patterns_oos.csv',index=False)

    lines=['PARADISE LOST — PROSPECTIVE LIVE TOPOLOGY','No True60/True120 future-top filter. A is only a trailing-60d-high pivot known at the time; B is first meaningful lower rebound high; entry next open after B pivot confirmation. Hold20, 1ATR.','N='+str(len(z)),'','B-GAP OOS',bg.round(3).to_string(index=False),'','NAMED PATTERNS OOS',res.round(3).to_string(index=False)]
    (OUT/'summary.txt').write_text('\n'.join(lines),encoding='utf-8');print((OUT/'summary.txt').read_text(encoding='utf-8'))

if __name__=='__main__':main()
