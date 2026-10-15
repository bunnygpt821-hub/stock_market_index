#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'runtime/nikkei225_yahoo_clean_2016_present.csv'
OUT=ROOT/'analysis_outputs/paradise_lost_neckline'
OUT.mkdir(parents=True,exist_ok=True)

PERIODS=[('TRAIN_2016_2020',pd.Timestamp('2016-01-01'),pd.Timestamp('2020-12-31')),('VALID_2021_2023',pd.Timestamp('2021-01-01'),pd.Timestamp('2023-12-31')),('TEST_2024_2026',pd.Timestamp('2024-01-01'),pd.Timestamp('2026-09-18'))]

def pf(x):
    r=pd.to_numeric(x,errors='coerce').dropna();gp=r[r>0].sum();gl=-r[r<0].sum()
    if gl==0:return float('inf') if gp>0 else np.nan
    return float(gp/gl)
def load_src():
    s=pd.read_csv(SRC);s['Date']=pd.to_datetime(s.Date,errors='coerce')
    for c in ['Open','High','Low','Close','Volume']:s[c]=pd.to_numeric(s[c],errors='coerce')
    return s.dropna(subset=['Ticker','Date','Open','High','Low','Close']).copy()
def prep(g):
    d=g.sort_values('Date').drop_duplicates('Date',keep='last').reset_index(drop=True).copy();pc=d.Close.shift(1)
    tr=pd.concat([(d.High-d.Low),(d.High-pc).abs(),(d.Low-pc).abs()],axis=1).max(axis=1);d['ATR20']=tr.rolling(20,min_periods=20).mean()
    for n in [25,50,100]:d[f'MA{n}']=d.Close.rolling(n,min_periods=n).mean();d[f'd{n}']=(d.Close/d[f'MA{n}']-1)*100
    d['ret20']=(d.Close/d.Close.shift(20)-1)*100;d['ret60']=(d.Close/d.Close.shift(60)-1)*100
    return d
def local_high(d,i,k=2):
    if i-k<0 or i+k>=len(d):return False
    h=float(d.at[i,'High']);return h>=float(d.loc[i-k:i-1,'High'].max()) and h>float(d.loc[i+1:i+k,'High'].max())
def find_left(d,a):
    atr=float(d.at[a,'ATR20']);c=[]
    for i in range(max(5,a-60),a-2):
        if not local_high(d,i,2):continue
        p=float(d.at[i,'High']);lo=float(d.loc[i+1:a-1,'Low'].min());dd=(lo/p-1)*100;dda=(p-lo)/atr if np.isfinite(atr) and atr>0 else np.nan
        if dd<=-2 or (np.isfinite(dda) and dda>=1):c.append(i)
    return c[-1] if c else None
def find_pb_b(d,a):
    A=float(d.at[a,'High']);atr=float(d.at[a,'ATR20'])
    if not np.isfinite(atr) or atr<=0:return None
    pull=None
    for j in range(a+1,min(a+20,len(d)-3)+1):
        lo=float(d.loc[a+1:j,'Low'].min())
        if (lo/A-1)*100<=-3 or (A-lo)/atr>=1.5:pull=int(d.loc[a+1:j,'Low'].idxmin());break
    if pull is None:return None
    low=float(d.at[pull,'Low'])
    for b in range(pull+2,min(pull+20,len(d)-3)+1):
        if not local_high(d,b,1):continue
        B=float(d.at[b,'High'])
        if (B/low-1)*100<2 and (B-low)/atr<1:continue
        if B>=A:return None
        return pull,b
    return None
def sim(d,trigger,hold=20,stop_atr=1.0):
    ei=trigger+1;xi=ei+hold-1
    if xi>=len(d):return None
    atr=float(d.at[trigger,'ATR20']);entry=float(d.at[ei,'Open'])
    if not np.isfinite(atr) or atr<=0:return None
    stop=entry+stop_atr*atr;hit1=hit2=False;d1=d2=np.nan;stopped=False;r=np.nan
    for k,j in enumerate(range(ei,xi+1),start=1):
        hi=float(d.at[j,'High']);lo=float(d.at[j,'Low']);st=hi>=stop
        if not st:
            if not hit1 and lo<=entry-atr:hit1=True;d1=k
            if not hit2 and lo<=entry-2*atr:hit2=True;d2=k
        if st:stopped=True;r=-stop_atr;break
    if not stopped:r=(entry-float(d.at[xi,'Close']))/atr
    return {'R':r,'stopped':stopped,'hit1':hit1,'hit2':hit2,'DayTo1ATR':d1,'DayTo2ATR':d2}
def idx():
    q=yf.download('^N225',start='2015-01-01',end='2026-10-01',progress=False,auto_adjust=False)
    if isinstance(q.columns,pd.MultiIndex):q.columns=[c[0] for c in q.columns]
    q=q.reset_index();q['Date']=pd.to_datetime(q.Date,errors='coerce');q['Close']=pd.to_numeric(q.Close,errors='coerce');q=q.dropna(subset=['Date','Close']).sort_values('Date')
    o=pd.DataFrame({'TriggerDate':q.Date,'IDXClose':q.Close});ma25=q.Close.rolling(25,min_periods=25).mean();o['IDX25']=ma25.values;o['IDX25DOWN']=(ma25-ma25.shift(5)).values<0;o['IDXWEAK']=o.IDXClose<o.IDX25
    return o
def stats(g):
    if len(g)==0:return {'N':0,'MeanR':np.nan,'PF':np.nan,'WinRate':np.nan,'StopRate':np.nan,'Hit1R':np.nan,'Hit2R':np.nan,'Day1':np.nan,'Day2':np.nan}
    return {'N':len(g),'MeanR':g.R.mean(),'PF':pf(g.R),'WinRate':(g.R>0).mean(),'StopRate':g.stopped.mean(),'Hit1R':g.hit1.mean(),'Hit2R':g.hit2.mean(),'Day1':g.DayTo1ATR.median(),'Day2':g.DayTo2ATR.median()}
def main():
    src=load_src();rows=[]
    for ticker,g in src.groupby('Ticker',sort=False):
        d=prep(g)
        for a in range(120,len(d)-70):
            dt=pd.Timestamp(d.at[a,'Date'])
            if dt<pd.Timestamp('2016-01-01') or dt>pd.Timestamp('2026-09-18'):continue
            if not local_high(d,a,3):continue
            A=float(d.at[a,'High']);h60=float(d.loc[max(0,a-59):a,'High'].max())
            if A<h60*0.995:continue
            x=find_pb_b(d,a)
            if x is None:continue
            pull,b=x;B=float(d.at[b,'High']);P_i=find_left(d,a);P=float(d.at[P_i,'High']) if P_i is not None else np.nan
            # double-top neckline = valley between A and B. H&S proxy = higher of the two valleys P-A and A-B.
            valley2=float(d.loc[a+1:b-1,'Low'].min())
            valley1=float(d.loc[P_i+1:a-1,'Low'].min()) if P_i is not None and a-P_i>1 else np.nan
            neck_double=valley2;neck_hs=max(valley1,valley2) if np.isfinite(valley1) else valley2
            for necktype,neck in [('DOUBLE_NECK',neck_double),('HS_NECK',neck_hs)]:
                trig=None
                for j in range(b+1,min(b+25,len(d)-22)+1):
                    if float(d.at[j,'Close'])<neck:
                        trig=j;break
                if trig is None:continue
                s=sim(d,trig,20,1.0)
                if s is None:continue
                rows.append({'Ticker':str(ticker),'ADate':dt,'BDate':d.at[b,'Date'],'TriggerDate':d.at[trig,'Date'],'NeckType':necktype,'A':A,'B':B,'P':P,'BGapPct':(B/A-1)*100,'PGapPct':(P/A-1)*100 if np.isfinite(P) else np.nan,'ShoulderDiffPct':abs(B-P)/A*100 if np.isfinite(P) else np.nan,'PeakD50':d.at[a,'d50'],'PeakRet60':d.at[a,'ret60'],'DelayBtoBreak':trig-b,**s})
    z=pd.DataFrame(rows).sort_values(['Ticker','ADate','TriggerDate']);
    # de-dup same ticker/A cluster by first setup every ~20 sessions
    keep=[]
    for ticker,g in z.groupby('Ticker',sort=False):
        last=None
        for i,r in g.iterrows():
            if last is None or (r.ADate-last).days>35:keep.append(i);last=r.ADate
    z=z.loc[keep].copy();z=z.merge(idx(),on='TriggerDate',how='left');z['IDXWEAKDOWN']=z.IDXWEAK.fillna(False)&z.IDX25DOWN.fillna(False)
    z.to_csv(OUT/'01_neckline_events.csv',index=False)
    gap=-z.BGapPct
    pats={'DOUBLE_0_1':gap<1,'DOUBLE_0_3':gap<3,'DOUBLE_1_3':(gap>=1)&(gap<3),'DOUBLE_3_5':(gap>=3)&(gap<5),'HS_TIGHT_2':z.ShoulderDiffPct.le(2)&(-z.BGapPct).ge(3)&(-z.PGapPct).ge(3),'HS_MEDIUM_3':z.ShoulderDiffPct.le(3)&(-z.BGapPct).ge(3)&(-z.PGapPct).ge(3)}
    out=[]
    for name,pm in pats.items():
        for neck in ['DOUBLE_NECK','HS_NECK']:
            for ir,im in [('ANY',pd.Series(True,index=z.index)),('IDX25_WEAK',z.IDXWEAK.fillna(False)),('IDX25_WEAK_DOWN',z.IDXWEAKDOWN)]:
                for pn,st,en in PERIODS:
                    g=z[pm&(z.NeckType==neck)&im&(z.ADate>=st)&(z.ADate<=en)]
                    out.append({'Pattern':name,'NeckType':neck,'IndexRule':ir,'Period':pn,'BGapMean':(-g.BGapPct).mean() if len(g) else np.nan,'DelayMean':g.DelayBtoBreak.mean() if len(g) else np.nan,'PeakD50Mean':g.PeakD50.mean() if len(g) else np.nan,'PeakRet60Mean':g.PeakRet60.mean() if len(g) else np.nan,**stats(g)})
    res=pd.DataFrame(out);res.to_csv(OUT/'02_neckline_oos.csv',index=False)
    (OUT/'summary.txt').write_text('PARADISE LOST — PROSPECTIVE NECKLINE BREAK\nNo future-top filter. A recent high -> pullback -> B<A -> wait for close below neckline -> next-open short. Hold20, 1ATR.\nN='+str(len(z))+'\n\n'+res.round(3).to_string(index=False),encoding='utf-8');print((OUT/'summary.txt').read_text(encoding='utf-8'))
if __name__=='__main__':main()
