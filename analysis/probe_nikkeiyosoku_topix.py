#!/usr/bin/env python3
import requests
import pandas as pd
from io import StringIO

base='https://nikkeiyosoku.com/topix/data/'
for page in [1,2,5,10,15,20,30]:
    params={'start_date':'2024-01-01','end_date':'2026-09-18','page':str(page)}
    print('\nPAGE',page)
    try:
        r=requests.get(base,params=params,timeout=30,headers={'User-Agent':'Mozilla/5.0'})
        print('status',r.status_code,'len',len(r.text),'url',r.url)
        tabs=pd.read_html(StringIO(r.text)) if r.ok else []
        print('tables',len(tabs))
        for i,t in enumerate(tabs):
            cols=' '.join(map(str,t.columns))
            if '日付' in cols and '終値' in cols:
                print('TABLE',i,t.shape)
                print(t.head(3).to_string(index=False))
                print(t.tail(3).to_string(index=False))
    except Exception as e:
        print('ERR',repr(e))
