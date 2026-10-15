#!/usr/bin/env python3
import requests
import pandas as pd
from io import StringIO

for page in [1,5,10,15,20,25,30,35,40,50]:
    url=f'https://en.kabutan.com/jp/indices/%5ETOPIX/historical_prices/daily?page={page}'
    print('\nPAGE',page,url)
    try:
        r=requests.get(url,timeout=20,headers={'User-Agent':'Mozilla/5.0'})
        print('status',r.status_code,'len',len(r.text))
        tabs=pd.read_html(StringIO(r.text)) if r.ok else []
        print('tables',len(tabs))
        for i,t in enumerate(tabs):
            if any('Date' in str(c) for c in t.columns):
                print('TABLE',i,t.shape)
                print(t.head(2).to_string(index=False))
                print(t.tail(2).to_string(index=False))
    except Exception as e:
        print('ERR',repr(e))
