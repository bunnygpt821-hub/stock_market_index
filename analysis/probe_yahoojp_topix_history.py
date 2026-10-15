#!/usr/bin/env python3
import requests
import pandas as pd
from io import StringIO

url='https://finance.yahoo.co.jp/quote/998405.T/history?from=20150101&to=20261001&timeFrame=d&page=1'
r=requests.get(url,timeout=30,headers={'User-Agent':'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124 Safari/537.36'})
print('status',r.status_code,'len',len(r.text),'url',r.url)
print('prefix',repr(r.text[:300]))
try:
    tables=pd.read_html(StringIO(r.text))
    print('tables',len(tables))
    for i,t in enumerate(tables[:5]):
        print('\nTABLE',i,'shape',t.shape)
        print(t.head(10).to_string(index=False))
except Exception as e:
    print('READHTML_ERR',repr(e))
