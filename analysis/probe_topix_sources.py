#!/usr/bin/env python3
import io
import requests
import pandas as pd

urls = [
    'https://stooq.com/q/d/l/?s=%5Etpx&i=d&d1=20150101&d2=20261001',
    'https://stooq.com/q/d/l/?s=%5Etopx&i=d&d1=20150101&d2=20261001',
    'https://stooq.com/q/d/l/?s=topix&i=d&d1=20150101&d2=20261001',
]
for url in urls:
    print('\nURL', url)
    try:
        r=requests.get(url, timeout=30, headers={'User-Agent':'Mozilla/5.0'})
        print('status', r.status_code, 'len', len(r.text), 'prefix', repr(r.text[:120]))
        if r.ok and 'Date' in r.text[:200]:
            df=pd.read_csv(io.StringIO(r.text))
            print('shape', df.shape)
            print(df.head(3).to_string(index=False))
            print(df.tail(3).to_string(index=False))
    except Exception as e:
        print('ERR', repr(e))
