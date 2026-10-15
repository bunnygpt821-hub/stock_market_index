#!/usr/bin/env python3
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

url='https://www.jpx.co.jp/english/markets/indices/topix/'
r=requests.get(url,timeout=30,headers={'User-Agent':'Mozilla/5.0'})
print('status',r.status_code,'len',len(r.text))
r.raise_for_status()
soup=BeautifulSoup(r.text,'html.parser')
for a in soup.find_all('a'):
    text=' '.join(a.stripped_strings)
    href=a.get('href')
    if not href: continue
    key=(text+' '+href).lower()
    if any(k in key for k in ['histor','2025','topix','index-value','index_value','stockprice']):
        print(repr(text), urljoin(url,href))
