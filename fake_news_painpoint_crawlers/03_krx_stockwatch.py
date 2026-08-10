"""KRX Stockwatch / unfair-trading public-page crawler.
Collects public case/guide pages and filters for false-information / manipulation / investor-harm pain points.
"""
from __future__ import annotations
import argparse, csv, re, time
from collections import deque
from pathlib import Path
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup

SEEDS=["https://stockwatch.krx.co.kr/", "https://help.krx.co.kr/contents/SVL/H/02010100/HEL02010100.jsp"]
KEYWORDS=["허위사실","거짓","가짜","루머","풍문","시세조종","주가조작","불공정거래","투자자 피해","사기","인터넷카페","유사투자자문","SNS","리딩방"]
ALLOWED={"stockwatch.krx.co.kr","help.krx.co.kr"}
HEADERS={"User-Agent":"Mozilla/5.0 (compatible; painpoint-research/1.0)"}

def page_text(soup):
    for x in soup(["script","style","nav","footer"]): x.decompose()
    return re.sub(r"\s+"," ",soup.get_text(" ",strip=True))

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--out",default="outputs/krx_stockwatch.csv");ap.add_argument("--max-pages",type=int,default=80);ap.add_argument("--depth",type=int,default=2);args=ap.parse_args()
    q=deque((u,0) for u in SEEDS); seen=set(); rows=[]
    while q and len(seen)<args.max_pages:
        url,depth=q.popleft()
        if url in seen: continue
        seen.add(url)
        try:
            r=requests.get(url,headers=HEADERS,timeout=20); r.raise_for_status(); soup=BeautifulSoup(r.text,"html.parser")
        except Exception as e:
            print("skip",url,e); continue
        text=page_text(soup); hits=[k for k in KEYWORDS if k in text]
        title=(soup.title.get_text(" ",strip=True) if soup.title else "")
        if hits:
            pos=min((text.find(k) for k in hits if text.find(k)>=0), default=0); snippet=text[max(0,pos-180):pos+600]
            rows.append({"channel":"KRX Stockwatch","title":title,"url":url,"matched_keywords":"|".join(hits),"relevance_score":len(hits),"snippet":snippet,"body":text[:12000]})
        if depth<args.depth:
            for a in soup.select("a[href]"):
                nxt=urljoin(url,a["href"]); p=urlparse(nxt)
                if p.netloc in ALLOWED and ("/contents/SVL/" in p.path or p.path in ["/",""]):
                    nxt=nxt.split("#")[0]
                    if nxt not in seen:q.append((nxt,depth+1))
        time.sleep(.5)
    rows.sort(key=lambda x:x["relevance_score"],reverse=True)
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys() if rows else ["channel"]);w.writeheader();w.writerows(rows)
    print(f"saved {len(rows)} relevant pages -> {out}")
if __name__=="__main__":main()
