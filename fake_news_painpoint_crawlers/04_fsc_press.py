"""Korea Financial Services Commission (FSC) press-release crawler.
Uses public search/list pages; rate-limited and no login/paywall bypass.
"""
from __future__ import annotations
import argparse,csv,re,time
from pathlib import Path
from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup

BASE="https://www.fsc.go.kr"
LIST="https://www.fsc.go.kr/no010101"
QUERIES=["딥페이크","투자사기","유명인 사칭","허위사실","허위정보","가짜뉴스","주가조작","불공정거래","투자자 피해","피싱"]
HEADERS={"User-Agent":"Mozilla/5.0 (compatible; painpoint-research/1.0)"}

def clean_soup(soup):
    for x in soup(["script","style","nav","footer"]):x.decompose()
    return re.sub(r"\s+"," ",soup.get_text(" ",strip=True))

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--out",default="outputs/fsc_press.csv");ap.add_argument("--pages",type=int,default=3);args=ap.parse_args()
    s=requests.Session();s.headers.update(HEADERS);seen=set();rows=[]
    for q in QUERIES:
        for page in range(1,args.pages+1):
            r=s.get(LIST,params={"curPage":page,"srchKey":"sj","srchText":q},timeout=20);r.raise_for_status();soup=BeautifulSoup(r.text,"html.parser")
            links=[]
            for a in soup.select("a[href]"):
                href=a.get("href","")
                if re.search(r"/no010101/\d+",href):links.append((a.get_text(" ",strip=True),urljoin(BASE,href)))
            if not links: break
            for anchor_title,url in links:
                if url in seen:continue
                seen.add(url)
                try:
                    d=s.get(url,timeout=20);d.raise_for_status();ds=BeautifulSoup(d.text,"html.parser");text=clean_soup(ds)
                except Exception as e: print("skip",url,e);continue
                title=(ds.find("h3") or ds.find("h1"));title=title.get_text(" ",strip=True) if title else anchor_title
                hits=[k for k in QUERIES if k in (title+" "+text)]
                if not hits:continue
                pos=min((text.find(k) for k in hits if text.find(k)>=0),default=0)
                rows.append({"channel":"FSC","query":q,"title":title,"url":url,"matched_keywords":"|".join(hits),"relevance_score":len(hits),"snippet":text[max(0,pos-180):pos+700],"body":text[:15000]})
                time.sleep(.35)
    rows.sort(key=lambda x:x["relevance_score"],reverse=True)
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys() if rows else ["channel"]);w.writeheader();w.writerows(rows)
    print(f"saved {len(rows)} rows -> {out}")
if __name__=="__main__":main()
