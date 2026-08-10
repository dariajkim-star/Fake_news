"""FSS/DART public press-release crawler.
Starts from DART press-release list and follows discovered detail/pagination links.
"""
from __future__ import annotations
import argparse,csv,re,time
from collections import deque
from pathlib import Path
from urllib.parse import urljoin,urlparse
import requests
from bs4 import BeautifulSoup

SEED="https://dart.fss.or.kr/info/searchBodo.do"
KEYWORDS=["딥페이크","투자사기","사칭","허위","과장","가짜","주가조작","시세조종","불공정거래","투자자 피해","사기","리딩방","SNS"]
HEADERS={"User-Agent":"Mozilla/5.0 (compatible; painpoint-research/1.0)"}

def textify(soup):
    for x in soup(["script","style","nav","footer"]):x.decompose()
    return re.sub(r"\s+"," ",soup.get_text(" ",strip=True))

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--out",default="outputs/fss_dart_press.csv");ap.add_argument("--max-list-pages",type=int,default=12);args=ap.parse_args()
    sess=requests.Session();sess.headers.update(HEADERS);q=deque([SEED]);seen_list=set();seen_detail=set();rows=[]
    while q and len(seen_list)<args.max_list_pages:
        u=q.popleft()
        if u in seen_list:continue
        seen_list.add(u)
        try:r=sess.get(u,timeout=20);r.raise_for_status();soup=BeautifulSoup(r.text,"html.parser")
        except Exception as e:print("skip list",u,e);continue
        for a in soup.select("a[href]"):
            href=urljoin(u,a["href"])
            if "selectBodoMain.ax?seqno=" in href and href not in seen_detail:
                seen_detail.add(href)
                try:d=sess.get(href,timeout=20);d.raise_for_status();ds=BeautifulSoup(d.text,"html.parser");txt=textify(ds)
                except Exception as e:print("skip detail",href,e);continue
                title=(ds.find("h1") or ds.find("h2") or ds.find("h3"));title=title.get_text(" ",strip=True) if title else a.get_text(" ",strip=True)
                hits=[k for k in KEYWORDS if k in txt]
                if hits:
                    pos=min((txt.find(k) for k in hits if txt.find(k)>=0),default=0)
                    rows.append({"channel":"FSS/DART","title":title,"url":href,"matched_keywords":"|".join(hits),"relevance_score":len(hits),"snippet":txt[max(0,pos-180):pos+700],"body":txt[:15000]})
                time.sleep(.3)
            elif "searchBodo.do" in href and urlparse(href).netloc.endswith("fss.or.kr") and href not in seen_list:
                q.append(href)
    rows.sort(key=lambda x:x["relevance_score"],reverse=True)
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys() if rows else ["channel"]);w.writeheader();w.writerows(rows)
    print(f"saved {len(rows)} rows -> {out}")
if __name__=="__main__":main()
