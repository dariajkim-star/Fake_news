"""FINRA crawler: official news RSS + investor-insights pages.
"""
from __future__ import annotations
import argparse,csv,re,time
from pathlib import Path
from urllib.parse import urljoin
import requests,feedparser
from bs4 import BeautifulSoup

NEWS_RSS="http://feeds.finra.org/FINRANews"
INSIGHTS="https://www.finra.org/investors/insights"
# word-boundary regex patterns: plain substring matching let "AI" hit "said"/"email"
# and "loss" hit "glossary", inflating relevance_score. \w* allows morphological variants.
KEYWORDS=[r"deepfake",r"\bAI\b",r"artificial intelligence",r"fraud\w*",r"scam\w*",r"impersonat\w*",r"fake",r"misleading",r"market manipulation",r"social media",r"investor\w*",r"victim\w*",r"loss\w*"]
HEADERS={"User-Agent":"Mozilla/5.0 (compatible; painpoint-research/1.0)"}
def find_hits(text):
    return [k for k in KEYWORDS if re.search(rf"\b{k}" if not k.startswith(r"\b") else k, text, re.I)]

def textify(html):
    s=BeautifulSoup(html,"html.parser");m=s.find("main") or s
    for x in m(["script","style","nav","footer"]):x.decompose()
    return re.sub(r"\s+"," ",m.get_text(" ",strip=True))

def collect_article(sess,url,title,published=""):
    try:r=sess.get(url,timeout=25);r.raise_for_status();body=textify(r.text)
    except Exception as e:print("skip",url,e);return None
    hits=find_hits(title+" "+body)
    if not hits:return None
    m=[re.search(rf"\b{k}" if not k.startswith(r"\b") else k,body,re.I) for k in hits]
    pos=min((x.start() for x in m if x),default=0)
    return {"channel":"FINRA","published":published,"title":title,"url":url,"matched_keywords":"|".join(hits),"relevance_score":len(hits),"snippet":body[max(0,pos-200):pos+800],"body":body[:18000]}

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--out",default="outputs/finra.csv");args=ap.parse_args();sess=requests.Session();sess.headers.update(HEADERS);rows=[];seen=set()
    # News RSS
    try:
        rr=sess.get(NEWS_RSS,timeout=25);rr.raise_for_status();feed=feedparser.parse(rr.content)
        for e in feed.entries:
            title=e.get("title","");pre=title+" "+e.get("summary","")
            if find_hits(pre):
                u=e.get("link","");seen.add(u);row=collect_article(sess,u,title,e.get("published",""));
                if row:rows.append(row)
    except Exception as e:print("RSS failed",e)
    # Investor insights listing; always seed the FINRA AI/deepfake investor alert page.
    seeds=["https://www.finra.org/investors/insights/artificial-intelligence-and-investment-fraud"]
    try:
        r=sess.get(INSIGHTS,timeout=25);r.raise_for_status();s=BeautifulSoup(r.text,"html.parser")
        seeds += [urljoin(INSIGHTS,a["href"]) for a in s.select('a[href*="/investors/insights/"]')]
    except Exception as e:print("insights listing failed",e)
    for u in dict.fromkeys(seeds):
        if u in seen:continue
        seen.add(u); title=u.rstrip('/').split('/')[-1].replace('-',' ');row=collect_article(sess,u,title)
        if row:rows.append(row)
        time.sleep(.25)
    rows.sort(key=lambda x:x["relevance_score"],reverse=True)
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys() if rows else ["channel"]);w.writeheader();w.writerows(rows)
    print(f"saved {len(rows)} rows -> {out}")
if __name__=="__main__":main()
