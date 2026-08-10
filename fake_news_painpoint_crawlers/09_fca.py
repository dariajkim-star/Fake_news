"""UK FCA official news RSS crawler.
Focuses on scams, misleading promotions, impersonation, deepfakes and consumer losses.
"""
from __future__ import annotations
import argparse,csv,re,time
from pathlib import Path
import requests,feedparser
from bs4 import BeautifulSoup
RSS="https://www.fca.org.uk/news/rss.xml"
# word-boundary regex patterns: plain substring matching let "AI" hit "said"/"email"
# and "loss" hit "glossary", inflating relevance_score. \w* allows morphological variants.
KEYWORDS=[r"deepfake",r"\bAI\b",r"artificial intelligence",r"scam\w*",r"fraud\w*",r"impersonat\w*",r"fake",r"misleading",r"financial promotion",r"market abuse",r"social media",r"consumer\w*",r"victim\w*",r"loss\w*"]
HEADERS={"User-Agent":"Mozilla/5.0 (compatible; painpoint-research/1.0)"}
def find_hits(text):
    return [k for k in KEYWORDS if re.search(rf"\b{k}" if not k.startswith(r"\b") else k, text, re.I)]
def textify(h):
    s=BeautifulSoup(h,"html.parser");m=s.find("main") or s
    for x in m(["script","style","nav","footer"]):x.decompose()
    return re.sub(r"\s+"," ",m.get_text(" ",strip=True))
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--out",default="outputs/fca.csv");args=ap.parse_args();sess=requests.Session();sess.headers.update(HEADERS)
    r=sess.get(RSS,timeout=30);r.raise_for_status();feed=feedparser.parse(r.content);rows=[]
    for e in feed.entries:
        title=e.get("title","");summary=re.sub(r"<[^>]+>"," ",e.get("summary",""));pre=title+" "+summary
        if not find_hits(pre):continue
        url=e.get("link","");body=""
        try:d=sess.get(url,timeout=25);d.raise_for_status();body=textify(d.text)
        except Exception as ex:print("skip",url,ex)
        hits=find_hits(pre+" "+body)
        m=[re.search(rf"\b{k}" if not k.startswith(r"\b") else k,body,re.I) for k in hits];pos=min((x.start() for x in m if x),default=0)
        rows.append({"channel":"FCA","published":e.get("published",""),"title":title,"url":url,"matched_keywords":"|".join(hits),"relevance_score":len(hits),"snippet":body[max(0,pos-200):pos+800] if body else summary[:1000],"body":body[:18000]})
        time.sleep(.2)
    rows.sort(key=lambda x:x["relevance_score"],reverse=True)
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("w",newline="",encoding="utf-8-sig") as f:w=csv.DictWriter(f,fieldnames=rows[0].keys() if rows else ["channel"]);w.writeheader();w.writerows(rows)
    print(f"saved {len(rows)} rows -> {out}")
if __name__=="__main__":main()
