"""U.S. SEC press-release RSS crawler + article text fetch.
Set SEC_USER_AGENT='Your Name your.email@example.com' to identify your automated request per SEC guidance.
"""
from __future__ import annotations
import argparse,csv,os,re,time
from pathlib import Path
import requests,feedparser
from bs4 import BeautifulSoup

RSS="https://www.sec.gov/news/pressreleases.rss"
# word-boundary regex patterns: plain substring matching let "AI" hit "said"/"email"
# and "loss" hit "glossary", inflating relevance_score. \w* allows morphological variants.
KEYWORDS=[r"deepfake",r"artificial intelligence",r"\bAI\b",r"fake",r"false",r"misleading",r"fraud\w*",r"impersonat\w*",r"social media",r"market manipulation",r"pump.?and.?dump",r"rumor\w*",r"retail investor",r"deceptive"]
def find_hits(text):
    return [k for k in KEYWORDS if re.search(rf"\b{k}" if not k.startswith(r"\b") else k, text, re.I)]

def body_from_html(html):
    s=BeautifulSoup(html,"html.parser"); main=s.find("main") or s
    for x in main(["script","style","nav","footer"]):x.decompose()
    return re.sub(r"\s+"," ",main.get_text(" ",strip=True))

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--out",default="outputs/sec_press.csv");args=ap.parse_args()
    ua=os.getenv("SEC_USER_AGENT")
    if not ua: raise SystemExit("Set SEC_USER_AGENT, e.g. 'Jane Doe jane@example.com'.")
    sess=requests.Session();sess.headers.update({"User-Agent":ua,"Accept-Encoding":"gzip, deflate","Host":"www.sec.gov"})
    r=sess.get(RSS,timeout=30);r.raise_for_status();feed=feedparser.parse(r.content);rows=[]
    for e in feed.entries:
        title=e.get("title","");summary=re.sub(r"<[^>]+>"," ",e.get("summary", ""));pre=title+" "+summary
        if not find_hits(pre):continue
        url=e.get("link","");body=""
        try:
            d=sess.get(url,timeout=30);d.raise_for_status();body=body_from_html(d.text)
        except Exception as ex:print("article fetch failed",url,ex)
        hits=find_hits(pre+" "+body)
        m=[re.search(rf"\b{k}" if not k.startswith(r"\b") else k,body,re.I) for k in hits]
        pos=min((x.start() for x in m if x),default=0)
        rows.append({"channel":"SEC","published":e.get("published",""),"title":title,"url":url,"matched_keywords":"|".join(hits),"relevance_score":len(hits),"snippet":body[max(0,pos-200):pos+800] if body else summary[:1000],"body":body[:18000]})
        time.sleep(.15)
    rows.sort(key=lambda x:x["relevance_score"],reverse=True)
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys() if rows else ["channel"]);w.writeheader();w.writerows(rows)
    print(f"saved {len(rows)} rows -> {out}")
if __name__=="__main__":main()
