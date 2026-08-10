"""GDELT DOC 2.0 crawler for global news discovery.
Useful for high-volume multilingual discovery; stores article metadata returned by GDELT, not paywalled article text.
"""
from __future__ import annotations
import argparse,csv,time
from pathlib import Path
import requests
ENDPOINT="https://api.gdeltproject.org/api/v2/doc/doc"
QUERIES=[
    '(deepfake OR "AI generated") (investment OR investor OR stock) fraud',
    '("fake news" OR misinformation OR "false rumor") (stock OR shares OR investor)',
    '("market manipulation" OR "pump and dump") "social media"',
    '(impersonation OR impersonating) (CEO OR banker OR investor) fraud',
]
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--out",default="outputs/gdelt.csv");ap.add_argument("--maxrecords",type=int,default=250);ap.add_argument("--timespan",default="1y");args=ap.parse_args();rows=[];seen=set()
    for q in QUERIES:
        params={"query":q,"mode":"ArtList","maxrecords":args.maxrecords,"format":"json","timespan":args.timespan,"sort":"HybridRel"}
        r=requests.get(ENDPOINT,params=params,timeout=45,headers={"User-Agent":"painpoint-research/1.0"});r.raise_for_status();data=r.json()
        for a in data.get("articles",[]):
            u=a.get("url","")
            if u in seen:continue
            seen.add(u)
            rows.append({"channel":"GDELT","query":q,"title":a.get("title",""),"url":u,"domain":a.get("domain",""),"language":a.get("language",""),"sourcecountry":a.get("sourcecountry",""),"seendate":a.get("seendate",""),"socialimage":a.get("socialimage","")})
        time.sleep(.5)
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("w",newline="",encoding="utf-8-sig") as f:w=csv.DictWriter(f,fieldnames=rows[0].keys() if rows else ["channel"]);w.writeheader();w.writerows(rows)
    print(f"saved {len(rows)} rows -> {out}")
if __name__=="__main__":main()
