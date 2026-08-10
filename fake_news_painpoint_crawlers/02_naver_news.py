"""NAVER News Search crawler via NAVER API HUB (current API).
Required env: NAVER_CLIENT_ID, NAVER_CLIENT_SECRET
"""
from __future__ import annotations
import argparse, csv, html, os, re, time
from pathlib import Path
import requests

# NAVER API HUB endpoint (2026 migration). If first run returns 401/404, fall back to
# the legacy endpoint: https://openapi.naver.com/v1/search/news.json with headers
# X-Naver-Client-Id / X-Naver-Client-Secret (same credentials, different header names).
ENDPOINT = "https://naverapihub.apigw.ntruss.com/search/v1/news"
QUERIES = [
    "딥페이크 투자사기 피해", "유명인 사칭 투자사기", "가짜뉴스 주가조작",
    "허위사실 유포 주가 투자자 피해", "허위정보 투자 손실", "SNS 주식 리딩방 사기",
    "주가조작 거짓 정보", "투자자 가짜뉴스 손해", "AI 사칭 투자 피해",
]
TAG_PATTERNS = {
    "deepfake_impersonation": ["딥페이크","사칭","AI 합성","음성 복제"],
    "false_information": ["가짜뉴스","허위사실","허위정보","거짓 정보","오도"],
    "market_manipulation": ["주가조작","시세조종","불공정거래","펌프앤덤프"],
    "investor_harm": ["투자자 피해","투자 손실","피해액","손해","편취"],
    "social_media": ["SNS","유튜브","텔레그램","리딩방","오픈채팅"],
}
def clean(s): return re.sub(r"<[^>]+>", " ", html.unescape(s or "")).strip()
def tags(text):
    return [k for k, words in TAG_PATTERNS.items() if any(w.lower() in text.lower() for w in words)]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out", default="outputs/naver_news.csv"); ap.add_argument("--pages", type=int, default=3); args=ap.parse_args()
    cid=os.getenv("NAVER_CLIENT_ID"); secret=os.getenv("NAVER_CLIENT_SECRET")
    if not cid or not secret: raise SystemExit("Set NAVER_CLIENT_ID and NAVER_CLIENT_SECRET (NAVER API HUB credentials).")
    headers={"X-NCP-APIGW-API-KEY-ID":cid, "X-NCP-APIGW-API-KEY":secret, "User-Agent":"painpoint-research/1.0"}
    rows=[]; seen=set()
    for q in QUERIES:
        for p in range(args.pages):
            start=1+p*100
            r=requests.get(ENDPOINT, params={"query":q,"display":100,"start":start,"sort":"date","format":"json"}, headers=headers, timeout=30); r.raise_for_status()
            items=r.json().get("items",[])
            for it in items:
                title=clean(it.get("title")); snippet=clean(it.get("description")); url=it.get("originallink") or it.get("link")
                if url in seen: continue
                seen.add(url); tg=tags(title+" "+snippet)
                rows.append({"channel":"NAVER News","query":q,"published":it.get("pubDate", ""),"title":title,"url":url,"naver_url":it.get("link",""),"snippet":snippet,"painpoint_tags":"|".join(tg),"relevance_score":len(tg)})
            if len(items)<100: break
            time.sleep(.25)
    rows.sort(key=lambda x:x["relevance_score"], reverse=True)
    out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys() if rows else ["channel"]);w.writeheader();w.writerows(rows)
    print(f"saved {len(rows)} rows -> {out}")
if __name__=="__main__":main()
