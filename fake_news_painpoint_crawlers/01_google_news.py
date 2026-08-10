"""Google News RSS crawler for financial misinformation / investor harm references.
Public RSS only; does not bypass paywalls or login walls.
"""
from __future__ import annotations
import argparse, csv, re, time
from html import unescape
from pathlib import Path
from urllib.parse import quote_plus
import feedparser

QUERIES = [
    '"deepfake" investment fraud investor loss',
    '"fake news" stock market investor losses',
    '"false rumor" stock manipulation investors',
    '"misleading information" stock price manipulation',
    '"AI-generated" CEO investment scam',
    '"pump and dump" social media retail investors',
    'financial misinformation investors loss',
]
TAG_PATTERNS = {
    "deepfake_impersonation": [r"deepfake", r"impersonat", r"AI-generated", r"clone"],
    "false_information": [r"fake news", r"false rumor", r"misleading", r"misinformation", r"false information"],
    "market_manipulation": [r"market manipulation", r"stock manipulation", r"pump.?and.?dump", r"price manipulation"],
    "investor_harm": [r"investor loss", r"investors lost", r"defraud", r"victim", r"losses?"],
    "social_media": [r"social media", r"telegram", r"youtube", r"whatsapp", r"facebook", r"x.com", r"twitter"],
}

def clean_html(s: str) -> str:
    return re.sub(r"<[^>]+>", " ", unescape(s or "")).strip()

def tags(text: str) -> list[str]:
    t = text.lower()
    return [k for k, pats in TAG_PATTERNS.items() if any(re.search(p, t, re.I) for p in pats)]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/google_news.csv")
    ap.add_argument("--lang", default="en-US")
    ap.add_argument("--country", default="US")
    ap.add_argument("--sleep", type=float, default=0.4)
    args = ap.parse_args()
    rows, seen = [], set()
    for q in QUERIES:
        ceid_lang = args.lang.split("-")[0]
        url = f"https://news.google.com/rss/search?q={quote_plus(q)}&hl={args.lang}&gl={args.country}&ceid={args.country}:{ceid_lang}"
        feed = feedparser.parse(url)
        for e in feed.entries:
            link = e.get("link", "")
            key = (e.get("title", ""), link)
            if key in seen: continue
            seen.add(key)
            summary = clean_html(e.get("summary", ""))
            title = clean_html(e.get("title", ""))
            src = ""
            if isinstance(e.get("source"), dict): src = e.source.get("title", "")
            text = f"{title} {summary}"
            tg = tags(text)
            rows.append({
                "channel":"Google News", "query":q, "publisher":src,
                "published":e.get("published", ""), "title":title,
                "url":link, "snippet":summary, "painpoint_tags":"|".join(tg),
                "relevance_score":len(tg)
            })
        time.sleep(args.sleep)
    rows.sort(key=lambda r: r["relevance_score"], reverse=True)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys() if rows else ["channel"]); w.writeheader(); w.writerows(rows)
    print(f"saved {len(rows)} rows -> {out}")
if __name__ == "__main__": main()
