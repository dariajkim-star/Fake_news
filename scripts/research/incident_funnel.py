"""기사 레코드 → 사건 단위 퍼널.

2,995 article records 는 2,995 독립 사건이 아니다. URL/제목 중복과 동일 사건
중복 보도를 걷어내고, 수동 코딩 후보(verified incident candidate)를 추린다.
"""
from __future__ import annotations
import csv, glob, io, os, re, sys
from collections import Counter, defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
OUT = r"C:\Users\user\Desktop\Fake_news\fake_news_painpoint_crawlers\output"

# 사건 후보 판정: 사칭/딥페이크 축(①)이 있고, 동시에 금전 피해 흔적이 있는 것.
IMPERSONATION = ["딥페이크", "딥페이스", "사칭", "AI 합성", "음성 복제", "합성 영상",
                 "deepfake", "deep fake", "impersonat", "voice clon", "face swap",
                 "ai-generated video", "synthetic media"]
HARM = ["피해", "손실", "편취", "사기", "송금", "구속", "기소", "적발", "검거",
        "fraud", "scam", "loss", "victim", "stole", "defraud", "charged", "arrest"]
MONEY = re.compile(r"(\d[\d,\.]*\s*(?:억|조)\s*원|[\$＄]\s?\d[\d,\.]*\s*(?:billion|million|bn|m\b)?)", re.I)

STOP = re.compile(r"[^\w가-힣]+")


def norm_title(t: str) -> str:
    """제목 정규화 — 매체 접미사·대괄호 태그·공백 제거 후 앞 40자."""
    t = re.sub(r"\s*[-–|]\s*[^-–|]{2,30}$", "", t or "")   # " - 매체명" 꼬리
    t = re.sub(r"^\[[^\]]{1,12}\]\s*", "", t)               # "[영상]" 머리
    return STOP.sub("", t.lower())[:40]


def shingles(t: str, k: int = 6) -> set[str]:
    s = STOP.sub("", (t or "").lower())
    return {s[i:i + k] for i in range(max(1, len(s) - k + 1))}


def main():
    recs = []
    for p in sorted(glob.glob(os.path.join(OUT, "*.csv"))):
        ch = os.path.basename(p)[:2]
        for r in csv.DictReader(open(p, encoding="utf-8-sig")):
            recs.append({
                "ch": ch,
                "title": (r.get("title") or "").strip(),
                "url": (r.get("url") or "").strip(),
                "text": " ".join(filter(None, (r.get("title"), r.get("snippet"), r.get("body")))),
            })

    n0 = len(recs)
    print(f"STEP 0  수집 기사 레코드          : {n0:,}")

    # 1) URL 중복 제거
    seen_url, step1 = set(), []
    for r in recs:
        key = r["url"] or r["title"]
        if key in seen_url:
            continue
        seen_url.add(key)
        step1.append(r)
    print(f"STEP 1  URL 중복 제거 후          : {len(step1):,}  (-{n0-len(step1):,})")

    # 2) 정규화 제목 중복 제거 (동일 기사 재배포·매체 접미사 차이)
    seen_t, step2 = set(), []
    for r in step1:
        k = norm_title(r["title"])
        if not k or k in seen_t:
            continue
        seen_t.add(k)
        step2.append(r)
    print(f"STEP 2  제목 정규화 중복 제거 후  : {len(step2):,}  (-{len(step1)-len(step2):,})")

    # 3) pain point ① 관련 필터
    cand = [r for r in step2
            if any(w.lower() in r["text"].lower() for w in IMPERSONATION)
            and any(w.lower() in r["text"].lower() for w in HARM)]
    print(f"STEP 3  사칭① + 피해 흔적 필터   : {len(cand):,}")

    # 4) 사건 clustering — 제목 shingle 자카드 유사도 단일연결
    sh = [shingles(r["title"]) for r in cand]
    parent = list(range(len(cand)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(cand)):
        for j in range(i + 1, len(cand)):
            a, b = sh[i], sh[j]
            if not a or not b:
                continue
            inter = len(a & b)
            if inter / min(len(a), len(b)) >= 0.5:      # 포함 유사도
                pi, pj = find(i), find(j)
                if pi != pj:
                    parent[pj] = pi

    clusters = defaultdict(list)
    for i, r in enumerate(cand):
        clusters[find(i)].append(r)
    print(f"STEP 4  사건 단위 클러스터        : {len(clusters):,}")

    # 5) 금액이 명시된 클러스터 = 수동 코딩 최우선 후보
    ranked = []
    for members in clusters.values():
        amounts = set()
        for m in members:
            amounts.update(x.strip() for x in MONEY.findall(m["text"]))
        ranked.append((len(members), bool(amounts), members, sorted(amounts)[:3]))
    ranked.sort(key=lambda x: (x[1], x[0]), reverse=True)

    with_money = sum(1 for _, hm, _, _ in ranked if hm)
    print(f"STEP 5  그중 금액 명시 클러스터   : {with_money:,}")
    print(f"\n→ 수동 코딩 후보 상위 20 (보도 건수 / 금액 / 대표 제목)\n" + "-" * 78)
    for size, hm, members, amts in ranked[:20]:
        chans = "".join(sorted({m["ch"] for m in members}))
        print(f"[{size:2d}건 {chans:8s}] {members[0]['title'][:62]}")
        if amts:
            print(f"            금액: {', '.join(amts)}")


if __name__ == "__main__":
    main()
