"""§1.1 실측치 집계 — 크롤러 수집분 전체에 통일 태거 적용.

출력: 채널별 건수, pain point 축별 비율, 피해액 추출, Case A~D 분류.
"""
from __future__ import annotations
import csv, glob, io, os, re, sys, json
from collections import Counter, defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
OUT = r"C:\Users\user\Desktop\Fake_news\fake_news_painpoint_crawlers\output"

# ── 통일 태거 ────────────────────────────────────────────────────────────
# 채널마다 painpoint_tags / matched_keywords로 스키마가 달라 원본 태그를 믿지 않고
# title+snippet+body 원문에 동일 규칙을 다시 적용한다.
AXES = {
    # Pain Point ① media authenticity
    "impersonation": [
        "딥페이크", "딥페이스", "사칭", "AI 합성", "음성 복제", "합성 영상", "가짜 영상",
        "deepfake", "deep fake", "impersonat", "synthetic media", "ai-generated video",
        "voice clon", "face swap",
    ],
    "celebrity_ceo": [
        "유명인", "연예인", "셀럽", "회장", "대표", "CEO", "총수", "인플루언서",
        "celebrity", "ceo", "executive", "elon musk", "public figure", "famous",
    ],
    # Pain Point ② claim verification
    "false_info": [
        "가짜뉴스", "허위사실", "허위정보", "거짓 정보", "풍문", "오도", "허위 공시",
        "fake news", "false information", "misinformation", "disinformation",
        "false claim", "misleading",
    ],
    "market_manipulation": [
        "주가조작", "시세조종", "불공정거래", "부정거래", "펌프앤덤프", "작전세력",
        "market manipulation", "pump and dump", "manipulat", "insider",
    ],
    # 피해 행동/결과
    "money_transfer": [
        "송금", "입금", "이체", "자금 이체", "투자금 납입", "계좌 이체", "편취",
        "wire transfer", "sent money", "transferr", "deposit", "remit", "payment to",
    ],
    "investor_harm": [
        "투자자 피해", "투자 손실", "피해액", "손해", "편취", "피해자", "원금 손실",
        "investor loss", "victim", "defraud", "losses", "fraud loss", "stole",
    ],
    "social_media": [
        "SNS", "유튜브", "텔레그램", "리딩방", "오픈채팅", "카카오톡", "인스타",
        "youtube", "telegram", "whatsapp", "social media", "facebook", "instagram",
    ],
}


def tag(text: str) -> set[str]:
    low = text.lower()
    return {k for k, words in AXES.items() if any(w.lower() in low for w in words)}


# ── 피해액 추출 ──────────────────────────────────────────────────────────
KRW = re.compile(r"(\d[\d,\.]*)\s*(조|억|천만|백만)\s*원")
USD = re.compile(r"[\$＄]\s?(\d[\d,\.]*)\s*(billion|million|bn|m\b)?", re.I)
USD_WORD = re.compile(r"(\d[\d,\.]*)\s*(billion|million)\s*(?:dollars|usd)", re.I)
KRW_UNIT = {"조": 1e12, "억": 1e8, "천만": 1e7, "백만": 1e6}
USD_UNIT = {"billion": 1e9, "bn": 1e9, "million": 1e6, "m": 1e6}


def num(s: str) -> float:
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return 0.0


def extract_amounts(text: str):
    krw = [num(v) * KRW_UNIT[u] for v, u in KRW.findall(text)]
    usd = []
    for v, u in USD.findall(text):
        n = num(v)
        if u:
            usd.append(n * USD_UNIT[u.lower().rstrip(".")])
        elif n >= 10000:  # 단위어 없는 $12,345 형태만 채택
            usd.append(n)
    for v, u in USD_WORD.findall(text):
        usd.append(num(v) * USD_UNIT[u.lower()])
    return krw, usd


# ── Case 코딩 (README §1.1 표) ───────────────────────────────────────────
def case_of(t: set[str]) -> str | None:
    if "impersonation" in t and "celebrity_ceo" in t:
        return "A/B"  # CEO·유명인 deepfake 사칭
    if "impersonation" in t:
        return "D"    # AI 영상 기반 사기 (인물 특정 없음)
    if "false_info" in t and "market_manipulation" in t:
        return "C"    # 조작된 기업발표/허위사실 → 주가
    return None


def main():
    per_channel = {}
    all_tags = Counter()
    cases = Counter()
    krw_all, usd_all = [], []
    n_total = 0
    examples = defaultdict(list)

    for path in sorted(glob.glob(os.path.join(OUT, "*.csv"))):
        name = os.path.basename(path)
        rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
        if not rows:
            per_channel[name] = {"n": 0}
            continue
        c = Counter()
        for r in rows:
            text = " ".join(filter(None, (r.get("title"), r.get("snippet"), r.get("body"))))
            t = tag(text)
            all_tags.update(t)
            c.update(t)
            case = case_of(t)
            if case:
                cases[case] += 1
                if len(examples[case]) < 4 and r.get("title"):
                    examples[case].append((name, r["title"][:75]))
            k, u = extract_amounts(text)
            krw_all += k
            usd_all += u
        n_total += len(rows)
        per_channel[name] = {"n": len(rows), **c}

    print("=" * 70)
    print(f"TOTAL COLLECTED: {n_total}")
    print("=" * 70)
    for k, v in per_channel.items():
        print(f"{k:28s} n={v['n']:5d}  " +
              " ".join(f"{a}={v.get(a,0)}" for a in AXES if v.get(a)))

    print("\n" + "=" * 70)
    print("PAIN POINT 축별 (전체 대비 %)")
    print("=" * 70)
    for a in AXES:
        n = all_tags[a]
        print(f"  {a:22s} {n:5d}  {n/n_total*100:5.1f}%")

    print("\n" + "=" * 70)
    print("CASE 코딩")
    print("=" * 70)
    tot_case = sum(cases.values())
    for c, n in cases.most_common():
        print(f"  Case {c:4s} {n:5d}  {n/n_total*100:5.1f}%")
    print(f"  분류됨 합계 {tot_case} ({tot_case/n_total*100:.1f}%)")
    for c, ex in examples.items():
        print(f"\n  [Case {c}]")
        for src, t in ex:
            print(f"    - ({src[:2]}) {t}")

    print("\n" + "=" * 70)
    print("피해액 추출")
    print("=" * 70)
    krw_all.sort(reverse=True); usd_all.sort(reverse=True)
    print(f"  KRW 언급 {len(krw_all)}건, 합계 {sum(krw_all)/1e8:,.0f}억원, "
          f"최대 {krw_all[0]/1e8:,.0f}억원" if krw_all else "  KRW 없음")
    print(f"  USD 언급 {len(usd_all)}건, 합계 ${sum(usd_all)/1e6:,.0f}M, "
          f"최대 ${usd_all[0]/1e6:,.0f}M" if usd_all else "  USD 없음")
    print(f"  KRW 상위 10: {[f'{v/1e8:,.0f}억' for v in krw_all[:10]]}")
    print(f"  USD 상위 10: {[f'${v/1e6:,.0f}M' for v in usd_all[:10]]}")


if __name__ == "__main__":
    main()
