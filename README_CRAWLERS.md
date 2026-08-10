# Financial Fake-News / Investor-Harm Pain-Point Crawlers

목적: `가짜뉴스·딥페이크·사칭·허위정보·시세조종` 때문에 투자자가 실제 피해를 입거나, 사실을 호도한 주체가 이익을 취한 사례를 모아 문제정의 레퍼런스로 사용한다.

## 채널

1. `01_google_news.py` — 글로벌 언론 discovery (Google News RSS)
2. `02_naver_news.py` — 국내 언론 discovery (NAVER API HUB News Search)
3. `03_krx_stockwatch.py` — KRX 불공정거래 사례/투자자 안내
4. `04_fsc_press.py` — 금융위원회 보도자료
5. `05_fss_dart_press.py` — 금융감독원/DART 보도자료
6. `06_sec_press.py` — 미국 SEC 제재/사기/시장조작 사례
7. `07_finra.py` — FINRA 투자자 경보/AI·deepfake 사기 사례
8. `08_cftc.py` — CFTC Enforcement fraud/manipulation 사례
9. `09_fca.py` — 영국 FCA scam/misleading promotion/consumer harm
10. `10_gdelt.py` — 글로벌 뉴스 대량 discovery

## 설치

```bash
pip install -r requirements.txt
```

## 실행 예

```bash
python 01_google_news.py
python 03_krx_stockwatch.py
python 04_fsc_press.py
```

### NAVER API HUB

2026년 현재 Search API는 NAVER API HUB의 `/search/v1/news`를 사용한다.

```bash
# Windows CMD
set NAVER_CLIENT_ID=...
set NAVER_CLIENT_SECRET=...
python 02_naver_news.py
```

### SEC

SEC automated access는 식별 가능한 User-Agent 사용을 권장하므로 연락 가능한 이메일을 넣는다.

```bash
# Windows CMD
set SEC_USER_AGENT=YourName your_email@example.com
python 06_sec_press.py
```

## 출력

각 스크립트는 `outputs/*.csv`를 만든다. 가능한 경우 아래 필드를 포함한다.

- `channel`
- `query`
- `published`
- `title`
- `url`
- `snippet`
- `body`
- `matched_keywords` / `painpoint_tags`
- `relevance_score`

`relevance_score`는 진실성 점수가 아니라 **pain-point 키워드 매칭 개수**다. 실제 문제정의에 사용하기 전에 원문을 확인한다.

## 권장 검색 프레임

수집 후 사례를 아래 5가지로 코딩하면 pain point 정리가 쉽다.

- `deepfake_impersonation`: 유명인/CEO/금융전문가 사칭, AI 영상·음성
- `false_information`: 가짜뉴스, 허위사실, 허위·과장 정보
- `market_manipulation`: 주가조작, 시세조종, pump-and-dump
- `investor_harm`: 피해액, 손실, 편취, 투자자 피해
- `social_media`: SNS, YouTube, Telegram, 리딩방 등 확산 채널

## 수집 원칙

- 공개 RSS/API/공개 웹페이지만 사용한다.
- 로그인·CAPTCHA·paywall을 우회하지 않는다.
- HTML 크롤러는 낮은 호출 빈도로 실행한다.
- 기사 전문 재배포가 아니라 내부 리서치용 메타데이터/근거 추출에 사용한다.
