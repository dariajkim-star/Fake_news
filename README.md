# FinDeepfake-48h

**Face-Level Object Detection + Evidence-Based Financial Claim Verification for Deepfake-Related Investment Risk**

48시간 안에 완주하는 것을 설계 제약으로 삼은 멀티모달 딥러닝 프로젝트다.
영상이 **AI로 합성·조작되었는지**와, 영상 속 **금융 주장이 제공된 근거와 일치하는지**를 각각 판정한다.

---

## 1. 문제 정의

### 1.1 출발점 — 기술이 아니라 실제 피해 사례

이 프로젝트는 "딥페이크가 위험할 것 같다"는 직감에서 출발하지 않는다.
**크롤링으로 실제 피해 사례를 먼저 확보하고, 그 피해가 왜 발생했는지를 기술적으로 분해**해서 문제를 정의한다.

수집 채널: 언론 discovery(Google News·NAVER) + 감독기관 사실확인(KRX·금융위·금감원·SEC·FINRA·CFTC·FCA) — [README_CRAWLERS.md](README_CRAWLERS.md) 참조.
이 작업은 48시간 시계 **밖**(D-1 이전)에 완료했다 (2026-08-11 실행).

**수집 실적**: 8개 채널 **2,995건** (국내 2,571 / 해외 424).
미수집 2건 — GDELT(API rate limit), 금감원 DART(목록이 AJAX 렌더링). 두 채널 없이도 표본은 충분하다고 판단해 종결했다.

수집된 사건을 다음 구조로 코딩했다. 각 행의 사례는 수집분에서 실제로 나온 것이다.

| 사건 | 허위정보 형태 | 투자자가 믿은 것 | 행동 | 피해 | 코딩된 실제 사례 |
|---|---|---|---|---|---|
| Case A | CEO·임원 deepfake | 경영진의 실제 지시 | 송금 | 금전손실 | **C15** 홍콩 다국적기업 재무직원, CFO deepfake 화상회의 후 **US$25M** 송금 |
| Case B | 유명인 deepfake | 검증된 투자상품 | 가입/송금 | 투자금 손실 | **C11** Martin Lewis 딥페이크 광고 **£75,000** · **C12** 인도 재무장관 딥페이크 'SpaceX 주식' **₹1.07 crore** |
| Case C | 허위 투자정보·사칭 권유 | 검증된 투자정보·손실복구 기회 | 매수/송금 | 투자금 손실 | **C09** 투자자문업체 사칭 '손실복구·비상장주 저가매수' **18억원** |
| Case D | AI 영상 + 가짜 사이트·앱 | 공식 금융서비스 | 자금이체 | 사기피해 | **C03** FXRP 가짜 스테이킹 사이트(대역배우 동원) **273억원** · **C02** 증권사 사칭 'AI 추천주' 리딩방 **99억원** |

여기서 공통 pain point 두 개가 나온다.

```
Pain Point ①  실제 촬영된 발언 영상인지, AI로 합성·조작된 영상인지 판단하기 어렵다   → Media Authenticity 문제
Pain Point ②  발언 내용이 확인 가능한 근거와 일치하는지 판단하기 어렵다   → Evidence-based Financial Claim Verification 문제
```

### 1.1.1 코딩 결과 — 투자자는 무엇을 확인하지 못했는가

15건의 `verification_failure` 필드를 귀납하면 **4개 유형**으로 수렴한다. 이것이 pain point ①②의 실증 근거다.

| 유형 | 투자자가 답하지 못한 질문 | 해당 | 대응 축 |
|---|---|---:|---|
| **A. 발언 영상 진위 불확인** | "이 영상은 실제 촬영본인가, AI로 합성·조작된 것인가?" | **13/15** | ① |
| **B. 채널 정통성 불확인** | "이 계정·앱·링크가 공식인가?" | 6/15 | ①의 주변 |
| **C. 주장 근거 일치 여부 불확인** | "이 금융 주장이 확인 가능한 근거와 일치하는가?" | **12/15** | ② |
| **D. 위조된 사후 증거** | "내가 본 수익 화면은 진짜인가?" | 5/15 | ①②의 결합 |

**A**의 대표 진술: C04 피해자는 영상이 실제 촬영된 콘텐츠처럼 보여 의심하지 못했고,
**사후에야** 딥페이크임을 알았다. C15(홍콩 CFO 화상회의, US$25M)는 실시간 화상회의조차
합성·조작 미디어의 신뢰성을 높이는 장치로 악용될 수 있음을 보여주는 **인접 금융 딥페이크 사례**다.

**C**의 주장들은 상당수가 **확인 가능한 외부 근거와 대조했을 때 지지되지 않거나 반박되는 형태**였다 — "600% 수익 보장", "매일 이자 지급", 비상장 주식 저가 매수, 유명인의 투자 사실 등이다. 핵심 문제는 투자 판단 시점에 이를 대조할 검증 경로가 부족했다는 점이다.

**D가 이 프로젝트가 두 축을 모두 만드는 이유다.** 투자자가 스스로 검증하려 할 때 마주치는 증거(가짜 HTS 수익 화면,
위조 인증, 초기 며칠의 실제 이자 지급)마저 조작돼 있어 **자가검증 루프가 닫힌다.**
매체 진위만 봐도 뚫리고, 주장 검증만 해도 뚫린다.

**축 분포가 이를 뒷받침한다.**

```
① media authenticity 만 :  3건   (C07 사칭 이메일, C11 Martin Lewis, C15 CFO 화상회의)
② claim verification 만 :  1건   (C09 투자자문업체 사칭 손실복구)
①② 둘 다               : 11건   ← 73%
```

**15건 중 11건에서 두 종류의 검증 실패가 함께 관찰됐다.** 따라서 한 가지 검증 축만으로는
사례에서 드러난 두 위험을 모두 다루기 어렵다고 보고, §1.3에서 두 판정을 분리해 **둘 다** 제시한다.

전체 코딩 표는 [`data/research/coded_cases.csv`](data/research/coded_cases.csv) 참조 (15행 × 12필드).

**이 코딩의 한계 — 발표에서 먼저 밝힌다.**

- **국내 편중**: 15건 중 10건이 국내(네이버) 출처. 해외 5건은 전량 Google News 의존이며, 스니펫만 있어
  피해자 행동·검증 실패 지점을 제목에 명시된 범위로만 코딩했다.
- **해외 규제기관 채널에서 딥페이크 집행 사례 0건**: SEC·FINRA·CFTC·FCA 31건을 훑었으나 전부 일반 사기·정책·인사
  발표였다. 이건 현상이 없다는 뜻이 아니라 **보도자료 채널의 특성**이며, 감독기관 1차 출처만으로는 이 문제를
  포착할 수 없다는 것 자체가 관찰 결과다.
- **피해액은 합산·순위 비교하지 않는다**: 통화가 다르고 집계 기준(개인 피해 / 조직 총 편취액 / 부당이득)이
  뒤섞여 있다. C05는 개별 사건이 아니라 감독당국 집계 평균값이라 성격이 다르다.
- **잔여 병합 위험**: 사건 병합은 제목·스니펫의 금액·주체·수사기관 일치로 판정했다. 금액이 다르게 보도된
  동일 사건이 남아 있을 수 있다(C03이 실제로 "123억"과 "273억"으로 따로 보도된 사례였다).

**문제정의 문장:**

> 딥페이크를 활용한 투자사기와 허위 금융정보 피해 사례를 분석한 결과, 피해자는 투자 판단에 앞서
> **온라인 영상이 실제 촬영된 발언인지 AI로 합성·조작된 것인지**, 그리고 **영상 속 금융 주장이
> 확인 가능한 근거와 일치하는지**를 검증하는 데 어려움을 겪었다. 본 프로젝트는 심층 코딩한 대표 피해사례에서
> 반복된 검증 실패 중 기술적으로 다룰 수 있는 두 문제를 ① 영상의 조작 여부 판별과 ② 근거 기반 금융 주장 검증으로
> 정의한다. 이를 위해 pretrained Object Detection으로 발언자의 얼굴 ROI를 추출한 Deepfake Detection과
> NLP 기반 Evidence-based Financial Claim Verification을 구현하고, 두 판단을 독립적으로 제시하는
> 투자정보 검증 PoC를 구축한다.

즉 문제는 "deepfake 영상을 탐지한다"가 아니라
**"투자자의 판단 이전에 영상의 조작 여부와 금융 주장의 근거 일치 여부를 함께 확인할 수단이 부족하다"**이다.

**다루는 것과 다루지 않는 것을 여기서 분명히 한다.** §1.1.1의 4개 유형 중 우리 시스템이
답하는 것은 A와 C뿐이다. 사례에서 나온 문제를 전부 푸는 것처럼 말하지 않는다.

| 유형 | 해당 | 본 PoC | 근거 |
|---|---:|---|---|
| **A. 발언 영상 진위 불확인** | 13/15 | ✅ **축 ①** | Deepfake Detection이 영상의 합성·조작 가능성을 직접 평가한다 |
| **C. 주장 근거 일치 여부 불확인** | 12/15 | ✅ **축 ②** | Evidence-based Claim Verification이 claim과 근거의 관계를 평가한다 |
| **B. 채널 정통성 불확인** | 6/15 | ❌ **범위 밖** | "이 계정·앱·링크가 공식인가"는 도메인 평판·인증서·플랫폼 메타데이터의 문제이며, 영상·텍스트 콘텐츠 분석으로 답할 수 없다 (C02 유튜브 댓글의 상담 링크, C05 로고 도용 가짜 채널, C07 사칭 이메일) |
| **D. 위조된 사후 증거** | 5/15 | △ **간접** | 직접 탐지하지 않는다. 다만 D는 **①②를 함께 제시해야 하는 이유**다 — 자가검증 증거가 조작돼 있으면 한 축만으로는 뚫린다 |

**B를 범위 밖으로 두는 것은 축소가 아니라 정직한 경계 설정이다.** 15건 중 6건에서 관찰됐으므로
실재하는 문제이고, §9(주장하지 않는 것)와 향후 확장 방향에 함께 남긴다.

또한 코딩된 15건 중 **13건이 영상 기반**이고 2건(C07 사칭 이메일, C09 오픈채팅 사칭)은 영상이 아니다.
"온라인 영상"이라는 범위 한정은 사례의 87%를 덮지만 전부는 아니며, 이 역시 명시한다. C15는 개인 투자자 피해가 아니라 기업 내부 송금 사기이므로, 투자자 pain point의 핵심 표본이라기보다 **인접 금융 딥페이크 사례**로 구분해 해석한다.

### 1.1.2 기사에서 사건으로 — 무엇을 세었는가

**가장 중요한 구분: 수집한 2,995건은 2,995개의 독립 사건이 아니다.** 동일 사건을 여러 매체가 중복 보도한다.
실제로 "캄보디아 투자사기 부부" 한 사건이 제목만 달리한 채 20건 이상으로 흩어져 있었다.
따라서 **기사 수를 분모로 한 백분율을 사건 비율처럼 제시하지 않는다.**

```
2,995  기사 레코드
   ↓   제목 정규화 중복 제거
2,806  고유 기사
   ↓   사칭·조작매체 × 실피해·사법처리 필터
  778  관련 기사
   ↓   제목 유사도 기반 동일 사건 병합
  189  사건 후보 클러스터
   ↓   예방캠페인·정책·홍보 제외 + 동일사건 수동 병합
   15  대표 피해사례  ← 심층 코딩 완료 (D-1 산출물)
```

병합 규모가 이 구분의 필요성을 보여준다 — **캄보디아 부부 로맨스스캠(101억원) 한 사건이 구속기소·재판·선고
보도로 35건 이상 흩어져 있었다.** 이걸 세면 사건 1건이 기사 35건이 된다.

발표에서 쓸 표현은 이것이다:

> 국내외 금융 딥페이크·투자사기 기사 **2,995건**을 수집해 관련 사건을 선별했고,
> **대표 피해사례 15건**을 심층 코딩해 pain point를 도출했다.

심층 코딩 결과는 [`data/research/coded_cases.csv`](data/research/coded_cases.csv)에 있고, 각 사건마다
사칭 대상 · 조작 매체 · 허위 claim · 피해자 행동 · 피해 방식 · 피해액 · **기존 검증 실패지점**을 기록한다.
이 중 마지막 항목이 pain point ①②를 직접 뒷받침한다.

> ℹ️ **왜 15건인가.** 이 프로젝트는 시장 규모를 추정하는 것이 아니라 pain point를 발견하는 것이 목적이다.
> 사건 유형이 반복되기 시작하면 표본을 늘려도 새로운 pain point가 나오지 않는다. 15건이면 목적에 충분하다.

> ⚠️ **피해액 총합은 산출하지 않는다.** 본문에서 금액을 자동 추출하면 연기금 규모(1,400조원)·시장안정
> 프로그램(100조원) 같은 무관한 숫자가 섞여 들어온다. 실제로 시도했고 폐기했다. 피해액은 **개별 사건 단위로만** 인용한다.

확인된 개별 피해액 예: **$25M**(CFO deepfake 화상회의, CNN) · **$1.7M**(Musk deepfake 크립토) ·
**$950M**(deepfake pump-and-dump 주주소송) · **100억원**(딥페이크 투자사기단) · **99억원**(증권사 사칭 리딩방).

**전체 논리 연결 (발표 한 장):**

```
STEP 0  Problem Discovery
        국내외 금융 딥페이크·투자사기 기사 크롤링 (2,995건)
          → 중복 제거 / 동일 사건 병합
          → 대표 피해사례 15건 심층 코딩
          → Pain Point Map
            ↓
STEP 1  Problem Definition
        ① 투자자는 영상이 실제 촬영본인지 AI로 합성·조작된 것인지 판단하기 어렵다
        ② 영상 속 금융 주장이 확인 가능한 근거와 일치하는지 판단하기 어렵다
            ↓
STEP 2  Technical Problems
        ① Media Authenticity Detection
        ② Evidence-based Financial Claim Verification
            ↓
STEP 3  Models
        ① YOLO Face Detection → EfficientNet-B0
        ② Whisper → Claim → DeBERTa (Claim + Evidence)
            ↓
STEP 4  Outputs
        Media authenticity  FAKE 92%
        Financial claim     REFUTED 87%      ← 두 결과를 독립적으로 제시
            ↓
STEP 5  Validation
        H1  V0 (full frame)  vs  V1 (face ROI)
        H2  N1 (claim only)  vs  N2 (claim + evidence)
```

이 사슬의 목적은 **모든 작업에 "왜 하는가"를 부여하는 것**이다. 크롤링은 시장조사 부록이 아니라
모델 요구사항을 만든 근거이고, ablation은 성능 자랑이 아니라 두 pain point에 대한 답이다.

### 1.2 용어 — 우리가 잡으려는 것

용어를 정확히 못 박는다. 이 구분이 프로젝트 범위 전체를 결정한다.

| 개념 | 의미 | AI 조작 필수 |
|---|---|:--:|
| Fake news | 거짓·오도 내용을 실제 뉴스처럼 제시 | ❌ |
| Misinformation | 잘못된 정보 (고의 아닐 수 있음) | ❌ |
| Disinformation | 속일 의도로 만든 거짓 정보 | ❌ |
| Deepfake | AI로 생성·조작해 실제처럼 보이는 미디어 | ✅ |
| **Deepfake-enabled disinformation** | **deepfake로 허위 사실을 믿게 만드는 것** | ✅ |

이 프로젝트의 대상은 **맨 아래 줄**이다.

> 단, **disinformation은 속이려는 의도(intent)를 포함하는 개념**이다. 본 모델은 의도 자체를 추론하지 않으므로 최종 출력에서는 `disinformation`을 직접 판정하지 않는다. 모델은 `media authenticity`와 `claim verification`을 각각 출력하고, 두 신호의 조합을 위험 신호로 해석한다.

구체적 시나리오 — SNS에 이런 영상이 돈다:

> **Jamie Dimon (JPMorgan CEO)**: "저희는 이번 분기 100억 달러의 예상치 못한 손실을 기록했습니다."

영상은 AI 합성이고, 발언 내용도 사실이 아니다. 주가는 이미 움직인다.
이건 가상의 위협이 아니라 FINRA와 SEC가 투자자·기업에 실제로 경고한 사기 유형이다.

### 1.3 왜 두 판정을 분리하는가

시스템은 두 질문에 **따로** 답한다.

```
Q1. 이 영상은 AI로 조작되었는가?          → Media Authenticity
Q2. 영상 속 금융 주장은 제공된 근거에 의해 지지되는가? → Evidence-based Claim Verification
```

합치지 않는 이유는 명확하다. **합성·조작된 영상의 주장도 근거에 의해 지지될 수 있고, 실제 영상의 주장도 근거와 불일치할 수 있다.**
두 신호를 하나의 "Fake News Probability"로 뭉개면 정보가 사라지고, 무엇보다 그렇게 학습시킬 근거가 없다
(→ §8.3). 대신 두 축을 그대로 보여주고 조합만 해석한다.

### 1.4 가설과 연구 질문

두 기술 문제 각각에 검증 가능한 가설을 하나씩 건다.

| # | 가설/질문 | 평가 방식 |
|---|---|---|
| **H1** | 투자 영상에서 얼굴 영역을 Object Detection으로 추출하면 full-frame 대비 deepfake 탐지 성능이 개선된다 | 정량 (V0 vs V1, paired ΔAUROC) |
| **H3** | deepfake 도메인 데이터(OpenForensics)로 face detector를 fine-tuning하면 pretrained 대비 검출 성능이 개선된다 | 정량 (D1 vs D2, ΔmAP@50) |
| **H2** | claim만 사용하는 것보다 evidence를 함께 사용하면 금융 claim 검증 성능이 개선된다 | 정량 (N1 vs N2, Macro-F1) |
| ~~H1-b~~ | ~~ROI margin — 배경 문맥은 도움인가 방해인가?~~ | **철회** — GPU 슬롯을 H3(D2 학습)에 배정 (§6.4) |
| **RQ3** | 조작 여부와 주장 근거 검증 결과를 분리 제시하는 것이 단일 라벨보다 설명 가능한 위험 신호를 주는가? | 정성 (사례 분석) |

**H1은 정직하게 말해 "확인"에 가깝다.** 얼굴 crop이 유리하다는 건 forensics에서 널리 쓰이는 전제다.
그래서 H1-b를 붙였다. crop margin의 최적점은 실제로 알려져 있지 않고, 학습 한 번이면 답이 나온다.

**H2가 NLP 축의 메인 검증이다.** 부수 질문 — claim만 보고도 맞힌다면 그건 무엇을 학습한 것인가 — 를 포함해 자세한 설계와 전제조건은 §7.2.

---

## 2. 시스템 구조

```
                         INPUT VIDEO
                              │
              ┌───────────────┴───────────────┐
              │                               │
           VIDEO                            AUDIO
              │                               │
      Frame Sampling (2 fps)          FFmpeg 16kHz mono
              │                               │
      ┌───────▼────────┐                Whisper (STT)
      │ Face Detection │                      │
      │  YOLOv8n-Face  │  ← 직접 학습    Transcript
      │ (OpenForensics │     (D2)             │
      │   fine-tuned)  │              금융 문장 필터
      └───────┬────────┘                      │
              │ bbox                   Financial Claim
         Face Crop 224²                       │
              │                               │
      ┌───────▼────────┐   Provided Evidence ─┤  ← 자동 생성 아님 (§2.1)
      │   Deepfake     │                      │
      │ EfficientNet-B0│              ┌───────▼────────┐
      │  ← 직접 학습    │              │ Claim Verifier │
      └───────┬────────┘              │ DeBERTa-v3-small│ ← 직접 학습
              │                       └───────┬────────┘
     frame prob → median                      │
              │                       SUPPORTED / REFUTED
      Video-level P(fake)                     │
              └───────────────┬───────────────┘
                              │
                      RISK MATRIX (§8.3)
```

**Evidence는 영상에서 자동으로 생기지 않는다.** Whisper가 만드는 것은 transcript → claim까지다.
evidence는 **별도로 제공되어야 하며**(PoC에서는 Fin-Fact가 제공, 실서비스에서는 retrieval 단계가 필요),
이 경계가 §2.1의 "Evidence Retrieval System이 아니다"의 도식적 표현이다.

**딥러닝 사용 지점 4곳**: Object Detection(YOLO) · 이미지 분류(EfficientNet) · 음성인식(Whisper) · **NLP 텍스트 분류(DeBERTa)**.
이 중 NLP는 선택이 아니라 필수 축이며, 실제로 fine-tuning하고 별도 ablation으로 평가한다.

각 모듈의 역할을 정확히 정의한다. 과장하지 않는 것이 이 프로젝트의 원칙이다.

- **Object Detection**: 전체 화면이 아니라 **발언자의 얼굴 영역을 식별해 forensic analysis 대상으로 제한**한다.
  **OpenForensics의 GT bbox로 직접 fine-tuning하며(§5.3), mAP@50으로 정식 평가한다(D1 vs D2).**
  클래스는 FACE 하나다 — YOLO는 "어디에 얼굴이 있는가", EfficientNet은 "그 얼굴이 진짜인가"로
  역할을 나눠 의존 관계를 만든다. ROI selection이 downstream 분류에 미치는 효과는 V0 vs V1로 따로 검증한다.
- **NLP**: "가짜뉴스를 맞히는" 모델이 아니다. **금융 claim과 evidence 간의 entailment/contradiction을 판별**한다.

### 2.1 Evidence는 어디서 오는가 — 이 PoC의 정확한 범위

파이프라인을 `Video → Whisper → Claim → DeBERTa → 판정`으로만 쓰면 **evidence의 출처가 빠진다.**
이틀 안에 SEC·DART·Reuters 실시간 검색을 만들 수 없으므로, 범위를 다음과 같이 못 박는다.

```
[이번 PoC]                          [실서비스 확장]
Video                               Claim
  ↓ Whisper                           ↓ Retrieval
Claim                               SEC / DART / IR / Trusted News
  ↓                                   ↓
Claim + 제공된 Evidence      →      Evidence
  ↓ NLP Verifier                      ↓ NLI Verification
판정                                판정
```

즉 이 프로젝트는 **Evidence Retrieval System이 아니라 Evidence-based Claim Verification PoC**다.
evidence는 Fin-Fact 데이터셋이 제공하는 것을 사용하며, 임의의 새로운 발언에 대해 근거를 스스로 찾아오지 않는다.
이 구분을 README·발표·산출물 JSON(`claim.caveat`)에서 일관되게 유지한다.

---

## 3. 48시간 범위

### 구현한다

- 영상 frame sampling → 얼굴 검출 → crop
- Deepfake 이진 분류 (full-frame baseline vs face ROI)
- Whisper STT (pretrained inference only)
- 금융 claim 분류 (DistilBERT baseline vs DeBERTa, claim-only vs claim+evidence)
- 통합 추론 파이프라인 + Streamlit 데모
- 신뢰구간을 포함한 평가표

### 하지 않는다 — 그리고 그 이유

| 제외 항목 | 이유 |
|---|---|
| End-to-end multimodal fusion | joint label이 붙은 데이터셋이 존재하지 않음 (§8.3) |
| Audio deepfake detection | 별도 데이터셋(ASVspoof/FakeAVCeleb) 확보 시간 없음 |
| Lip-sync 정합성 분석 | 구현 난이도 대비 이틀 예산 초과 |
| 실시간 evidence retrieval | 외부 API·인덱싱 필요, PoC 범위 밖 |
| CEO 신원 인식 (face recognition) | 인물 DB 구축 필요 |
| Object detector 신규 학습 | GT bbox 라벨링 불가 (§6.3) |
| DFDC 전체(10만+) 학습 | 다운로드만 며칠 |

---

## 4. 산출물

### 4.1 모델

```
models/
├── face_detector/            # YOLOv8n-Face (pretrained, 학습 안 함)
├── deepfake_fullframe.pt     # baseline
├── deepfake_roi.pt           # proposed
└── claim_classifier/         # DeBERTa-v3-small fine-tuned
```

### 4.2 추론 출력

```json
{
  "video": "sample_001.mp4",
  "media": {
    "face_detected": true,
    "frames_used": 18,
    "deepfake_probability": 0.91,
    "aggregation": "median"
  },
  "speech": {
    "transcript": "NVIDIA reported a 40 percent decline in quarterly revenue.",
    "financial_sentences": 1
  },
  "claim": {
    "text": "NVIDIA reported a 40 percent decline in quarterly revenue.",
    "label": "REFUTED",
    "confidence": 0.84,
    "caveat": "evidence retrieval 없음 — §9 참조"
  },
  "risk": {
    "media_authenticity": "SYNTHETIC",
    "claim_verification": "REFUTED",
    "combined": "HIGH RISK — synthetic media + refuted claim"
  }
}
```

**투자자 관점 표시 예** — 산출물은 "이 영상은 가짜다" 한 줄이 아니라, pain point ①②에 대응하는 두 개의 evidence다:

```
영상 분석 결과
─────────────────────────────
발언자 영상          Deepfake probability  92%   ⚠ AI 조작 가능성 높음

추출된 주장          "NVIDIA reported a 40% decline in quarterly revenue."
근거 검증            REFUTED (confidence 87%)
─────────────────────────────
Investment Information Risk
  Media Authenticity : HIGH RISK
  Claim Verification : REFUTED
```

### 4.3 결과 파일

```
results/
├── vision_metrics.json        # video-level, bootstrap CI 포함
├── nlp_metrics.json           # seed 42, test set bootstrap CI
├── ablation.csv               # 최종 비교표
├── confusion_matrix_*.png
└── qualitative_cases.md       # RQ3용 사례 분석
```

---

## 5. 데이터셋

### 5.1 DFDC train part — Deepfake Vision

**출처**
- 원출처(Meta): https://ai.meta.com/datasets/dfdc/
- 실제 취득처: HuggingFace 미러 https://huggingface.co/datasets/gonnerthetooner/DFDC-extracted-full
- metadata만 별도: https://huggingface.co/datasets/scarlettss/dfdc_metadata

> **Decision — DFDC Acquisition (확정, 재논의 금지)**
>
> 본 프로젝트는 **Meta DFDC**를 Vision 학습 데이터로 사용한다. 48시간 프로젝트의 접근성과 재현 가능한
> 부분 다운로드를 위해 검증된 **HuggingFace mirror를 primary acquisition path**로 사용한다.
> **Kaggle은 공식 provenance 및 fallback 경로로 유지하되, 프로젝트 실행의 blocking dependency로 두지 않는다.**
> Raw DFDC media는 원 데이터 이용조건에 따라 저장소에 배포하지 않는다.

즉 **HF는 취득 경로이고, 데이터의 정체성은 DFDC다.** 보고서·발표에는 이렇게 표기한다.

```
Dataset            : Meta Deepfake Detection Challenge (DFDC)
Acquisition        : HuggingFace mirror (project-time accessibility)
Original data terms: Meta DFDC terms apply
```

"HF 미러를 썼으니 자유롭게 재배포 가능"은 **성립하지 않는다.** 전체 DFDC는 10만 개 이상이라 손댈 수 없으므로
**`dfdc_train_part_XX` 단위로 받는다.**

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('gonnerthetooner/DFDC-extracted-full', repo_type='dataset', allow_patterns='dfdc_train_part_00/**', local_dir='data/model/raw/dfdc')"
```

**part_00 실측** (2026-08-11 확인, 인증 불필요):

| 항목 | 값 |
|---|---|
| 영상 수 | **1,334** (mp4) |
| 라벨 분포 | FAKE **1,248** / REAL **86** → **14.5 : 1** |
| **고유 원본(family) 수** | **86** |
| 용량 | 약 **10.6 GB** |
| 사용 가능 part | `part_00` ~ `part_09` (10개) |

> 🚨 **이 표에서 가장 중요한 숫자는 1,334가 아니라 86이다.** family 단위로 자르면 **독립 표본은 86개**다.
> 70/15/15로 나누면 test family는 **13개**뿐이다. 영상 수를 표본 수로 착각하면 신뢰구간을 15배 좁게
> 보고하게 된다.

**어느 part를 받을 것인가 — part는 균질하지 않다.**

영상을 받기 전에 metadata만으로(수 MB) 50개 part의 family 규모를 먼저 계산했다. 결과는 **part_00이 최악**이다.

| part | 영상 | family | FAKE:REAL | **영상/family** | test family(15%) |
|---:|---:|---:|---:|---:|---:|
| 00 | 1,334 | 86 | 14.5:1 | **15.5** ← 최악 | 12 |
| 01 | 1,699 | 108 | 14.7:1 | 15.7 | 16 |
| **02** | **1,748** | **230** | **6.6:1** | **7.6** | **34** |
| 03 | 1,455 | 219 | 5.6:1 | 6.6 | 32 |
| 06 | 3,464 | 423 | 7.2:1 | 8.2 | 63 |
| 09 | 1,736 | 288 | 5.0:1 | **6.0** ← 최선 | 43 |

**같은 10GB를 받아도 part_00은 family 86개, part_09는 288개를 준다 — 3.3배 차이다.**
part_00·01은 한 원본에서 15개씩 파생시킨 반면 뒤쪽 part는 6개 수준이라, 앞쪽 part는
**같은 얼굴을 반복해서 받는 셈**이다.

> ✅ **결정: `part_02` 단독 취득.** family 230개 → test family 34개로 목표(30~40)를 한 번에 충족하고,
> 불균형도 6.6:1로 완화된다. part_00+01(24GB, family 194, test 29)보다 **적게 받고 더 얻는다.**
> 부족하면 `part_03`(+219 family)을 추가한다.

**family당 FAKE 상한(K)** — 학습 시 family당 FAKE를 최대 K개로 제한한다.

```
part_00 기준  family당 FAKE 수: 최소 1 / 중앙 11 / 최대 36
K=5 적용 시   FAKE 1,248 → 393   (불균형 14.5:1 → 4.6:1)
```

한 원본에서 파생된 36개 fake는 독립 관측 36건이 아니다. K 상한은 **학습시간 감소·family 편중 완화·
FAKE 클래스 지배 완화**를 동시에 해결한다. **단, 이 상한은 train split에만 적용하고 test split은
원본 분포를 유지한다** — 평가 대상 분포를 인위적으로 바꾸면 지표의 의미가 사라진다.

> ⚠️ **클래스 불균형이 예상보다 훨씬 심하다.** 당초 "FAKE가 많다" 정도로 적었으나 실측은 **14.5:1**이다.
> 전부 FAKE로 찍으면 Accuracy 93.6%가 나온다. §6.2에서 Accuracy를 주 지표로 쓰지 않기로 한 결정이
> 이 수치로 정당화된다.

**metadata 구조** (실물 확인)

```json
{ "owxbbpjpch.mp4": { "label": "FAKE", "split": "train", "original": "wynotylpnm.mp4" },
  "vpmyeepbep.mp4": { "label": "REAL", "split": "train" } }
```

`original` 필드가 FAKE 1,248건 전부에 존재한다 — **family split의 전제가 성립함을 확인했다.**
경로는 `dfdc_train_part_00/dfdc_train_part_0/metadata.json` (디렉터리가 한 단계 더 중첩된 점에 주의).

**내부 라벨 규약**

```
REAL = 0
FAKE = 1
```

> ✅ **육안 확인 완료 (2026-08-11, part_02)**. 같은 family의 REAL 원본과 파생 FAKE를 같은 프레임에서
> 나란히 뽑아 비교했다. 결과: **인물·의상·배경·자세가 완전히 동일하고 얼굴만 교체**돼 있었으며,
> 조작된 쪽에 FAKE 라벨이 붙어 있어 라벨 방향(FAKE=1)이 옳음을 확인했다.
>
> **이 관찰이 family split의 필요성을 그대로 증명한다.** 배경과 의상이 픽셀 단위로 같으므로, 이 쌍이
> train/test로 갈리면 모델은 조작 흔적이 아니라 "빨간 상의 + 흰 벽"을 외우고 정답을 맞힌다.
> 그러면 test AUROC가 비현실적으로 높게 나오고 아무도 눈치채지 못한다.
>
> (비교 이미지는 실제 인물의 얼굴이므로 저장소에 커밋하지 않는다 — License 절 참조.)

**영상 규격 실측**: 1920×1080 / 30fps / 300 frame(10초). 샘플 8건 전부 OpenCV로 정상 디코딩.
§8.1의 `sample_fps=2, max_frames=20` 설정은 10초 영상에서 정확히 20 프레임을 뽑는다 — 일치한다.
디코딩 비용은 영상당 1.94초로, 1,748개 전체가 단일 프로세스 57분 / 멀티프로세스 10~15분이다.

**⚠️ 가장 중요한 전처리 조건 — family 단위 split**

DFDC의 fake 영상은 특정 real 영상에서 파생된다. 파생 관계가 split을 가로지르면
모델은 조작 흔적이 아니라 **인물 얼굴과 배경을 외운다.** 그러면 테스트 AUROC가 비현실적으로 높게 나오고,
아무도 눈치채지 못한 채 "성공"으로 보고된다.

```
❌ 잘못됨              ✅ 올바름
TRAIN: real_A         TRAIN: real_A, fake_A_1, fake_A_2
TEST:  fake_A_1       TEST:  real_C, fake_C_1
```

`original` 필드로 family를 묶고, **family 전체를 하나의 단위로** train/val/test에 배정한다.
`split_report.json`에 family 수·영상 수·클래스 비율을 남긴다.

### 5.2 Fin-Fact Dataset — Financial NLP

> ⚠️ **명칭 고정 (혼동 주의).** 이 리포의 이전 프로젝트 이름이 `FinFact`였고, 이번 NLP 학습 데이터셋 이름도
> `Fin-Fact`다. 문서·코드에서 다음 표기를 고정한다.
>
> | 표기 | 의미 |
> |---|---|
> | **Legacy FinFact Project** | 이미지-텍스트 entity 불일치를 다루던 **이전 프로젝트** (피벗으로 폐기) |
> | **Fin-Fact Dataset** | 이번 프로젝트의 **NLP 학습 데이터셋** (IIT-DM) |
>
> 디렉토리는 `data/model/finfact_dataset/`로 쓴다. `finfact/`만 쓰면 반드시 다시 사고가 난다.

**출처**
- GitHub: https://github.com/IIT-DM/Fin-Fact
- HuggingFace: https://huggingface.co/datasets/amanrangapur/Fin-Fact

```python
from datasets import load_dataset
ds = load_dataset("amanrangapur/Fin-Fact")
```

**실제 컬럼** (실측): `url`, `claim`, `author`, `posted`, `sci_digest`, `justification`, `issues`,
`image_data`, `evidence`, `label`, `visualization_bias`. 이 중 **`claim` · `evidence` · `label`**만 쓴다.

**규모와 라벨 분포** (2026-08-11 실측):

| 항목 | 값 |
|---|---|
| 총 건수 | **3,369** (split은 `train` 하나뿐 — **자체 분할 필요**) |
| 라벨 | `false` 1,485 / `true` 1,273 / `neutral` 611 |
| 최소 클래스 | 611 |

> ✅ **Day 2 게이트 ① 통과 — 클래스 병합 불필요.** 최소 클래스가 611로 기준(50)을 크게 넘는다.
> 3-class Macro-F1을 그대로 쓴다.
>
> ⚠️ **데이터셋이 split을 제공하지 않는다.** train/val/test를 직접 만들어야 하며, 라벨 층화(stratify)와
> seed 42 고정, 분할 해시 기록을 Vision과 동일하게 적용한다.

> 🚨 **Day 2 게이트 ② 발동 — evidence가 max_len을 크게 초과한다.** 중앙 574 / P95 1,753 / 최대 8,046 토큰.
> 처리 방식(claim-guided 문장 선택)과 근거는 **§8.2**에 확정해 두었다. 학습 전에 반드시 읽을 것.

Evidence는 옵션이 아니라 **H2의 핵심 변수**다 (§7.2).

### 5.3 얼굴 검출기 — OpenForensics로 직접 fine-tuning한다 (설계 변경)

> **Decision — Object Detection 축 신설 (2026-08-11, 기존 결정 번복)**
>
> 초기 설계는 "bbox GT가 없으니 detector는 pretrained만 쓴다"였다. **이건 데이터 선택이 만든
> 제약을 설계 원칙처럼 쓴 오류였다.** deepfake 도메인에 face-wise bbox GT가 이미 붙어 있는
> 데이터셋(OpenForensics, ICCV 2021)이 존재하고, 접근성·포맷·학습 비용을 전부 실측 검증했다.
> **YOLO를 직접 fine-tuning하고 mAP를 정식으로 계산한다.**

- pretrained 초기 가중치: https://github.com/lindevs/yolov8-face (WIDER FACE)
- fine-tuning 데이터: **OpenForensics Validation split** (§5.6)

**단, DFDC를 대체하지 않는다 — 축을 분리한다.** OpenForensics는 순수 이미지 데이터셋이라
(§5.6) 한 이미지에 real face와 forged face가 섞여 있어 **full-frame 라벨(V0)이 정의되지 않고**,
`original` 파생관계가 없어 family split·video-level 평가도 성립하지 않는다. 교체하면 H1이 통째로
증발한다. 그래서:

```
축 ①-a  Deepfake 분류    DFDC part_02 (영상)          V0 vs V1 · family split · video-level   [불변]
축 ①-b  Face Detection   OpenForensics Val (이미지)    YOLOv8n fine-tuning · mAP@50 · D1 vs D2  [신설]
축 ②    Claim 검증       Fin-Fact                      N1 vs N2                                 [불변]
```

이로써 **세 모델(YOLO · EfficientNet · DeBERTa)을 전부 직접 학습한다.**

### 5.6 OpenForensics — Face Detection 학습 데이터 (신설)

**출처**: Zenodo record 5528418 (DOI `10.5281/zenodo.5528418`) — **승인·계정 불필요 (open access)**
- 논문: *OpenForensics: Large-Scale Challenging Dataset for Multi-Face Forgery Detection and Segmentation* (ICCV 2021)
- 전체 115,325 이미지 / 334,136 얼굴, face별 **bbox + real/forged 라벨 + segmentation polygon** (표준 COCO JSON)

**역할 분담 — Train으로 학습하고 Val로 평가한다** (공식 split 프로토콜 준수):

| split | 용도 | 취득 |
|---|---|---|
| **Train** (44,122장, 19.9GB, zip 5개 전부 필요) | **D2 fine-tuning** — 단, 전량이 아니라 **부분표본 ~10k장**(seed 42 고정 추출)으로 학습 | D-1 밤 백그라운드 (약 4시간, 시계 밖) |
| **Val** (7,308장 / 15,345 얼굴, Real 4,782:Fake 10,563) | **D1·D2 공통 평가셋** — 학습에 쓰지 않는다 | 3.24GB ≈ 40분 ✅ |

**Val을 학습에 쓰지 않는 이유**: D1(zero-shot)과 D2(fine-tuned)를 **같은 Val에서** 재야 ΔmAP가
공정한 비교가 된다. Val을 쪼개 학습에 쓰면 저자 공식 경계를 깨고 평가셋도 줄어든다.

**Train을 전량 쓰지 않는 이유** (실측): Zenodo가 0.79MB/s라 19.9GB ≈ 4시간 — 다운로드는 D-1 밤
시계 밖으로 밀면 흡수되지만, **44,122장 학습은 47분/epoch × 15 = 12시간이라 시계 안에서 기각**이다.
~10k장 부분표본이면 11분/epoch × 15 ≈ 3시간으로 성립한다. 표본 추출 seed와 목록을 `results/`에 기록한다.

> **Fallback**: D-1 밤 Train 다운로드가 실패하면(미러 소멸·회선 문제) Val 7,308장을 내부 분할해
> 학습하되, **저자 공식 split 경계를 깼음을 한계로 명시**한다. 이건 차선이지 정본이 아니다.

| 기타 실측 | 값 |
|---|---|
| 해상도 | 최대 1024×1024, bbox 중앙값 한 변 ≈191px — YOLO 640 입력에 적정 |
| 어노테이션 | `category_id` 0=Real 1=Fake. COCO 표준, 변환 15줄/0.07초, 좌표 이상치 0건 |
| YOLOv8n FT | GTX 1650, 640², AMP, batch 16: peak **2.15GB** / 7.3k장 1 epoch ≈ 8분 |

> ⚠️ **identity 누수는 검증 불가다 — 검증했더니 없더라가 아니다.** 어노테이션에 source identity
> 필드가 없어, 같은 인물·같은 촬영 세트가 분할을 가로지르는지 **확인할 방법 자체가 없다.**
> DFDC에서 `original` 필드로 했던 leakage 검사를 여기서는 할 수 없으며, 이를 한계로 명시한다.
> 자체 재분할 대신 **저자 공식 split 경계를 존중**하는 이유이기도 하다.

> ⚠️ **라이선스 표기가 출처마다 다르다.** Zenodo API는 CC-BY-4.0, 프로젝트 페이지는 CC-BY-NC-SA-4.0,
> GitHub는 "academic purpose only". **가장 제한적인 해석(비영리·학술, 재배포 금지)을 채택**한다.
> 원본이 Google Open Images의 실존 인물 사진이므로, 얼굴 이미지 커밋 금지 규정(License 절)이
> DFDC와 동일하게 적용된다.

### 5.4 Synthetic Integration Test Case — 시연용 fixture (학습·평가 데이터 아님)

**DFDC 영상 속 사람들은 금융 얘기를 하지 않는다.** 일반인이 아무 말이나 하는 영상이다.
따라서 DFDC로 통합 데모를 돌리면 NLP 칸이 **항상 비어 있다.**

그래서 DFDC 영상에 금융 발언 오디오를 붙인 fixture를 만든다. 다만 이것을 **실제 피해사례처럼 제시하면 안 된다** —
얼굴 영상과 합성 음성이 따로 놀아 입 모양이 안 맞고, 관객이 즉시 "합성한 데모"임을 알아챈다.
그러면 시연의 신뢰도가 오히려 떨어진다.

따라서 명시적으로 **Synthetic Integration Test Case**라고 라벨링하고, §8.3 risk matrix의 네 칸에 대응하는
**4개만** 만든다. 개수를 늘릴 이유가 없다 — 목적은 성능 측정이 아니라 **시스템이 네 조합에서 각각 올바르게 동작함을 보이는 것**이다.

| fixture | Media | Claim | 확인하는 것 |
|---|---|---|---|
| F1 | REAL | SUPPORTED | 정상 경로 (오탐 없음) |
| F2 | REAL | REFUTED | 실제 영상 + 근거와 불일치하는 주장 → CLAIM RISK |
| F3 | FAKE | SUPPORTED | 합성 미디어지만 내용은 사실 → SYNTHETIC MEDIA |
| F4 | FAKE | REFUTED | 합성·조작 미디어 + 근거와 불일치하는 주장 → HIGH RISK |

**이건 모델 평가 데이터셋이 아니라 시스템 동작 시연용 fixture다.** §6.4 결과표에 어떤 수치도 기여하지 않는다.

### 5.5 (선택) FakeAVCeleb

https://github.com/DASH-Lab/FakeAVCeleb — 영상+합성 음성을 함께 제공해 이상적이지만,
**다운로드에 access request 승인이 필요하다.** 이틀 일정의 주 데이터로 의존하면 안 된다.
승인이 미리 나 있으면 external test set으로만 쓴다.

---

## 6. 평가 설계

평가는 이 프로젝트에서 가장 공들인 부분이다. **Accuracy 한 줄로 끝내지 않는다.**

### 6.1 왜 순진한 평가가 여기서 특히 위험한가

세 가지 함정이 겹쳐 있다.

**함정 1 — frame-level AUROC는 가짜 정밀도를 만든다.**
같은 영상에서 뽑은 20개 프레임은 서로 독립이 아니다. frame 단위로 재면 n이 수천으로 부풀고
신뢰구간이 인위적으로 좁아진다. → **모든 vision 지표는 video-level에서만 보고한다.**
프레임 확률은 median으로 집계한다 (평균보다 검출 실패 프레임에 강건).

**함정 2 — 클래스 불균형에서 Accuracy는 무의미하다.**
실측: DFDC part_00~09의 FAKE:REAL은 **6.9 : 1**이고, part_00만 보면 **14.5 : 1**이다.
전부 FAKE로 찍기만 해도 Accuracy **87.4%**(part_00 단독이면 93.6%)가 나온다.

```
Primary   : AUROC · AUPRC
Secondary : Macro-F1 · FAKE Recall · Balanced Accuracy
참고용    : Accuracy
```

**AUPRC를 주 지표에 넣는 이유**는 불균형 데이터에서 AUROC가 낙관적으로 나오기 때문이다.
음성(REAL)이 희소하면 위양성 몇 건이 FPR을 거의 움직이지 못해 ROC 곡선이 좋아 보이는데,
PR 곡선은 그걸 그대로 드러낸다.

**함정 3 — 표본이 작아서 작은 차이는 노이즈다.**
**독립 표본은 영상이 아니라 family다.** `part_02` 기준 family 230개, 70/15/15 분할 시
**test family는 34개**다. 영상 수(1,748)를 표본 수로 쓰면 신뢰구간이 7배 좁게 나온다.
bootstrap도 **family 단위로 resampling**해야 하며, 영상 단위로 하면 같은 오류를 반복한다.
이 규모에서 **AUROC 0.02 차이는 아무 의미가 없다.**
→ 모든 수치에 **bootstrap 95% CI(2000회 resampling)**를 붙이고,
모델 A vs B는 **같은 테스트 영상에 대한 paired bootstrap**으로 비교한다.
CI가 0을 포함하면 "개선 없음"이라고 쓴다. 이건 사후에 정하지 않고 지금 못 박는다.

### 6.2 지표 정의

| 모듈 | 주 지표 | 보조 | 단위 | 불확실성 |
|---|---|---|---|---|
| Deepfake | **AUROC + AUPRC** | Macro-F1, FAKE Recall, Balanced Accuracy, Accuracy(참고) | **video** | bootstrap 95% CI |
| Deepfake A vs B | **ΔAUROC** | — | video (paired) | paired bootstrap CI |
| NLP | **Macro-F1** | per-class P/R/F1, Accuracy | claim | **test set bootstrap CI** (seed 42 고정) |
| Face Detection | **Detection Success Rate** | 수동 스팟체크 정확도 | frame | — |
| STT | WER (선택) | — | 샘플 | 필수 아님 |

**NLP 불확실성은 3-seed가 아니라 bootstrap CI로 낸다.** 이틀짜리에서 중요한 건 seed 간 분산 측정이 아니라
**N1 vs N2 격차가 실재하는가**이며, 그건 같은 test set 위의 bootstrap으로 답할 수 있다.
3구성 × 3seed = 9회 학습은 4GB VRAM에서 Day 2의 3시간 슬롯에 들어가지 않는다.
seed는 42로 고정하고 `split_report.json`에 기록한다.

**FAKE Recall을 반드시 별도 보고한다.** 금융 사기 스크리닝에서 놓친 deepfake의 비용이
잘못 경보한 진짜 영상의 비용보다 훨씬 크다. 운영 임계값은 FAKE Recall 기준으로 잡는다.

### 6.3 Object Detection 지표 — 축별로 다르다

**OpenForensics 축(①-b)에서는 mAP를 정식으로 계산한다.** GT bbox가 있기 때문이다.

```
Primary   : mAP@50 · mAP@50:95
보조      : AP(Real) · AP(Fake) 클래스별, Precision, Recall
표본 단위 : 이미지 (각 이미지가 독립 촬영본 — DFDC 프레임과 달리 비독립성 문제 없음)
```

초기 설계는 "mAP는 계산할 수 없다"였다 — DFDC에 bbox GT가 없었기 때문이다. 그건 데이터의
제약이었지 과제의 본질이 아니었고, OpenForensics 도입으로 이 빈칸이 메워졌다.

**DFDC 축(①-a)에는 여전히 bbox GT가 없다.** 거기서는 기존 두 지표를 유지한다.

```
Detection Success Rate = 얼굴 ROI를 정상 추출한 영상 수 / 전체 영상 수
Manual Spot Check      = 무작위 50 프레임의 bbox를 눈으로 검수, 오검출·미검출 건수 기록
```

**채울 수 없는 칸을 발표 자료에 넣는 것이 가장 큰 리스크다**라는 원칙은 그대로다 — 바뀐 건
"채울 수 있는 데이터를 찾아냈다"는 것이다.

### 6.4 최종 결과표 (이 표를 채우는 것이 프로젝트 완료 조건)

**핵심 비교는 세 개다** — `V0 vs V1`(ROI가 분류에 도움이 되는가), `D1 vs D2`(deepfake 도메인
fine-tuning이 검출을 개선하는가), `N1 vs N2`(evidence가 검증에 도움이 되는가).

| # | 모듈 | 구성 | 주 지표 | 값 | 95% CI | 우선순위 |
|---|---|---|---|---|---|---|
| V0 | Vision(분류) | Full frame → EfficientNet-B0 | AUROC | – | – | **필수** |
| V1 | Vision(분류) | YOLO ROI (tight) → EfficientNet-B0 | AUROC | – | – | **필수** |
| — | Vision(분류) | **V1 − V0 (paired)** ← H1 | ΔAUROC | – | – | **필수** |
| D1 | Detection | YOLOv8n-Face pretrained, zero-shot — **OpenForensics Val 평가** | mAP@50 | – | – | **필수** |
| D2 | Detection | YOLOv8n-Face **Train(~10k) fine-tuned** — 동일 Val 평가 | mAP@50 | – | – | **필수** |
| — | Detection | **D2 − D1** ← H3 | ΔmAP@50 | – | – | **필수** |
| N1 | NLP | DeBERTa-v3-small, claim only | Macro-F1 | – | bootstrap | **필수** |
| N2 | NLP | DeBERTa-v3-small, claim + evidence | Macro-F1 | – | bootstrap | **필수** |
| — | NLP | **N2 − N1** ← H2 | ΔMacro-F1 | – | bootstrap | **필수** |
| D0 | Detection | DFDC 축 Detection Success Rate + 스팟체크 | Success Rate | – | – | **필수** |
| V2 | Vision(분류) | YOLO ROI (margin 1.3×) → EfficientNet-B0 | AUROC | – | – | ❌ **사전 포기 확정** (아래) |
| N0 | NLP | DistilBERT, claim only | Macro-F1 | – | bootstrap | 부가 |

> **V2는 "지연 시 포기"가 아니라 지금 포기를 확정한다.** GPU가 한 장인데 D2(YOLO fine-tuning
> 약 3시간)가 새로 들어왔다. Day 1 학습 슬롯에서 V0·V1·D2를 돌리면 V2 자리가 없다.
> §11의 포기 순서 1순위를 사전 발동하는 것이며, H1-b는 철회한다.

---

## 7. Ablation — 무엇을 왜 비교하는가

### 7.1 Vision: H1 / H1-b

```
V0  Full Frame ──────────────────► EfficientNet ──► P(fake)
V1  Frame ─► YOLO ─► tight crop ──► EfficientNet ──► P(fake)
V2  Frame ─► YOLO ─► 1.3× crop ───► EfficientNet ──► P(fake)
```

**V1 − V0**이 Object Detection을 파이프라인에 넣은 이유를 정량적으로 설명한다.
**V2 − V1**(margin ablation)은 GPU 슬롯을 D2에 내주면서 **사전 포기를 확정**했다 (§6.4). H1-b 철회.

### 7.1.1 Detection: H3 — YOLO를 직접 학습한다 (신설)

H3가 검증하는 것: **deepfake 도메인 데이터로 face detector를 fine-tuning하면,
일반 얼굴 데이터(WIDER FACE)로 학습된 pretrained detector 대비 검출 성능이 개선되는가 (D2 − D1).**

```
                     OpenForensics Train (부분표본 ~10k, seed 42)
                                   ↓
D1  YOLOv8n-Face (WIDER FACE pretrained) ── zero-shot ──►  OpenForensics Val  →  mAP@50
D2  YOLOv8n-Face + Train fine-tune ─────────────────────►  동일 Val           →  mAP@50
                                                                              →  ΔmAP = H3
```

**Train으로 학습하고 Val로 평가한다 — Val은 두 모델의 공통 평가셋이며 학습에 쓰지 않는다** (§5.6).

**클래스는 FACE 하나로 통일한다** (Real/Fake bbox를 모두 class 0으로 변환). 이유:

- **역할이 겹치면 안 된다.** YOLO가 Real/Fake 2-class로 검출하면 EfficientNet(분류 축)과 같은 문제를
  두 모델이 풀게 되어 "왜 둘 다 있는가"에 답이 없어진다. YOLO는 **어디에 얼굴이 있는가**,
  EfficientNet은 **그 얼굴이 진짜인가**로 의존 관계를 만든다.
- **D1과의 비교가 공정해진다.** pretrained detector는 FACE 단일 클래스라, 2-class로 학습하면
  D1 vs D2가 같은 과제의 비교가 아니게 된다.
- 단, GT의 real/forged 라벨은 버리지 않는다 — **forged face에 대한 recall을 따로 보고**해
  "fine-tuning이 특히 합성 얼굴 검출을 개선하는가"를 본다. 이건 2-class 학습 없이도 계산된다.

fine-tuning은 **Train split 부분표본(~10k, seed 42)**으로, 평가는 **Val 전체**로 한다 (§5.6).
Val을 쪼개 학습에 쓰지 않는다 — 저자 공식 split 경계를 존중하고 D1/D2의 평가셋을 동일하게 유지하기 위해서다.

### 7.2 NLP: H2 — 이 프로젝트에서 가장 중요한 ablation

H2가 검증하는 것: **evidence를 함께 사용하면 금융 claim 검증 성능이 개선되는가 (N2 − N1).**

> ⚠️ **전제 게이트 (Day 2 첫 작업, 필수)**: H2가 메인 검증으로 승격되었으므로
> Fin-Fact **evidence 필드의 토큰 길이 분포를 학습 전에 반드시 측정**한다.
> evidence가 max_length(512)를 크게 초과하면 N2는 "evidence를 본 모델"이 아니라
> "evidence 앞조각을 본 모델"이 되어 아래 해석표가 오염된다. 초과 시 처리 방식
> (선두 절단 명시 / 관련 문장 추출)을 **학습 전에** 결정하고 기록한다.

왜 claim-only가 순진한 설계인가: `Claim → DeBERTa → Label`이 잘 동작해도
**무엇을 학습한 건지 설명할 수 없다.**

생각해보면 명백하다. "Apple이 Tesla를 3천억 달러에 인수한다"가 참인지 거짓인지,
**증거 없이 문장만 보고 알 방법은 없다.** 그런데도 모델이 맞힌다면 그건 사실 검증이 아니라
**주제·문체·출처 아티팩트**를 학습한 것이다 (fact verification 문헌에서 반복 보고된 문제다).

그래서 이렇게 설계한다.

```
N1   Claim              → DeBERTa → Label      # 아티팩트 상한선
N2   Claim + Evidence   → DeBERTa → Label      # 실제 검증에 가까움
```

**N2 − N1 격차 자체가 결과다.**

| 관찰 | 해석 |
|---|---|
| N1이 이미 높다 | 데이터셋에 아티팩트가 강하다 → 정직하게 보고, claim-only 수치의 의미 축소 |
| N2 ≫ N1 | evidence가 실제 판별 정보를 제공 → NLP 축의 기여가 입증됨 |
| N2 ≈ N1 | 모델이 evidence를 활용하지 못함 → 한계로 명시 |

**어느 결과가 나와도 보고할 내용이 있다.** 이게 좋은 실험 설계의 조건이다.
그리고 학습 한 번 더 돌리는 비용밖에 안 든다.

> 결과를 본 뒤에 서사를 바꾸지 않는다. 위 해석표는 **실험 전에** 확정한 것이다.

---

## 8. 구현 세부

### 8.1 전처리

**Vision**

```python
sample_fps  = 2       # 초당 2 프레임
max_frames  = 20      # 영상당 상한
crop_size   = 224
margin      = 1.0     # V1 / V2에서 1.0 vs 1.3
```

프레임당 confidence가 가장 높은 얼굴 하나만 사용한다 (다중 얼굴은 Nice-to-Have).
얼굴이 한 프레임도 안 잡히면 해당 영상을 `detection_failed`로 기록하고
**분류 평가에서 제외하되 Detection Success Rate에는 반드시 포함**한다.
검출 실패 영상을 조용히 버리면 성능이 낙관적으로 왜곡된다.

정규화는 ImageNet 통계 (`mean=[0.485,0.456,0.406]`, `std=[0.229,0.224,0.225]`).

**Audio → NLP**

```bash
ffmpeg -i input.mp4 -ar 16000 -ac 1 output.wav
```

Whisper로 전사 후 문장 분리 → 금융 키워드 포함 문장만 선택:

```
revenue, profit, loss, earnings, guidance, acquisition, merger,
stock, shares, dividend, investment, billion, million, quarter, SEC
```

키워드 필터는 학습 대상이 아닌 **규칙 기반 라우팅**이다. README와 발표에서 그렇게 명시한다.

### 8.2 학습

| | Deepfake Classifier | Claim Classifier |
|---|---|---|
| 모델 | EfficientNet-B0 (ImageNet) | DeBERTa-v3-small / DistilBERT |
| 1단계 | backbone freeze, 2–3 epoch | — |
| 2단계 | 상위 블록 unfreeze, 2–5 epoch | 3–5 epoch |
| LR | 1e-3 → 1e-4 | 2e-5 |
| Batch | 32 | **16** (N1 64토큰 / N2 256토큰) |
| max_len | — | **N1 64 · N2 256** (실측 근거는 아래) |
| 기타 | AMP, early stopping | AMP, early stopping, seed 42 고정 |

#### 실측 — GTX 1650 4GB에서 512는 쓸 수 없다

DeBERTa-v3-small 학습 스텝을 직접 측정했다 (AMP fp16, 3,369건 1 epoch 환산):

| max_len | batch | peak VRAM | 1 epoch | 판정 |
|---:|---:|---:|---:|---|
| 512 | 8 | **4.40 GB** | 39.7분 | ❌ 카드 용량(4.29GB) 초과 — 스텝이 batch 4보다 **2배 느리다**(5.7s vs 2.7s). 메모리 스래싱 |
| 512 | 4 | 3.14 GB | 38.5분 | ⚠️ 돌지만 느림. 4 epoch = 2.6시간 |
| **256** | **16** | 3.99 GB | **16.2분** | ✅ 채택 |

**batch를 8로 키웠더니 오히려 2배 느려진 것**이 결정적 증거다. 4.40GB는 이 카드에 안 들어가고,
초과분이 호스트 메모리로 밀려나면서 스텝 시간이 무너진다. Day 2 오전 3시간 슬롯에 N1·N2를
모두 넣으려면 512는 성립하지 않는다.

#### 그래서 evidence 처리 방식이 강제된다

Fin-Fact 토큰 길이 실측 (n=3,369):

```
claim     중앙   14  P95    29  최대    66   ← 짧다. N1은 max_len 64로 충분
evidence  중앙  574  P95 1,753  최대 8,046   ← 길다
max_len=512 → 43.6%만 절단 없음
max_len=256 → 17.9%만 절단 없음
```

**앞에서부터 자르는 방식(head truncation)은 채택하지 않는다.** 256에서 82%가 잘리면 N2는
"evidence를 본 모델"이 아니라 "evidence 첫 문단을 본 모델"이 되고, §7.2의 해석표가 통째로 오염된다.

> **결정: claim-guided 문장 선택.** evidence를 문장 단위로 쪼갠 뒤 claim과의 어휘 유사도(TF-IDF)
> 상위 문장부터 256 토큰 예산이 찰 때까지 담는다. 선택된 문장은 원문 순서를 유지한다.
>
> 이것은 **외부 검색이 아니다.** 검색 범위가 "이 claim에 딸려 제공된 evidence 필드 내부"로 한정되므로
> §2.1의 "Evidence Retrieval System이 아니다"와 충돌하지 않는다. 발표에서는
> **"제공된 evidence 내에서 claim 관련 문장을 선택했다"**고 정확히 표현한다.
>
> 선택 전후의 토큰 길이 분포와 평균 문장 보존율을 `results/nlp_metrics.json`에 기록한다.

**OOM 대응은 미리 config 기본값으로 넣어둔다** — AMP 강제, batch 축소 + gradient accumulation으로
effective batch를 유지. Day 1 16:00에 OOM을 발견하면 학습 슬롯을 통째로 날린다.

**모델 계약** — Vision/NLP 모두 동일 인터페이스를 지킨다:

```python
model(batch: dict) -> logits  # [B, num_classes]
```

이 계약 하나로 Trainer·평가 코드를 두 축이 공유한다. 이틀 일정에서 이건 사치가 아니라 필수다.

### 8.3 두 축을 왜 합치지 않는가

다음과 같은 융합은 **하지 않는다.**

```
0.6 × Deepfake Score + 0.4 × Claim Score = Fake News Score   ← 근거 없음
```

가중치를 정할 근거가 없다. deepfake 라벨과 금융 claim 라벨이 **동시에 붙어 있는 데이터셋이 존재하지 않기 때문**이다.
임의의 계수를 붙이면 숫자는 나오지만 검증할 방법이 없고, 이건 데이터셋의 한계를 숨기는 행위다.

대신 두 축을 그대로 두고 **해석만 조합**한다.

| Media Authenticity | Claim Verification | 판정 | 의미 |
|---|---|---|---|
| REAL | Supported | **LOW** | 실제 영상 + 근거에 의해 지지되는 주장 |
| SYNTHETIC | Supported | **SYNTHETIC MEDIA** | 합성·조작 미디어지만 주장은 제공된 근거에 의해 지지됨 |
| REAL | Refuted | **CLAIM RISK** | 실제 영상이지만 제공된 근거와 일치하지 않는 주장 |
| SYNTHETIC | Refuted | **HIGH RISK** | synthetic media + refuted claim; 기만 의도 자체는 모델이 판정하지 않음 |

**이 4칸 구조가 RQ3의 답이다.** 단일 확률 하나로는 두 번째·세 번째 칸을 구분할 수 없다.
단, **disinformation은 기만 의도까지 포함하는 개념이므로 모델의 직접 출력 라벨로 사용하지 않는다.** 발표에서는 각 칸을 Synthetic Integration Test Case로 시연한다.

---

## 9. 이 프로젝트가 주장하지 않는 것

정직한 한계 명시는 감점 요인이 아니라 설계 역량의 증거다.

- **"AI가 금융 뉴스를 자동 팩트체크한다"고 주장하지 않는다.**
  claim classifier는 외부 evidence retrieval을 하지 않는다. 임의의 새로운 발언에 대한
  절대적 진실 판별기가 아니라, **주어진 claim과 evidence의 관계를 Fin-Fact 라벨 체계 안에서 판정하는 모델**이다.
- **DFDC sample에서의 성능이 실제 금융 deepfake에 이전된다고 주장하지 않는다.**
  DFDC는 금융 도메인 데이터가 아니다. 도메인 이전은 검증되지 않았고, 이를 한계로 명시한다.
- ~~"YOLO를 학습하지 않는다"~~ → **이제 학습한다** (§5.3 설계 변경). 정확한 표현:
  **"OpenForensics의 GT bounding-box annotation으로 YOLOv8n-Face를 fine-tuning하고
  mAP@50으로 정식 평가했으며(D1 vs D2), 검출된 얼굴 ROI가 downstream deepfake 분류에 미치는
  효과를 별도로 검증했다(V0 vs V1)."** — bbox를 직접 그렸다고는 말하지 않는다. GT는 데이터셋이 제공한다.
- **cross-dataset transfer 성능을 주장하지 않는다.** OpenForensics에서 fine-tuning한 detector를
  DFDC 프레임의 **ROI extractor로 transfer**하지만, DFDC 프레임에는 bbox GT가 없어
  **OpenForensics → DFDC cross-dataset detection 성능은 직접 검증하지 못한다.**
  (YOLO의 입력은 영상 자체가 아니라 추출된 frame image다 — "이미지→영상 이전"이 아니라
  데이터셋 간 이전이며, 그 정확한 한계가 이것이다.)
- **N1(claim-only) 점수가 높다고 사실 검증 능력의 증거로 해석하지 않는다** (§7.2).
- **사례에서 나온 문제를 전부 푼다고 주장하지 않는다.** 코딩한 15건에서 검증 실패는 4개 유형으로
  나왔고 우리가 답하는 것은 A(발언 영상 진위)와 C(주장 근거 일치 여부)뿐이다. **B(채널 정통성) 6/15는 범위 밖**이다 —
  "이 계정·앱·링크가 공식인가"는 도메인 평판·인증서·플랫폼 메타데이터의 문제이고 콘텐츠 분석으로
  답할 수 없다. 실제로 C02·C05·C07은 콘텐츠가 아니라 **콘텐츠가 놓인 컨테이너**에서 뚫렸다 (§1.1).
- **코딩 사례의 87%(13/15)만 영상 기반이다.** C07(사칭 이메일)·C09(오픈채팅 사칭)처럼 영상이 없는
  경로도 존재하며, 영상 입력을 전제하는 이 PoC는 그 경로를 다루지 못한다.

실서비스 수준으로 가려면 이 단계가 추가되어야 한다:

```
① 축 확장   Claim → Evidence Retrieval → SEC Filing / IR / Trusted News → NLI Verification
② 축 확장   채널 정통성 검증 — 공식 계정·도메인 대조, 인증서, 플랫폼 메타데이터 (유형 B)
```

---

## 10. 폴더 구조

```
**`data/research/`와 `data/model/`은 목적이 다르다.** 크롤링 수집분은 **문제정의(pain point 발견)용**이고,
DFDC·Fin-Fact Dataset은 **모델 학습·평가용**이다. 둘은 섞이지 않는다 — 크롤링한 뉴스로 NLP를 학습하지 않는다.
발표에서 "크롤링한 기사로 학습했나요?"라는 질문을 받아도 이 구조면 답이 꼬이지 않는다.

```
findeepfake-48h/
├── README.md
├── requirements.txt
├── configs/
│   ├── vision_fullframe.yaml
│   ├── vision_roi.yaml
│   └── nlp_claim.yaml
├── data/
│   ├── research/                 # 문제정의용 — 학습에 쓰지 않음
│   │   ├── crawled/              #   크롤링 raw (git 제외, §1.1)
│   │   ├── coded_cases.csv       #   심층 코딩 15건 ← git 포함
│   │   ├── painpoint_summary.md  #   Pain Point Map  ← git 포함
│   │   └── source_manifest.csv   #   출처·수집일 목록 ← git 포함
│   └── model/                    # 학습·평가용
│       ├── raw/{dfdc,finfact_dataset}/
│       ├── processed/{frames,faces,audio,text}/
│       ├── fixtures/             #   §5.4 Synthetic Integration Test Case 4개
│       └── splits/{train,val,test}.csv + split_report.json
├── src/
│   ├── common/                   # config, seed, trainer, metrics (두 축 공유)
│   ├── preprocess/               # extract_frames, detect_faces, extract_audio, prep_finfact
│   ├── vision/                   # dataset, model, train, evaluate
│   ├── nlp/                      # dataset, model, train, evaluate
│   └── inference/pipeline.py
├── models/
├── results/
└── demo/app.py
```

---

## 11. 48시간 일정

### D-1 이전 — pain-point 크롤링 (48시간 시계 밖)

- [x] 크롤러 실행 → 사례 수집 ([README_CRAWLERS.md](README_CRAWLERS.md)) — 8/10 채널, 2,995건 (2026-08-11)
- [ ] **사건 코딩 표 + Pain Point Map** → `data/research/coded_cases.csv` (§1.1.1)

### D-1 (전날 밤) — 이걸 안 하면 이틀이 이틀이 아니다

우선순위는 **프로젝트 논리 > 환경 정비**다. CUDA가 막히면 Colab이라는 우회가 있지만,
문제정의 근거가 비면 "왜 이 모델을 만들었습니까?"에 답할 방법이 없다.

| 순위 | 항목 | 상태 | 비고 |
|---|---|---|---|
| **P0-A** | **사건 코딩 완료** (§1.1.1) | ✅ | 15건 코딩 완료. 48시간 일정표에 이 작업 슬롯은 **0분**이라 시계 시작 전에 끝내야 했다 |
| **P0-B** | HF에서 `dfdc_train_part_02` 다운로드 | ✅ | 1,748개 / 9.5GB 확보, 라벨 육안 확인·family split 완료 (§5.1) |
| **P0-D** | **OpenForensics `Val.zip`+`Val_poly.json` 다운로드** (3.24GB, 평가셋) | 🔄 | **Zenodo가 0.79MB/s로 느리다 — 약 40분.** 승인 불필요 (§5.6) |
| **P0-E** | **OpenForensics `Train_part_1~5`+`Train_poly.json` 다운로드** (20.4GB, D2 학습용) | ⬜ | **약 4시간 — D-1 밤 백그라운드로.** zip 5개 전부 있어야 압축해제된다. 실패 시 Val 내부분할 fallback (§5.6) |
| **P0-C** | 학습환경 — `torch.cuda.is_available()` | ✅ | `torch 2.12.1+cu126` / GTX 1650 (compute 7.5, 4.29GB) / AMP fp16 동작 확인 |
| P1 | `ffmpeg` 설치 | ✅ | 9.0 (winget `Gyan.FFmpeg`) |
| P1 | `requirements.txt` 재작성 | ✅ | FinDeepfake 기준으로 전면 교체. **torch는 의도적으로 제외** — CPU 빌드가 깔리면 학습이 통째로 막힌다 |
| P1 | Whisper·YOLOv8n-Face 가중치 사전 다운로드 | ⬜ | whisper 구현체 확정 후 (오너 미결) |
| P2 | Legacy FinFact 자산 정리 | ⬜ | **블로커 아님.** 신규 `src/`는 새로 만들면 되고, 이틀짜리에서 파일 이동하다 conflict 내면 손해. `git tag legacy-finfact-final` 찍어두고 **정리는 마지막에** |

**환경 설치 명령** (재현용):

```bash
pip install --index-url https://download.pytorch.org/whl/cu126 torch torchvision
```

```bash
pip install -r requirements.txt
```

`torch`를 `requirements.txt`에 넣지 않은 이유는 일반 PyPI에서 설치하면 **CPU 빌드가 조용히 깔리기 때문**이다.
실제로 이 리포에서 한 번 발생했고, `torch.cuda.is_available()`이 `False`인 채로 Day 1을 시작할 뻔했다.

### DAY 1

| 시간 | 작업 | 산출물 |
|---|---|---|
| 09:00–09:30 | ~~라벨 육안 확인~~ ✅완료 · OpenForensics **COCO→YOLO 변환** + bbox 시각 검수 20장 | YOLO 데이터셋 |
| 09:30–10:30 | **D1**: pretrained zero-shot mAP 평가 (학습 아님, 빠름) | D1 수치 |
| 10:30–13:30 | **D2**: YOLOv8n fine-tuning (~3h, GPU 점유) — 동시에 CPU로 DFDC frame sampling | 체크포인트 + `processed/frames/` |
| 13:30–15:00 | fine-tuned YOLO로 DFDC 얼굴 검출 → face crop (tight) 생성 | `processed/faces/` |
| 15:00–16:00 | D2 mAP 평가 + forged-face recall · DFDC Detection Success Rate + 스팟체크 50건 | D2, D0 |
| 16:00–20:00 | **V0 / V1 학습** (V2는 사전 포기 확정) | 체크포인트 2개 |
| 20:00–22:00 | video-level 평가, bootstrap CI, paired 비교 | `vision_metrics.json` |

### DAY 2

| 시간 | 작업 | 산출물 |
|---|---|---|
| 09:00–10:00 | Fin-Fact 라벨 분포 확인, 클래스 병합 결정 | — |
| 10:00–13:00 | **N1 / N2 학습** (seed 42, N0는 여유 시) | 체크포인트 |
| 13:00–14:00 | Macro-F1, per-class, confusion matrix, 오류 분석 | `nlp_metrics.json` |
| 14:00–16:00 | Synthetic Integration Test Case **4개** 제작 + 통합 pipeline | `inference/pipeline.py` |
| 16:00–18:00 | Streamlit 데모 | `demo/app.py` |
| 18:00–20:00 | 최종 결과표, RQ3 정성 사례, README 갱신, 스크린샷 | `ablation.csv` |

**버퍼 없음이 이 일정의 유일한 약점이다.** 지연 시 포기 순서를 미리 정해둔다:

```
0순위 (이미 발동): V2 (margin ablation) → H1-b 철회. GPU 슬롯을 D2(YOLO 학습)에 배정
1순위 포기: D2 epoch 축소 (15 → 8)     → H3은 유지하되 학습량 절반
2순위 포기: N0 (DistilBERT)            → NLP baseline 비교 철회, N1 vs N2는 유지
3순위 포기: Streamlit 데모             → CLI 출력 + 스크린샷으로 대체
절대 포기 불가: family split · video-level 평가 · bootstrap CI · V0 vs V1 · D1 vs D2 · N1 vs N2
```

아래 6개는 빼면 **결과 자체가 무의미해지므로** 어떤 상황에서도 유지한다.

> **"절대 포기 불가" 중 family split과 video-level 평가는 DFDC 분류 축(①-a)에 적용된다.**
> OpenForensics 검출 축(①-b)은 이미지 데이터셋이라 두 개념이 정의되지 않으며(§5.6),
> 거기서는 이미지 단위 평가 + 저자 공식 split 존중이 대응 규칙이다. 이 구분을 명시하지 않으면
> "검출 축에서 video-level 평가를 안 했다"는 자기모순 지적이 가능해진다.

---

## 12. 완료 기준

### 필수

- [ ] **사건 코딩 15건 + Pain Point Map** (§1.1.1) — 48시간 시계 시작 전
- [ ] family 단위 split 적용 및 `split_report.json` 생성
- [ ] 영상 1개 입력 → 얼굴 bbox · deepfake 확률 · transcript · claim verification label 전부 출력
- [ ] V0 vs V1 **video-level paired AUROC + CI** 산출 ← H1
- [ ] OpenForensics COCO→YOLO 변환 + **bbox 시각 검수 20장**
- [ ] D1 vs D2 **mAP@50 + forged-face recall** 산출 ← H3
- [ ] N1 vs N2 **Macro-F1 + bootstrap CI** 산출 ← H2
- [ ] Detection Success Rate + 수동 스팟체크 기록 (DFDC 축)
- [ ] **identity 누수 검증 불가**(OpenForensics)를 한계로 명시
- [ ] §6.4 결과표의 **필수** 행 전부 채움 (V2는 사전 포기, N0은 부가)
- [ ] Synthetic Integration Test Case 4개로 risk matrix 네 칸 시연 (§5.4)
- [ ] 한계(§9)를 README와 발표자료에 명시

### Nice to Have

- [ ] Grad-CAM으로 조작 영역 시각화
- [ ] V2 margin ablation
- [ ] FakeAVCeleb external test
- [ ] 다중 얼굴 처리
- [ ] STT WER 측정

---

## 13. 기술 스택

```
Python · PyTorch · torchvision · Ultralytics(YOLOv8-Face) · OpenCV · FFmpeg
HuggingFace Transformers · DeBERTa-v3-small · Whisper · scikit-learn · Streamlit
```

---

## 14. 제목 후보

- **기본** — FinDeepfake-48h: Object Detection and Evidence-Based NLP for Financial Deepfake Risk Screening
- **포트폴리오형** — Multimodal Financial Deepfake Screening with Face ROI Detection and Evidence-Based Claim Verification
- **논문형** — *Does Face-Level ROI Improve Deepfake Detection, and Does Evidence Improve Financial Claim Verification?* — A Lightweight Two-Track Study

---

## References

- Meta DFDC — https://ai.meta.com/datasets/dfdc/
- DFDC 미러(취득처) — https://huggingface.co/datasets/gonnerthetooner/DFDC-extracted-full
- DFDC metadata — https://huggingface.co/datasets/scarlettss/dfdc_metadata
- **OpenForensics** (ICCV 2021) — Zenodo DOI 10.5281/zenodo.5528418 · https://sites.google.com/view/ltnghia/research/openforensics · https://arxiv.org/abs/2107.14480
- WIDER FACE — https://shuoyang1213.me/WIDERFACE/
- YOLOv8-Face (pretrained) — https://github.com/lindevs/yolov8-face
- Fin-Fact — https://github.com/IIT-DM/Fin-Fact · https://huggingface.co/datasets/amanrangapur/Fin-Fact
- FakeAVCeleb — https://github.com/DASH-Lab/FakeAVCeleb
- Whisper — https://github.com/openai/whisper
- FINRA, *AI and Investment Fraud* — https://www.finra.org/investors/insights/artificial-intelligence-and-investment-fraud
- EU AI Act, deepfake 정의 및 라벨링 — https://digital-strategy.ec.europa.eu/en/policies/eu-icons-labelling-ai-generated-content

---

## License / Usage

각 데이터셋과 pretrained 가중치는 원 저작자의 라이선스를 따른다.
DFDC·FakeAVCeleb 등 실제 인물의 얼굴이 포함된 데이터는 제출·배포 전에 이용 조건을 다시 확인한다.
데모용으로 제작한 합성 영상은 프로젝트 시연 목적으로만 사용하고 외부 배포하지 않는다.

**HuggingFace 미러 사용에 관한 주의** (§5.1)

- 미러는 **비공식 재배포**이며 라이선스 필드가 비어 있다. 원본 DFDC는 Meta의 DFDC EULA 하에 배포된다.
  **연구·학습 목적으로 사용하는 것과 결과물을 배포하는 것은 다르다.** 출처는 항상 Meta DFDC로 표기한다.
- **미러는 예고 없이 내려갈 수 있다.** Day 1 첫 시간에 받아두는 것이 유일한 방어다.
- **얼굴이 식별 가능한 프레임·크롭을 리포지토리에 커밋하거나 공개 데모에 노출하지 않는다.**
  DFDC는 동의한 유급 배우로 구성되지만, 그것이 재배포 권한을 의미하지는 않는다.
  (`.gitignore`가 `data/model/` 전체를 제외하는 이유가 이것이다.)

**OpenForensics 라이선스 상충** (§5.6): Zenodo 메타데이터는 CC-BY-4.0, 프로젝트 페이지는
CC-BY-NC-SA-4.0, GitHub는 "academic purpose only"로 **세 출처의 표기가 서로 다르다.**
가장 제한적인 해석(비영리·학술 목적, 재배포 금지)을 채택한다. 원본이 Google Open Images의
실존 인물 사진이므로 얼굴 이미지 커밋 금지 규정이 동일하게 적용된다.
