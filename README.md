# FinDeepfake-48h

**Face-Level Object Detection + Financial Claim NLP for Deepfake-Enabled Financial Disinformation**

48시간 안에 완주하는 것을 설계 제약으로 삼은 멀티모달 딥러닝 프로젝트다.
영상이 **AI로 조작되었는지**와, 영상 속 **금융 주장이 신뢰할 만한지**를 각각 판정한다.

---

## 1. 문제 정의

### 1.1 출발점 — 기술이 아니라 실제 피해 사례

이 프로젝트는 "딥페이크가 위험할 것 같다"는 직감에서 출발하지 않는다.
**크롤링으로 실제 피해 사례를 먼저 확보하고, 그 피해가 왜 발생했는지를 기술적으로 분해**해서 문제를 정의한다.

수집 채널: 언론 discovery(Google News·NAVER) + 감독기관 사실확인(KRX·금융위·금감원·SEC·FINRA·CFTC·FCA) — [README_CRAWLERS.md](README_CRAWLERS.md) 참조.
이 작업은 48시간 시계 **밖**(D-1 이전)에 완료한다.

수집된 사건은 다음 구조로 코딩한다.

| 사건 | 허위정보 형태 | 투자자가 믿은 것 | 행동 | 피해 |
|---|---|---|---|---|
| Case A | CEO deepfake | CEO의 실제 투자권유 | 송금 | 금전손실 |
| Case B | 유명인 deepfake | 검증된 투자상품 | 가입/송금 | 투자금 손실 |
| Case C | 조작 기업발표 | 기업의 실제 발표 | 주식 매수 | 가격 하락 손실 |
| Case D | AI 영상 + 가짜 사이트 | 공식 금융서비스 | 자금이체 | 사기피해 |

여기서 공통 pain point 두 개가 나온다.

```
Pain Point ①  누가 실제로 말했는지 검증하기 어렵다   → Media Authenticity 문제
Pain Point ②  발언 내용이 사실인지 판단하기 어렵다   → Financial Claim Verification 문제
```

**문제정의 문장:**

> 딥페이크를 활용한 투자사기와 허위 금융정보 유포로, 투자자는 온라인 영상에서
> **실제 인물이 실제로 발언한 것인지**와 **발언 내용이 사실에 근거하는지**를
> 동시에 확인해야 하는 부담을 지게 된다. 본 프로젝트는 실제 피해 사례 분석에서
> 이 문제를 ① 영상 진위 판별과 ② 금융 주장 사실검증의 두 기술 문제로 정의하고,
> Object Detection 기반 Deepfake Detection과 NLP 기반 Financial Claim Verification을
> 결합한 투자정보 검증 PoC를 구현한다.

즉 문제는 "deepfake 영상을 탐지한다"가 아니라
**"투자자의 판단 이전에 영상의 진위와 금융 주장 신뢰도를 동시에 확인할 수단이 부족하다"**이다.

> 📝 크롤링 완료 후 이 절의 일반론을 실측치로 치환한다 (숫자 없이 주장만 쓰는 것을 막는 템플릿):
> `수집 사례 N건 중 X%가 유명인/CEO 사칭, Y%가 투자금 송금 유도, 확인된 피해액 합계 Z` — **채우기 전까지 발표자료에 쓰지 않는다.**

**전체 논리 연결 (발표 한 장):**

```
[크롤링] 실제 금융 deepfake 피해 사례
    ↓
[Pain Point] ① 실제 발언인지 알기 어렵다  ② 발언이 사실인지 알기 어렵다
    ↓
[기술 문제] ① Media Authenticity Detection  ② Financial Claim Verification
    ↓
[모델] YOLO + EfficientNet  /  Whisper + DeBERTa
    ↓
[산출물] Deepfake Probability + Claim Verification Result (분리 제시)
    ↓
[검증] H1: Face ROI가 탐지를 개선하는가?  H2: Evidence가 검증을 개선하는가?
```

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

구체적 시나리오 — SNS에 이런 영상이 돈다:

> **Jamie Dimon (JPMorgan CEO)**: "저희는 이번 분기 100억 달러의 예상치 못한 손실을 기록했습니다."

영상은 AI 합성이고, 발언 내용도 사실이 아니다. 주가는 이미 움직인다.
이건 가상의 위협이 아니라 FINRA와 SEC가 투자자·기업에 실제로 경고한 사기 유형이다.

### 1.3 왜 두 판정을 분리하는가

시스템은 두 질문에 **따로** 답한다.

```
Q1. 이 영상은 AI로 조작되었는가?          → Media Authenticity
Q2. 영상 속 금융 주장은 신뢰할 만한가?      → Claim Credibility
```

합치지 않는 이유는 명확하다. **조작된 영상이 참말을 할 수 있고, 진짜 영상이 거짓말을 할 수 있다.**
두 신호를 하나의 "Fake News Probability"로 뭉개면 정보가 사라지고, 무엇보다 그렇게 학습시킬 근거가 없다
(→ §8.3). 대신 두 축을 그대로 보여주고 조합만 해석한다.

### 1.4 가설과 연구 질문

두 기술 문제 각각에 검증 가능한 가설을 하나씩 건다.

| # | 가설/질문 | 평가 방식 |
|---|---|---|
| **H1** | 투자 영상에서 얼굴 영역을 Object Detection으로 추출하면 full-frame 대비 deepfake 탐지 성능이 개선된다 | 정량 (V0 vs V1, paired ΔAUROC) |
| **H1-b** | ROI margin — 배경 문맥은 도움인가 방해인가? | 정량 (V1 vs V2, paired ΔAUROC) |
| **H2** | claim만 사용하는 것보다 evidence를 함께 사용하면 금융 claim 검증 성능이 개선된다 | 정량 (N1 vs N2, Macro-F1) |
| **RQ3** | 조작 여부와 주장 신뢰도를 분리 제시하는 것이 단일 라벨보다 설명 가능한 위험 신호를 주는가? | 정성 (사례 분석) |

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
      ┌───────▼────────┐                      │
      │ Face Detection │  ← Object Detection  │
      │  YOLOv8n-Face  │                      │
      └───────┬────────┘                Whisper (STT)
              │ bbox + margin                 │
         Face Crop 224²                  Transcript
              │                               │
      ┌───────▼────────┐              금융 문장 필터
      │   Deepfake     │                      │
      │ EfficientNet-B0│              ┌───────▼────────┐
      └───────┬────────┘              │ Claim Classifier│  ← NLP
              │                       │ DeBERTa-v3-small│
     frame prob → median              └───────┬────────┘
              │                               │
      Video-level P(fake)              Claim label + conf
              └───────────────┬───────────────┘
                              │
                      RISK MATRIX (§8.3)
```

**딥러닝 사용 지점 4곳**: Object Detection(YOLO) · 이미지 분류(EfficientNet) · 음성인식(Whisper) · **NLP 텍스트 분류(DeBERTa)**.
이 중 NLP는 선택이 아니라 필수 축이며, 실제로 fine-tuning하고 별도 ablation으로 평가한다.

각 모듈의 역할을 정확히 정의한다. 과장하지 않는 것이 이 프로젝트의 원칙이다.

- **Object Detection**: 전체 화면이 아니라 **발언자의 얼굴 영역을 식별해 forensic analysis 대상으로 제한**한다.
- **NLP**: "가짜뉴스를 맞히는" 모델이 아니다. **금융 claim과 evidence 간의 entailment/contradiction을 판별**한다.

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
    "claim_credibility": "LOW",
    "combined": "HIGH RISK"
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
  Claim Reliability  : HIGH RISK
```

### 4.3 결과 파일

```
results/
├── vision_metrics.json        # video-level, bootstrap CI 포함
├── nlp_metrics.json           # 3-seed mean ± std
├── ablation.csv               # 최종 비교표
├── confusion_matrix_*.png
└── qualitative_cases.md       # RQ3용 사례 분석
```

---

## 5. 데이터셋

### 5.1 DFDC Sample — Deepfake Vision

**출처**
- Meta: https://ai.meta.com/datasets/dfdc/
- Kaggle: https://www.kaggle.com/competitions/deepfake-detection-challenge/data

전체 DFDC는 10만 개 이상이라 손댈 수 없다. Kaggle이 제공하는 **sample training set**만 쓴다.

**metadata 구조**

```json
{ "abc.mp4": { "label": "FAKE", "original": "xyz.mp4", "split": "train" } }
```

**내부 라벨 규약**

```
REAL = 0
FAKE = 1
```

> ⚠️ **Day 1 첫 30분에 반드시 눈으로 확인할 것**: 실제 배포본의 라벨 문자열과 클래스 비율.
> DFDC sample은 FAKE가 압도적으로 많다. 이 비율을 모르면 §6.2의 지표 선택이 통째로 틀어진다.

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

### 5.2 Fin-Fact — Financial NLP

**출처**
- GitHub: https://github.com/IIT-DM/Fin-Fact
- HuggingFace: https://huggingface.co/datasets/amanrangapur/Fin-Fact

```python
from datasets import load_dataset
ds = load_dataset("amanrangapur/Fin-Fact")
```

**사용 컬럼**: `Claim`, `Claim Label`, `Evidence`, `Justification`

Evidence는 옵션이 아니라 **H2의 핵심 변수**다 (§7.2).

> ⚠️ Day 2 첫 작업 두 가지: ① 라벨 분포와 클래스별 표본 수 확인 —
> 최소 클래스가 50 미만이면 Macro-F1이 불안정해지므로 클래스 병합 여부를 그 자리에서 결정.
> ② **evidence 토큰 길이 분포 측정** (§7.2 전제 게이트).

### 5.3 얼굴 검출기 — pretrained, 학습 안 함

- WIDER FACE 벤치마크: https://shuoyang1213.me/WIDERFACE/
- pretrained 구현: https://github.com/lindevs/yolov8-face

**직접 학습하지 않는다.** 이 프로젝트에서 detector의 역할은 성능 경쟁이 아니라 **ROI 공급**이다.
detector를 학습 대상으로 삼는 순간 bbox 라벨링이 필요해지고, 그건 이틀 예산을 초과한다.

### 5.4 데모 영상 — 직접 만든다 (필수)

**DFDC 영상 속 사람들은 금융 얘기를 하지 않는다.** 일반인이 아무 말이나 하는 영상이다.
따라서 DFDC로 통합 데모를 돌리면 NLP 칸이 **항상 비어 있다.**

데모용으로 금융 발언이 담긴 영상 10~15개를 별도로 준비한다 (TTS 합성 또는 직접 녹음, Fin-Fact claim 문장 낭독).
이건 학습 데이터가 아니라 **파이프라인 시연 및 RQ3 정성 분석용**이며, README와 발표에서 그렇게 명시한다.

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
DFDC sample은 FAKE 비율이 높다. 전부 FAKE로 찍어도 Accuracy가 높게 나온다.
→ **주 지표는 AUROC와 Average Precision(PR-AUC)**, Accuracy는 참고용으로만 병기한다.

**함정 3 — 표본이 작아서 작은 차이는 노이즈다.**
family 단위로 자르고 나면 테스트 영상은 100개 미만일 가능성이 크다.
이 규모에서 **AUROC 0.02 차이는 아무 의미가 없다.**
→ 모든 수치에 **bootstrap 95% CI(2000회 resampling)**를 붙이고,
모델 A vs B는 **같은 테스트 영상에 대한 paired bootstrap**으로 비교한다.
CI가 0을 포함하면 "개선 없음"이라고 쓴다. 이건 사후에 정하지 않고 지금 못 박는다.

### 6.2 지표 정의

| 모듈 | 주 지표 | 보조 | 단위 | 불확실성 |
|---|---|---|---|---|
| Deepfake | **AUROC** | AP, F1, FAKE Recall, Accuracy | **video** | bootstrap 95% CI |
| Deepfake A vs B | **ΔAUROC** | — | video (paired) | paired bootstrap CI |
| NLP | **Macro-F1** | per-class P/R/F1, Accuracy | claim | 3-seed mean ± std |
| Face Detection | **Detection Success Rate** | 수동 스팟체크 정확도 | frame | — |
| STT | WER (선택) | — | 샘플 | 필수 아님 |

**FAKE Recall을 반드시 별도 보고한다.** 금융 사기 스크리닝에서 놓친 deepfake의 비용이
잘못 경보한 진짜 영상의 비용보다 훨씬 크다. 운영 임계값은 FAKE Recall 기준으로 잡는다.

### 6.3 Object Detection 지표에 대한 정정

**mAP@50은 이 프로젝트에서 계산할 수 없다.**

mAP는 GT bounding box가 있어야 계산된다. DFDC에는 얼굴 bbox 정답이 없다.
WIDER FACE 벤치마크 수치를 인용하는 건 가능하지만, 그건 **우리 데이터에서의 성능이 아니다.**

대신 실제로 측정 가능한 두 가지를 보고한다.

```
Detection Success Rate = 얼굴 ROI를 정상 추출한 영상 수 / 전체 영상 수
Manual Spot Check      = 무작위 50 프레임의 bbox를 눈으로 검수, 오검출·미검출 건수 기록
```

이건 축소가 아니라 정정이다. **채울 수 없는 칸을 발표 자료에 넣는 것이 가장 큰 리스크다.**

### 6.4 최종 결과표 (이 표를 채우는 것이 프로젝트 완료 조건)

| # | 모듈 | 구성 | 주 지표 | 값 | 95% CI |
|---|---|---|---|---|---|
| V0 | Vision | Full frame → EfficientNet-B0 | AUROC | – | – |
| V1 | Vision | YOLO ROI (tight) → EfficientNet-B0 | AUROC | – | – |
| V2 | Vision | YOLO ROI (margin 1.3×) → EfficientNet-B0 | AUROC | – | – |
| — | Vision | **V1 − V0 (paired)** | ΔAUROC | – | – |
| — | Vision | **V2 − V1 (paired)** | ΔAUROC | – | – |
| N0 | NLP | DistilBERT, claim only | Macro-F1 | – | ±std |
| N1 | NLP | DeBERTa-v3-small, claim only | Macro-F1 | – | ±std |
| N2 | NLP | DeBERTa-v3-small, claim + evidence | Macro-F1 | – | ±std |
| D0 | Detection | YOLOv8n-Face | Success Rate | – | – |

---

## 7. Ablation — 무엇을 왜 비교하는가

### 7.1 Vision: H1 / H1-b

```
V0  Full Frame ──────────────────► EfficientNet ──► P(fake)
V1  Frame ─► YOLO ─► tight crop ──► EfficientNet ──► P(fake)
V2  Frame ─► YOLO ─► 1.3× crop ───► EfficientNet ──► P(fake)
```

**V1 − V0**이 Object Detection을 파이프라인에 넣은 이유를 정량적으로 설명한다.
**V2 − V1**은 실제로 답이 알려지지 않은 질문이다 — 조작 흔적은 얼굴 경계(턱선·헤어라인)에 몰려 있어
margin이 도움이 될 수도 있고, 배경 노이즈가 들어와 해로울 수도 있다. 학습 한 번 값으로 답이 나온다.

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
| Batch | 32 | 16 |
| max_len | — | 256 (claim) / 512 (claim+evidence) |
| 기타 | AMP, early stopping | AMP, early stopping, 3 seeds |

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

| Media Authenticity | Claim Credibility | 판정 | 의미 |
|---|---|---|---|
| REAL | Credible | **LOW** | 정상 |
| SYNTHETIC | Credible | **SYNTHETIC MEDIA** | 조작 미디어지만 내용은 사실 (합성 앵커 등) |
| REAL | Suspicious | **MISINFORMATION** | 진짜 사람의 허위·오도 발언 |
| SYNTHETIC | Suspicious | **HIGH RISK** | deepfake-enabled disinformation |

**이 4칸 구조가 RQ3의 답이다.** 단일 확률 하나로는 두 번째·세 번째 칸을 구분할 수 없다.
발표에서는 각 칸의 실제 사례를 데모 영상으로 시연한다.

---

## 9. 이 프로젝트가 주장하지 않는 것

정직한 한계 명시는 감점 요인이 아니라 설계 역량의 증거다.

- **"AI가 금융 뉴스를 자동 팩트체크한다"고 주장하지 않는다.**
  claim classifier는 외부 evidence retrieval을 하지 않는다. 임의의 새로운 발언에 대한
  절대적 진실 판별기가 아니라, Fin-Fact 라벨 패턴을 학습한 분류기다.
- **DFDC sample에서의 성능이 실제 금융 deepfake에 이전된다고 주장하지 않는다.**
  DFDC는 금융 도메인 데이터가 아니다. 도메인 이전은 검증되지 않았고, 이를 한계로 명시한다.
- **얼굴 검출 성능을 기여로 주장하지 않는다.** pretrained 모델을 그대로 쓴다.
- **N1(claim-only) 점수가 높다고 사실 검증 능력의 증거로 해석하지 않는다** (§7.2).

실서비스 수준으로 가려면 이 단계가 추가되어야 한다:

```
Claim → Evidence Retrieval → SEC Filing / IR / Trusted News → NLI Verification
```

---

## 10. 폴더 구조

```
findeepfake-48h/
├── README.md
├── requirements.txt
├── configs/
│   ├── vision_fullframe.yaml
│   ├── vision_roi.yaml
│   └── nlp_claim.yaml
├── data/
│   ├── raw/{dfdc,finfact}/
│   ├── processed/{frames,faces,audio,text}/
│   ├── demo_videos/              # §5.4 직접 제작
│   └── splits/{train,val,test}.csv + split_report.json
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

- [ ] 크롤러 실행 → 사례 수집 ([README_CRAWLERS.md](README_CRAWLERS.md))
- [ ] 사건 코딩 표 + Pain Point Map 작성 → §1.1 실측치 템플릿 채우기

### D-1 (전날 밤, 30~60분) — 이걸 안 하면 이틀이 이틀이 아니다

- [ ] Kaggle 계정 + DFDC 대회 규약 동의 (승인 지연 가능)
- [ ] DFDC sample 다운로드 시작 (백그라운드)
- [ ] `ffmpeg` 설치 확인
- [ ] `pip install -r requirements.txt`, Whisper·YOLOv8n-Face 가중치 사전 다운로드
- [ ] GPU 인식 확인 (`torch.cuda.is_available()`)

### DAY 1

| 시간 | 작업 | 산출물 |
|---|---|---|
| 09:00–10:00 | **라벨 인코딩 육안 확인**, 클래스 비율 측정 | `split_report.json` |
| 10:00–12:00 | family 단위 split, frame sampling | `splits/*.csv` |
| 13:00–15:00 | YOLO 검출 → face crop (tight / 1.3×) 생성 | `processed/faces/` |
| 15:00–16:00 | Detection Success Rate + 수동 스팟체크 50건 | D0 |
| 16:00–20:00 | V0 / V1 / V2 학습 | 체크포인트 3개 |
| 20:00–22:00 | video-level 평가, bootstrap CI, paired 비교 | `vision_metrics.json` |

### DAY 2

| 시간 | 작업 | 산출물 |
|---|---|---|
| 09:00–10:00 | Fin-Fact 라벨 분포 확인, 클래스 병합 결정 | — |
| 10:00–13:00 | N0 / N1 / N2 학습 (각 3 seed) | 체크포인트 |
| 13:00–14:00 | Macro-F1, per-class, confusion matrix, 오류 분석 | `nlp_metrics.json` |
| 14:00–16:00 | 데모 영상 10~15개 제작 + 통합 pipeline | `inference/pipeline.py` |
| 16:00–18:00 | Streamlit 데모 | `demo/app.py` |
| 18:00–20:00 | 최종 결과표, RQ3 정성 사례, README 갱신, 스크린샷 | `ablation.csv` |

**버퍼 없음이 이 일정의 유일한 약점이다.** 지연 시 포기 순서를 미리 정해둔다:

```
1순위 포기: V2 (margin ablation)  → H1-b 철회
2순위 포기: N0 (DistilBERT)       → NLP baseline 비교 철회, N1 vs N2는 유지
3순위 포기: Streamlit 데모        → CLI 출력 + 스크린샷으로 대체
절대 포기 불가: family split · video-level 평가 · bootstrap CI · N1 vs N2
```

아래 4개는 빼면 **결과 자체가 무의미해지므로** 어떤 상황에서도 유지한다.

---

## 12. 완료 기준

### 필수

- [ ] family 단위 split 적용 및 `split_report.json` 생성
- [ ] 영상 1개 입력 → 얼굴 bbox · deepfake 확률 · transcript · claim label 전부 출력
- [ ] V0 vs V1 **video-level paired AUROC + CI** 산출
- [ ] N1 vs N2 **Macro-F1 (3-seed mean±std)** 산출
- [ ] Detection Success Rate + 수동 스팟체크 기록
- [ ] §6.4 최종 결과표 전부 채움
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

- **기본** — FinDeepfake-48h: Object Detection and NLP for Financial Deepfake Risk Screening
- **포트폴리오형** — Multimodal Financial Deepfake Screening with Face Detection and Financial NLP
- **논문형** — *Does Face-Level Object Detection Improve Deepfake Screening, and Can Claim-Only NLP Verify Financial Facts?* — A Lightweight Two-Track Study

---

## References

- Meta DFDC — https://ai.meta.com/datasets/dfdc/
- Kaggle DFDC — https://www.kaggle.com/competitions/deepfake-detection-challenge/data
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
