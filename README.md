# FinFact

**Object-Level Multimodal Financial Misinformation Detection**
— 뉴스 이미지 속 시각적 증거와 기사 텍스트의 주장을 entity 단위로 대조하여 금융 가짜뉴스를 탐지하는 딥러닝 프로젝트.

---

## 1. 문제 정의

### 왜 금융 가짜뉴스인가?
- M&A 루머, 실적 조작, CEO 관련 허위 보도, 주가조작성 SNS 게시물 등 금융 가짜뉴스는 **실제 금전적 피해(주가 급등락, 개인 투자자 손실)** 로 직결된다.
- 금융 뉴스는 기업명, 인물, 금액, 날짜 같은 **entity 중심의 사실 주장**으로 구성되어 있어, entity 단위 검증이 특히 효과적인 도메인이다.

### 왜 object-level 접근인가?
기존 multimodal 탐지 모델은 이미지 전체와 문장 전체를 CLIP similarity 등으로 **한 번에** 비교한다. 그러나 가짜뉴스의 조작은 대개 국소적이다 — 예: "NVIDIA 계약 체결" 기사에 AMD 행사 사진을 붙이는 식.

FinFact는:
- **Object Detection**으로 이미지 속 증거를 객체 단위(PERSON, LOGO, CHART, ...)로 쪼개고,
- **NER / Event Extraction**으로 텍스트 속 주장을 entity 단위로 뽑아,
- 이를 **하나씩(word-region pair) 대조**하여 어떤 entity가 불일치(MISMATCH)인지 근거까지 제시한다.

즉, 단순 Fake/Real 분류를 넘어 **설명 가능한(explainable)** 탐지를 목표로 한다.

---

## 2. 전체 파이프라인

```
                ┌──────────────── INPUT: 뉴스 게시물 (이미지 + 텍스트) ────────────────┐
                │                                                                      │
        ┌───────▼────────┐                                              ┌──────────────▼─────────────┐
        │     IMAGE      │                                              │           TEXT             │
        └───────┬────────┘                                              └──────────────┬─────────────┘
                │                                                                      │
    ┌───────────▼────────────┐                                        ┌────────────────▼───────────────┐
    │  Object Detection      │                                        │  NLP (KLUE-BERT / DeBERTa)     │
    │  (YOLOv8 / DETR)       │                                        │  - NER: PERSON, ORG, PRODUCT,  │
    │  PERSON / LOGO /       │                                        │    LOCATION, DATE, MONEY       │
    │  PRODUCT / CHART /     │                                        │  - Event/Relation Extraction   │
    │  DOCUMENT / TEXT_REGION│                                        │    (예: 삼성전자 —CONTRACT_WITH→ │
    └───────────┬────────────┘                                        │     NVIDIA, AMOUNT ₩20조)      │
                │ bbox crop                                           └────────────────┬───────────────┘
    ┌───────────▼────────────┐                                                         │
    │ Visual Entity          │                                                         │
    │ Recognition            │                                                         │
    │  PERSON → 얼굴/인물 인식 │                                                        │
    │  LOGO   → CLIP/로고분류 │                                                         │
    │  CHART/TEXT → OCR      │                                                         │
    └───────────┬────────────┘                                                         │
                │        visual entities                     textual entities          │
                └───────────────────────┐             ┌────────────────────────────────┘
                                ┌───────▼─────────────▼────────┐
                                │   Cross-modal Matching       │
                                │   visual ↔ textual entity    │
                                │   MATCH / MISMATCH / UNKNOWN │
                                └──────────────┬───────────────┘
                                ┌──────────────▼───────────────┐
                                │  Cross-modal Attention Fusion│
                                │  - Entity consistency        │
                                │  - Event consistency         │
                                │  - Image/Text similarity     │
                                └──────────────┬───────────────┘
                                ┌──────────────▼───────────────┐
                                │  OUTPUT: Fake probability    │
                                │  + 근거 (mismatch entity 목록) │
                                └──────────────────────────────┘
```

---

## 3. 딥러닝 사용 지점

| 컴포넌트 | 모델 | 역할 | 평가지표 |
|---|---|---|---|
| Object Detection | YOLOv8 / DETR (CNN·Transformer 검출기) | 이미지에서 PERSON, LOGO, PRODUCT, CHART, DOCUMENT, TEXT_REGION bbox 검출 | mAP@50, Precision, Recall |
| Visual Entity Recognition | CLIP embedding, 얼굴/로고 분류기, 딥러닝 기반 OCR | bbox crop을 실제 entity로 식별 (예: "Jensen Huang", "NVIDIA"), 차트/문서에서 수치·날짜 추출 | Top-1 Accuracy, OCR CER |
| Text NLP | KoELECTRA / KLUE-BERT / DeBERTa fine-tuning | NER + Event/Relation Extraction으로 텍스트 주장 구조화 | NER F1 |
| Cross-modal Fusion | Cross-modal Attention (region feature × token feature) | entity/event consistency 통합 → 최종 Fake/Real 분류 | Accuracy, Precision, Recall, F1, AUROC |

---

## 4. 예시 시나리오

> **기사 텍스트**: "삼성전자, NVIDIA와 20조 원 규모 AI 칩 공급 계약 체결 — Jensen Huang CEO 서명식 참석"
>
> **첨부 이미지**: AMD 행사장에서 Lisa Su가 발표하는 사진

| 단계 | 결과 |
|---|---|
| Object Detection | PERSON bbox 1개, LOGO bbox 1개 검출 |
| Visual Entity Recognition | PERSON → "Lisa Su", LOGO → "AMD" |
| Text NER / Event | ORG: 삼성전자, NVIDIA / PERSON: Jensen Huang / MONEY: ₩20조 / Event: CONTRACT_WITH |
| Cross-modal Matching | "Jensen Huang" vs "Lisa Su" → **MISMATCH**, "NVIDIA" vs "AMD" → **MISMATCH** |
| 최종 출력 | Fake probability 0.93 + 근거: "이미지 속 인물·로고가 기사 주장 entity와 불일치" |

---

## 5. 데이터셋

- **Fakeddit** ([arXiv:1911.03854](https://arxiv.org/abs/1911.03854)) — 100만+ text-image 쌍, 2/3/6-way label. Phase 1 baseline 및 fusion 모델 학습의 주 데이터셋.
- 금융 도메인 특화 평가를 위해 금융 뉴스 subset 필터링 및 소규모 자체 수집 데이터(금융 뉴스 + mismatch 이미지 합성) 활용 계획.

---

## 6. 프로젝트 구조 (예상)

```
Fake_news/
├── README.md
├── requirements.txt
├── configs/                  # 학습/모델 설정 (yaml)
├── data/
│   ├── raw/                  # Fakeddit 원본
│   └── processed/            # 전처리 결과
├── src/
│   ├── detection/            # YOLOv8/DETR object detection
│   ├── visual_entity/        # CLIP/얼굴/로고 인식, OCR
│   ├── nlp/                  # NER, Event Extraction (BERT 계열)
│   ├── matching/             # Cross-modal entity matching
│   ├── fusion/               # Cross-modal attention + classifier
│   ├── train.py
│   └── infer.py
├── notebooks/                # 실험/EDA
├── experiments/              # ablation 결과, checkpoint
└── demo/                     # Streamlit/Gradio 데모
```

---

## 7. 설치 및 실행 (계획된 인터페이스)

> 아직 코드 구현 전이며, 아래는 계획된 CLI 인터페이스입니다.

```bash
# 환경 (Python 3.10+)
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt   # torch, transformers, ultralytics, timm, easyocr 등

# 데이터 준비
python src/data/prepare_fakeddit.py --root data/raw --out data/processed

# 학습 (Phase별)
python src/train.py --config configs/phase1_baseline.yaml   # ResNet + BERT late fusion
python src/train.py --config configs/phase2_regions.yaml    # + YOLO regions + Cross Attention
python src/train.py --config configs/phase4_full.yaml       # + Entity consistency (최종)

# 추론
python src/infer.py --image sample.jpg --text "삼성전자, NVIDIA와 20조 계약..." \
    --checkpoint experiments/phase4/best.pt
# 출력: fake_prob=0.93, mismatches=[("Jensen Huang","Lisa Su"), ("NVIDIA","AMD")]

# 데모
streamlit run demo/app.py
```

### 단계별 개발 로드맵
| Phase | 내용 |
|---|---|
| 1 | Baseline: ResNet(이미지) + BERT(텍스트) → late fusion (Fakeddit) |
| 2 | + YOLO object regions → Region Encoder → Cross Attention |
| 3 | + NER / Entity Extraction (텍스트) |
| 4 | + Visual–Textual entity consistency score → 최종 모델 |

핵심 실험: **Ablation table** (BERT only / ResNet only / BERT+Image / +Object Detection / +Entity consistency) — object-level 증거와 entity 불일치 정보가 F1을 추가로 개선함을 보이는 것이 발표 포인트.

---

## 8. 관련 연구

- **EM-FEND** — Visual/Textual Entity Inconsistency 기반 fake news detection ([arXiv:2108.10509](https://arxiv.org/abs/2108.10509))
- **CFFN** — Word-region consistency 기반 multimodal detection ([arXiv:2311.01807](https://arxiv.org/abs/2311.01807))
- **Event-Radar** — Event-level graph 기반 multimodal fake news detection (ACL 2024)
- **Fakeddit** — 대규모 multimodal fake news 데이터셋 ([arXiv:1911.03854](https://arxiv.org/abs/1911.03854))
- PROPOR 2026 — Financial fake news multimodal detection framework
