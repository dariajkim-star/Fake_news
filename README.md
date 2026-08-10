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
    │  Object Detection      │                                        │  NLP (BERT / DeBERTa, 영어)    │
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
| Text NLP | `bert-base-uncased` (주 실험, 영어 Fakeddit 기준; 여유 시 DeBERTa 비교) — KLUE-BERT/KoELECTRA는 자체 한국어 금융 데모 전용 | 텍스트 인코딩 → (Epic 3) NER + Event/Relation Extraction | Accuracy/F1, NER F1 |
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

`✅` = Story 1.1~1.4에서 **실제 구현 완료**된 것만 표기한다. 나머지는 해당 Epic에서 추가된다.

```
Fake_news/
├── README.md
├── requirements.txt          ✅
├── configs/                  ✅ 실험 설정 (yaml) — `_base_` 상속 구조
│   ├── base.yaml             ✅ 루트 (seed/device/output_dir/train/logging)
│   ├── fakeddit.yaml         ✅ 데이터 파이프라인 + 금융 키워드
│   ├── bert_only.yaml        ✅ BERT only baseline
│   ├── resnet_only.yaml      ✅ ResNet only baseline
│   └── late_fusion.yaml      ✅ BERT+ResNet late fusion
├── data/                     Fakeddit 원본/전처리 (git 제외, 규약은 docs/DATA.md)
├── src/
│   ├── data/                 ✅ 라벨 매핑·전처리·분할·금융 subset·FakedditDataset·데이터셋 레지스트리
│   ├── vision/               YOLOv8 detection + visual entity recognition (Epic 2·4) — 미구현
│   ├── text/                 NER / event extraction (Epic 3) — 미구현
│   ├── matching/             Cross-modal entity matching (Epic 4) — 미구현
│   ├── fusion/               ✅ 모델 레지스트리 + 인코더(TextEncoder/ImageEncoder) +
│   │                            single-modal · late fusion 분류기
│   │                            ※ cross-attention classifier는 Epic 2에서 추가 (미구현)
│   ├── training/             ✅ 공통 Trainer (모든 Phase 공유)
│   ├── evaluation/           ✅ 공통 지표 (Accuracy/P/R/F1/AUROC) + ablation 표 생성
│   └── utils/                ✅ config(_base_ 상속·--set), seed, 로깅(CSV/TensorBoard)
├── scripts/                  ✅ train/evaluate/predict, download/preprocess/filter_financial, collect_ablation
├── tests/                    ✅ pytest
├── outputs/                  ✅ 실험별 결과 (config 사본·best.pt·history.csv·metrics.json)
└── demo/                     Gradio 데모 (Epic 5) — 미구현
```

모델·데이터셋은 레지스트리에 등록되어 config `model.name`/`data.name`으로 선택된다
(현재 모델: `dummy`, `text_only`, `image_only`, `late_fusion`). 모든 모델은
`forward(batch: dict) -> logits [B,2]`, 라벨 키 `label` 계약을 지키므로 Trainer 수정 없이 교체된다
— 상세는 [`docs/bmad/architecture.md`](docs/bmad/architecture.md) "Architecture Contracts" 참조.

---

## 7. 설치 및 실행

### 설치 (Python 3.10+)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Phase 1 스캐폴딩 실행에는 `torch`, `scikit-learn`, `PyYAML`, `pytest`만 있으면 충분하다.
`ultralytics` / `paddleocr` / `insightface` 등은 Epic 2 이후에 필요하다.

### 학습 · 평가

실험 하나는 config 파일 하나로 정의된다. `--set`으로 임시 오버라이드할 수 있고,
실행 결과는 `outputs/<exp_name>/`에 **config 사본 + checkpoint + metrics.json + history.csv**로 남는다.

```bash
# 스캐폴딩 smoke run (더미 데이터, CPU)
python scripts/train.py --config configs/base.yaml --set exp_name=smoke train.epochs=2 device=cpu

# 저장된 checkpoint로 test split 평가
python scripts/evaluate.py --config outputs/smoke/config.yaml --split test

# 테스트
python -m pytest tests -q
```

### 데이터 준비 (Fakeddit)

원본 데이터는 라이선스상 repo에 포함되지 않는다. 디렉토리 규약·라벨 매핑·준비 절차는
**[`docs/DATA.md`](docs/DATA.md)** 참조.

```bash
python scripts/download_fakeddit.py --root data/fakeddit --check   # 배치 안내/점검
python scripts/preprocess_fakeddit.py --root data/fakeddit         # 클린 manifest + 고정 분할
python scripts/filter_financial.py --config configs/fakeddit.yaml  # 금융 subset (FR11)
python scripts/train.py --config configs/fakeddit.yaml
```

> 라벨 규약: 내부는 **0=REAL, 1=FAKE**이고 Fakeddit 원본 `2_way_label`은 그 **반대**(1=real)다.
> 변환은 반드시 `src/data/labels.py`의 `map_fakeddit_2way_label()`을 거친다.

`configs/base.yaml`을 `_base_`로 상속해 Phase별 config를 만든다 (모델·데이터만 교체 → ablation 성립).

### Phase 1 baseline 3종 (ablation table, FR10)

```bash
python scripts/train.py --config configs/bert_only.yaml      # BERT only
python scripts/train.py --config configs/resnet_only.yaml    # ResNet only
python scripts/train.py --config configs/late_fusion.yaml    # BERT+Image late fusion (인코더 frozen)
# (b) end-to-end fine-tuning
python scripts/train.py --config configs/late_fusion.yaml \
    --set model.freeze_encoders=false exp_name=late_fusion_e2e train.lr=0.00002

# 세 실험의 test 지표를 한 표로 취합 → outputs/ablation_table.csv + docs/bmad/results/ablation.md
python scripts/collect_ablation.py
```

> 오프라인/CI에서는 `--set model.text.pretrained=false model.image.pretrained=false`로
> 랜덤 초기화 경로를 쓴다 (네트워크 접근 0회).

### 단일 샘플 추론

```bash
python scripts/predict.py --config outputs/late_fusion/config.yaml \
    --image data/fakeddit/images/abc.jpg --text "tesla stock soars after earnings"
# 출력: {"fake_prob": 0.87, "label": 1, "label_name": "FAKE", "elapsed_sec": 0.42}
```

### 근거(mismatch) 포함 추론 — Epic 4 완료 후 (계획)

Epic 4에서 entity consistency가 붙으면 동일한 `scripts/predict.py`가 mismatch 근거를 함께 출력한다.

```bash
python scripts/predict.py --config outputs/phase4/config.yaml \
    --image sample.jpg --text "삼성전자, NVIDIA와 20조 계약..."
# (계획) 출력: fake_prob=0.93, mismatches=[("Jensen Huang","Lisa Su"), ("NVIDIA","AMD")]
```

### 단계별 개발 로드맵

| Phase | Epic | 내용 | 상태 |
|---|---|---|---|
| 1 | Epic 1 | Baseline: ResNet(이미지) + BERT(텍스트) → late fusion (Fakeddit) + 공용 Trainer/Config/평가 | 구현 완료(실데이터 학습 미실행) |
| 2 | Epic 2 | + YOLO object regions → Region Encoder → Cross Attention | 예정 |
| 3 | Epic 3 | + NER / Entity Extraction (텍스트) | 예정 |
| 4 | Epic 4 | + Visual–Textual entity consistency score → 최종 모델 | 예정 |
| 마무리 | Epic 5 | 전체 ablation 취합, 금융 셋 최종 평가, Gradio 데모, 리포트 | 예정 |
| (병행, 3~7주차) | Epic 6 | FinFact-Eval — 자체 금융 held-out 평가셋 구축 (Epic 5 착수 전 완료) | 예정 |

(Phase↔Epic 대응 상세는 [`docs/DEV_PLAN.md` §0](docs/DEV_PLAN.md) 참조.)

핵심 실험: **Ablation table** (BERT only / ResNet only / BERT+Image / +Object Detection / +Entity consistency) — object-level 증거와 entity 불일치 정보가 F1을 추가로 개선함을 보이는 것이 발표 포인트.

---

## 8. 관련 연구

- **EM-FEND** — Visual/Textual Entity Inconsistency 기반 fake news detection ([arXiv:2108.10509](https://arxiv.org/abs/2108.10509))
- **CFFN** — Word-region consistency 기반 multimodal detection ([arXiv:2311.01807](https://arxiv.org/abs/2311.01807))
- **Event-Radar** — Event-level graph 기반 multimodal fake news detection (ACL 2024)
- **Fakeddit** — 대규모 multimodal fake news 데이터셋 ([arXiv:1911.03854](https://arxiv.org/abs/1911.03854))
- PROPOR 2026 — Financial fake news multimodal detection framework

---

## 9. Change Log

| Date | Version | Description | Author |
|---|---|---|---|
| 2026-08-10 | 1.1 | 프로젝트 구조를 Story 1.3/1.4 산출물까지 반영해 갱신(configs 5종 명시, `src/fusion/`의 "cross-attention classifier 완료" 허위 표기 제거 — 실제로는 미구현), 추론 명령을 `scripts/infer.py` → `scripts/predict.py`로 정정, 텍스트 backbone을 영어 주 실험 + 한국어 데모 전용으로 통일, 로드맵에 Phase↔Epic 매핑·상태 열 추가 | Winston (Architect) |
