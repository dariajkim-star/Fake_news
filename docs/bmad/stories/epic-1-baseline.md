# Epic 1: Baseline 파이프라인 — User Stories

> 작성자: Bob (BMAD Scrum Master) · 작성일: 2026-08-10
> 근거 문서: `docs/bmad/prd.md` (Epic 1, FR1, FR10, FR11, NFR1~NFR6)

---

## Story 1.1: 프로젝트 스캐폴딩

### Status

Approved

### Story

As a **ML 엔지니어(개발자)**,
I want **repo 구조, config 시스템, seed 고정, 공통 학습/평가 루프가 갖춰진 프로젝트 스캐폴딩**,
so that **이후 모든 모델(baseline~최종 fusion 모델)을 동일한 실험 프레임 위에서 재현 가능하게 학습·평가할 수 있다**.

### Acceptance Criteria

1. Monorepo 구조가 architecture.md Source Tree와 정합되게 생성된다: `src/` 하위에 `data/`, `fusion/`(baseline 포함), `utils/` + Epic 1용 공통 모듈(`models/`, `training/`, `evaluation/` 또는 architecture 트리 내 상응 위치), 최상위에 `configs/`, `scripts/`, `tests/`, `outputs/`(실험 결과) 디렉토리가 존재한다. 이후 architecture에 정의된 `vision/`, `text/`, `matching/`, `demo/`, `experiments/` 확장이 가능한 구조여야 한다.
2. yaml 기반 config 시스템이 동작한다: `configs/*.yaml` 파일 하나로 실험(모델 종류, hyperparameter, 데이터 경로, seed)을 정의하고, `python scripts/train.py --config configs/xxx.yaml` 형태로 실행할 수 있다.
3. `set_seed(seed)` 유틸이 `random`, `numpy`, `torch`(cuda 포함)의 seed를 고정하며, config의 seed 값이 모든 실행 경로에 적용된다. 동일 config로 2회 실행 시 metric이 재현된다 (cudnn deterministic 설정 포함).
4. 공통 Trainer(학습 루프)가 구현된다: train/val loop, checkpoint 저장(best val metric 기준), early stopping(옵션), 결과 로깅(CSV 또는 TensorBoard 중 config로 선택).
5. 공통 평가 모듈이 Accuracy / Precision / Recall / F1 / AUROC를 계산하여 결과 파일(`outputs/<exp_name>/metrics.json`)로 저장한다.
6. `requirements.txt`(또는 `environment.yml`)에 PyTorch, HuggingFace Transformers, torchvision, scikit-learn 등 의존성이 명시되고, README에 설치·실행 방법이 기술된다.
7. `pytest` 실행 시 최소한 config 로딩, seed 고정, metric 계산에 대한 unit test가 통과한다.

### Tasks / Subtasks

- [ ] repo 디렉토리 구조 및 패키지 초기화 (AC: 1)
  - [ ] `src/` 패키지 생성 (architecture Source Tree 기준), 하위 모듈 `data/`, `fusion/`, `utils/` + 공통 학습/평가 모듈 및 `__init__.py`
  - [ ] `configs/`, `scripts/`, `tests/`, `outputs/`(.gitignore 처리) 생성
- [ ] config 시스템 구현 (AC: 2)
  - [ ] yaml 로더 + dataclass/dict 기반 config 객체 (`src/utils/config.py`)
  - [ ] `scripts/train.py`, `scripts/evaluate.py` 엔트리포인트에 `--config` 인자 연결
  - [ ] 예시 config `configs/base.yaml` 작성 (exp_name, seed, device, logging 옵션 포함)
- [ ] 재현성 유틸 구현 (AC: 3)
  - [ ] `set_seed()` — random/numpy/torch/cuda seed, `torch.backends.cudnn.deterministic=True`
  - [ ] DataLoader `worker_init_fn` 및 `generator` seed 처리
- [ ] 공통 Trainer 구현 (AC: 4)
  - [ ] train/val epoch loop, loss/metric 집계, best checkpoint 저장
  - [ ] CSV 로거 및 TensorBoard 로거 (config로 선택)
- [ ] 평가 모듈 구현 (AC: 5)
  - [ ] `compute_metrics(y_true, y_prob)` — Accuracy/P/R/F1/AUROC (scikit-learn)
  - [ ] `metrics.json` 저장 유틸
- [ ] 의존성 및 문서 (AC: 6)
  - [ ] `requirements.txt` 작성, README에 설치/실행 가이드
- [ ] Unit test 작성 (AC: 7)
  - [ ] `tests/test_config.py`, `tests/test_seed.py`, `tests/test_metrics.py`

### Dev Notes

- **Source Tree 정합** [Source: architecture.md#Source Tree]: 최종 트리는 `src/data`, `src/vision`, `src/text`, `src/matching`, `src/fusion`(내 `baseline.py`), `src/utils`, `scripts/`, `experiments/`, `demo/`, `tests/`이다. Epic 1의 공통 Trainer/평가 코드는 `src/utils/` 또는 `src/training`·`src/evaluation`에 두되, Phase 1 baseline 모델은 architecture대로 `src/fusion/baseline.py`(및 인코더 모듈)에 배치해 이후 Epic과 충돌하지 않게 한다. 스캐폴딩 시 architecture 트리를 기준으로 삼고, 편차가 필요하면 architecture.md를 함께 갱신할 것(Coding Standards의 스키마/문서 동기화 원칙).
- **아키텍처 방향** [Source: prd.md#Technical Assumptions]: Monorepo, 모듈 구성은 최종적으로 `detection → visual_entity → text_entity → matching → fusion → demo`로 확장된다. Epic 1에서는 그 토대(공통 학습/평가/실험 관리)만 구축하되, 모듈이 독립적으로 학습·평가·교체 가능해야 ablation이 성립함(NFR6)을 염두에 두고 인터페이스를 느슨하게 설계할 것.
- **Framework**: PyTorch + HuggingFace Transformers + torchvision. 단일 GPU(Colab Pro/로컬) 제약(NFR1) — config에 batch size / max epoch / AMP(mixed precision) 옵션을 노출해 GPU 메모리에 맞게 조정 가능하게.
- **실험 관리**(NFR3): config yaml + seed 고정 + 결과 로깅(CSV/TensorBoard 중 택1). 실험별 출력은 `outputs/<exp_name>/`에 config 사본과 함께 저장하여 어떤 설정으로 얻은 결과인지 추적 가능하게 한다.
- **평가 지표**(NFR4): 최종 분류기 지표는 Accuracy/Precision/Recall/F1/AUROC. 이 metric 모듈이 Epic 5 ablation table의 공통 기준이 되므로 여기서 한 번만 구현하고 전 실험이 재사용한다.

### Testing

- `tests/` 하위 pytest 기반 unit test: config 로딩(누락 키 에러 포함), seed 고정 후 동일 난수 재현, metric 계산 결과가 scikit-learn 기대값과 일치.
- 수동 검증: 더미 데이터로 `train.py` 1 epoch smoke run이 에러 없이 checkpoint/metrics.json을 생성.

---

## Story 1.2: Fakeddit 데이터 파이프라인

### Status

Approved

### Story

As a **ML 엔지니어(개발자)**,
I want **Fakeddit 데이터셋의 다운로드/전처리/2-way label 정리, train/val/test 분할, 금융 키워드 필터링 subset 스크립트**,
so that **모든 baseline과 최종 모델이 동일한 데이터·분할 위에서 학습/평가되어 공정한 ablation 비교가 가능하다**.

### Acceptance Criteria

1. Fakeddit 메타데이터(tsv)와 이미지의 다운로드/준비 절차가 스크립트 또는 README로 문서화되고, 로컬 데이터 디렉토리 구조(`data/fakeddit/`)가 표준화된다 (대용량 원본 데이터는 git 제외).
2. 전처리 스크립트가 Fakeddit 원본 `2_way_label`을 프로젝트 내부 라벨 규약 **0=REAL, 1=FAKE**(architecture.md `NewsSample` 스키마)로 명시적으로 매핑·검증하고, 이미지 파일이 존재하고 로딩 가능한(손상 파일 제외) 샘플만 남긴 클린 manifest(csv/parquet: sample_id, image_path, text, label, source)를 생성한다. 라벨 매핑 규칙은 코드 상수 + 문서로 기록된다.
3. train/val/test 분할이 seed 고정 하에 생성되어 파일로 저장되며(Fakeddit 공식 분할이 있으면 이를 사용), 이후 모든 실험이 동일 분할 파일을 참조한다. 각 분할의 샘플 수와 label 분포가 로그/문서로 기록된다.
4. 금융 키워드 필터링 스크립트가 config로 정의된 키워드 리스트(예: stock, invest, CEO, earnings, IPO, crypto, bank, merger 등)로 금융 subset manifest를 생성하고, subset 크기·label 분포를 리포트한다 (FR11).
5. PyTorch `Dataset` 클래스가 구현된다: manifest를 읽어 (image tensor, tokenized text, label)을 반환하며, 이미지 transform(resize/normalize)과 tokenizer(HuggingFace)를 config로 주입 가능하다. 텍스트 전용/이미지 전용 모드도 지원한다 (Story 1.3에서 사용).
6. 단일 GPU 환경을 고려해 config로 subsampling(예: 학습용 N만 샘플) 옵션을 지원한다 (NFR1).
7. 전처리·분할·필터링 로직에 대한 unit test가 통과한다.

### Tasks / Subtasks

- [ ] 데이터 다운로드/배치 문서화 및 준비 스크립트 (AC: 1)
  - [ ] `scripts/download_fakeddit.py`(또는 README 절차) — 메타데이터 tsv, 이미지 준비
  - [ ] `data/` .gitignore 처리, 디렉토리 규약 문서화
- [ ] 전처리 스크립트 구현 (AC: 2)
  - [ ] `scripts/preprocess_fakeddit.py` — 2-way label 정리, 결측/손상 이미지 필터링(PIL open 검증), 텍스트 필드(`clean_title`) 정리
  - [ ] 클린 manifest(csv) 출력
- [ ] 분할 생성 (AC: 3)
  - [ ] 공식 train/val/test 분할 매핑 또는 seed 고정 stratified split
  - [ ] 분할 파일 저장 + 분포 통계 리포트 출력
- [ ] 금융 subset 필터링 (AC: 4)
  - [ ] `scripts/filter_financial.py` — config 키워드 리스트 기반(대소문자 무시, 단어 경계 매칭) subset manifest 생성
  - [ ] subset 통계 리포트 (규모 부족 리스크 조기 감지용)
- [ ] `FakedditDataset` 구현 (AC: 5, 6)
  - [ ] `src/data/fakeddit.py` — image transform, tokenizer(max_length, truncation), 모달 선택(text/image/both), subsampling 옵션
  - [ ] collate function (padding 포함)
- [ ] Unit test (AC: 7)
  - [ ] 키워드 필터 매칭, 분할 재현성(동일 seed → 동일 분할), Dataset이 올바른 shape/type 반환

### Dev Notes

- **Fakeddit** [Source: prd.md#Technical Assumptions; arXiv:1911.03854]: 100만+ Reddit 기반 text-image pair, 2/3/6-way label 제공. 본 프로젝트는 2-way(Fake/Real)를 사용(FR1). 메타데이터 tsv의 `clean_title`을 텍스트 입력으로, `image_url` 기반 다운로드 이미지를 사용. 이미지 전량 다운로드는 비현실적이므로 필요한 subset만 준비하는 전략 허용.
- **언어/모델 선택**: Fakeddit은 영어 → 텍스트 인코더는 BERT/DeBERTa 계열 영어 모델 우선 [Source: prd.md#Technical Assumptions].
- **금융 subset**(FR11, NFR5): 키워드 필터링 기반. subset 규모 부족이 주요 리스크로 명시되어 있으므로(prd.md#Checklist) 통계 리포트를 반드시 출력해 조기에 규모를 확인한다. 공개 데이터만 사용, 크롤링 금지(NFR5).
- **라벨 규약 주의** [Source: architecture.md#Data Models]: 내부 표준은 `NewsSample.label` — **0=REAL, 1=FAKE**. Fakeddit 원본 `2_way_label`의 인코딩은 배포 버전에 따라 해석이 다를 수 있으므로, 다운로드한 메타데이터에서 실제 인코딩을 확인한 뒤 명시적 매핑 함수로 변환하고 unit test로 고정할 것 (라벨 반전은 전체 실험을 무효화하는 치명적 버그).
- **manifest 필드**: architecture의 `NewsSample`(sample_id, image_path, title, body, label, source)과 호환되게 구성. Fakeddit은 `clean_title`만 있으므로 body는 빈 문자열 허용, `source="fakeddit"` 기록.
- **재현성**(NFR3): 분할은 파일로 고정 저장 — 이후 Epic 2~5의 모든 ablation이 이 분할 파일을 참조해야 비교가 성립한다.
- **의존 스토리**: Story 1.1의 config 시스템과 seed 유틸을 사용한다.

### Testing

- pytest: 키워드 필터(경계 케이스: 부분 문자열 미매칭), stratified split 분포 검증, Dataset `__getitem__` 반환 타입/shape, 손상 이미지 제외 로직.
- 수동 검증: 소규모 샘플(수백 건)로 전체 파이프라인 실행 → manifest, 분할 파일, subset 리포트 생성 확인.

---

## Story 1.3: 단일 모달 baseline (BERT only / ResNet only)

### Status

Approved

### Story

As a **ML 엔지니어(개발자)**,
I want **텍스트 전용 BERT 분류기와 이미지 전용 ResNet 분류기를 학습·평가하는 baseline**,
so that **ablation table의 첫 두 행(BERT only, ResNet only) 기준 성능을 확보하고 이후 fusion 개선 폭을 측정할 수 있다**.

### Acceptance Criteria

1. BERT only baseline: HuggingFace pretrained 모델(예: `bert-base-uncased`) + classification head로 텍스트 → Fake/Real 이진 분류 모델이 구현되고, config(`configs/bert_only.yaml`)로 학습된다.
2. ResNet only baseline: torchvision pretrained ResNet(예: ResNet-50, ImageNet weights) + classification head로 이미지 → Fake/Real 이진 분류 모델이 구현되고, config(`configs/resnet_only.yaml`)로 학습된다.
3. 두 모델 모두 Story 1.1의 공통 Trainer/평가 모듈을 사용하며, Story 1.2의 고정 분할(train/val/test)로 학습·평가된다.
4. test set에 대해 Accuracy/Precision/Recall/F1/AUROC가 `outputs/<exp_name>/metrics.json`에 기록되고, 두 baseline 모두 **test AUROC ≥ 0.60**(chance 0.5 대비 명확한 학습 신호)을 달성한다. 미달 시 하이퍼파라미터/데이터 점검 후 결과와 원인을 문서화한다.
5. 각 모델은 분류 logit 외에 penultimate feature vector(예: BERT [CLS] 768-dim, ResNet pooled 2048-dim)를 추출하는 인터페이스를 제공한다 (Story 1.4 late fusion에서 재사용).
6. 학습은 단일 GPU에서 완료 가능하다 — config로 batch size/epoch/learning rate/frozen backbone 여부 조정 가능, AMP 지원 (NFR1).
7. 동일 config + seed 재실행 시 metric이 재현된다 (NFR3).

### Tasks / Subtasks

- [ ] BERT 분류기 구현 (AC: 1, 5)
  - [ ] `src/models/text_encoder.py` (또는 architecture 트리 내 상응 위치) — `AutoModel` 기반, [CLS](pooled) feature 추출 메서드 + classification head
  - [ ] `configs/bert_only.yaml` (max_length, lr, epochs, batch_size 등)
- [ ] ResNet 분류기 구현 (AC: 2, 5)
  - [ ] `src/models/image_encoder.py` (또는 architecture 트리 내 상응 위치) — torchvision ResNet, fc 교체, pooled feature 추출 메서드
  - [ ] `configs/resnet_only.yaml` (image size, augmentation, frozen/unfrozen 옵션)
- [ ] 공통 Trainer 연결 (AC: 3, 6)
  - [ ] 모델 팩토리(config `model.type` → 모델 생성) 구현
  - [ ] AMP, gradient accumulation 옵션 확인
- [ ] 학습 및 평가 실행 (AC: 4, 7)
  - [ ] BERT only 학습 → test 평가 → metrics.json 기록
  - [ ] ResNet only 학습 → test 평가 → metrics.json 기록
  - [ ] 동일 seed 재실행으로 재현성 확인
- [ ] Unit test
  - [ ] 모델 forward shape 검증(더미 입력), feature 추출 인터페이스 검증

### Dev Notes

- **모델 선택** [Source: prd.md#Technical Assumptions]: Fakeddit은 영어이므로 BERT 계열 영어 모델(`bert-base-uncased` 기본, 여유 시 DeBERTa 비교 가능). 이미지 측은 pretrained ResNet 활용(NFR1 — pretrained 최대 활용).
- **Feature 추출 인터페이스가 핵심**: Story 1.4의 late fusion과 Epic 2 이후 fusion 모델이 이 인코더들을 재사용한다. `forward()`와 별도로 `encode()`(feature 반환) 메서드를 분리해 두면 모듈 교체·ablation이 쉬워진다(NFR6).
- **학습 팁(단일 GPU)**: backbone freeze + head만 학습하는 옵션을 config에 두면 빠른 iteration 가능. 전체 fine-tuning은 lr 2e-5(BERT), 1e-4~1e-3(head) 수준 권장.
- **ablation 위치**: 이 두 결과는 최종 ablation table(FR10)의 "BERT only", "ResNet only" 행이 된다 — 이후 재학습하지 않도록 checkpoint와 config를 보존한다.

### Testing

- pytest: 더미 배치 forward output shape (logits [B,2] 또는 [B,1]), `encode()` feature dim 검증 (768 / 2048).
- 모델 검증: held-out test set 평가, AUROC ≥ 0.60 확인, 2회 실행 재현성 확인.

---

## Story 1.4: Late fusion baseline (ResNet + BERT)

### Status

Approved

### Story

As a **ML 엔지니어(개발자)**,
I want **ResNet image feature와 BERT text feature를 concat하여 분류하는 late fusion baseline**,
so that **"BERT+Image" ablation 기준선을 기록하고, 이후 Object Detection·Entity consistency 추가가 이 기준선 대비 개선되는지 입증할 토대를 만든다**.

### Acceptance Criteria

1. Late fusion 모델이 구현된다: Story 1.3의 BERT 인코더([CLS] 768-dim)와 ResNet 인코더(pooled 2048-dim)의 feature를 concat(2816-dim) → MLP classification head → Fake/Real 확률 출력 (FR1).
2. 두 가지 학습 모드를 config로 지원한다: (a) 양쪽 인코더 frozen + fusion head만 학습, (b) end-to-end fine-tuning. 최소 (a)는 반드시 완료하고, GPU 여건에 따라 (b)를 수행한다.
3. Story 1.2의 동일 분할로 학습·평가되며, test set Accuracy/Precision/Recall/F1/AUROC가 기록된다.
4. Late fusion(BERT+Image) F1이 단일 모달 baseline(BERT only, ResNet only) 각각의 F1과 비교 기록되며, 비교 결과(개선 여부 포함)가 문서화된다.
5. ablation 기준선 문서(`docs/bmad/results/ablation.md` 또는 `outputs/ablation_table.csv`)가 생성되어 BERT only / ResNet only / BERT+Image 3개 행의 metric이 동일 test set 기준으로 정리된다 (FR10).
6. 단일 샘플(이미지+텍스트) 추론 함수가 제공되어 fake probability를 반환하며, GPU 기준 수 초 이내에 완료된다 (NFR2 사전 검증).
7. 동일 config + seed 재실행 시 metric이 재현된다.

### Tasks / Subtasks

- [ ] Late fusion 모델 구현 (AC: 1)
  - [ ] `src/fusion/baseline.py` [Source: architecture.md#Source Tree] — text/image encoder 조합, concat + MLP head (hidden dim, dropout config화)
  - [ ] Story 1.3 인코더 checkpoint 로딩 지원 (pretrained 인코더 재사용)
- [ ] 학습 모드 지원 (AC: 2)
  - [ ] config `freeze_encoders: true/false` 분기, param group별 lr 설정
  - [ ] `configs/late_fusion.yaml` 작성
- [ ] 학습/평가 실행 (AC: 3, 7)
  - [ ] frozen 모드 학습 → test 평가 → metrics.json
  - [ ] (여건 시) end-to-end 모드 학습 → 비교
  - [ ] seed 재실행 재현성 확인
- [ ] Ablation 기준선 기록 (AC: 4, 5)
  - [ ] BERT only / ResNet only / BERT+Image metric을 하나의 표로 취합하는 스크립트(`scripts/collect_ablation.py`) 또는 문서 작성
  - [ ] 개선 폭 분석 코멘트 기록
- [ ] 단일 샘플 추론 (AC: 6)
  - [ ] `predict(image_path, text) -> fake_prob` 함수 + 추론 시간 측정
- [ ] Unit test
  - [ ] fusion forward shape, concat dim 검증, frozen 모드에서 encoder grad 미갱신 검증

### Dev Notes

- **Late fusion 정의** [Source: prd.md#Epic 1]: 각 모달을 독립 인코딩 후 feature 수준에서 concat하여 분류 — Epic 2의 cross-modal attention(early/intermediate fusion)과 대비되는 가장 단순한 결합. 이 단순함이 baseline으로서의 가치다.
- **이 스토리의 산출물이 프로젝트 핵심 가설의 비교 기준**: PRD 핵심 입증 목표는 "+Object Detection, +Entity consistency가 F1을 추가 개선"이므로, 여기서 기록한 BERT+Image 기준선의 실험 조건(분할, seed, 전처리, metric 코드)이 이후 전 실험에서 동결되어야 한다. checkpoint/config/metrics를 `outputs/`에 보존할 것.
- **구현 팁**: frozen 모드에서는 인코더 feature를 사전 추출·캐싱하면 학습이 수 분 내로 단축된다 (단일 GPU 제약 NFR1 대응). end-to-end 모드는 모달별 lr 차등(인코더 낮게, head 높게) 권장.
- **모듈성**(NFR6): fusion head는 입력 feature dim만 맞으면 인코더 교체가 가능하도록 dim을 config에서 주입 — Epic 2에서 region feature 시퀀스로 확장될 때 재작성 최소화.

### Testing

- pytest: concat 차원(768+2048=2816) 및 output shape 검증, `freeze_encoders=true` 시 인코더 `requires_grad=False` 확인, `predict()` 반환 범위(0~1) 검증.
- 모델 검증: 동일 test set에서 3개 구성 비교 평가, 재현성 2회 실행 확인.
- 수동 검증: 대표 샘플 수 건으로 단일 추론 실행 및 시간 측정(수 초 이내).
