# Epic 1: Baseline 파이프라인 — User Stories

> 작성자: Bob (BMAD Scrum Master) · 작성일: 2026-08-10
> 근거 문서: `docs/bmad/prd.md` (Epic 1, FR1, FR10, FR11, NFR1~NFR6)

---

## Story 1.1: 프로젝트 스캐폴딩

### Status

Ready for Review

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

- [x] repo 디렉토리 구조 및 패키지 초기화 (AC: 1)
  - [x] `src/` 패키지 생성 (architecture Source Tree 기준), 하위 모듈 `data/`, `fusion/`, `utils/` + 공통 학습/평가 모듈 및 `__init__.py`
  - [x] `configs/`, `scripts/`, `tests/`, `outputs/`(.gitignore 처리) 생성
- [x] config 시스템 구현 (AC: 2)
  - [x] yaml 로더 + dataclass/dict 기반 config 객체 (`src/utils/config.py`)
  - [x] `scripts/train.py`, `scripts/evaluate.py` 엔트리포인트에 `--config` 인자 연결
  - [x] 예시 config `configs/base.yaml` 작성 (exp_name, seed, device, logging 옵션 포함)
- [x] 재현성 유틸 구현 (AC: 3)
  - [x] `set_seed()` — random/numpy/torch/cuda seed, `torch.backends.cudnn.deterministic=True`
  - [x] DataLoader `worker_init_fn` 및 `generator` seed 처리
- [x] 공통 Trainer 구현 (AC: 4)
  - [x] train/val epoch loop, loss/metric 집계, best checkpoint 저장
  - [x] CSV 로거 및 TensorBoard 로거 (config로 선택)
- [x] 평가 모듈 구현 (AC: 5)
  - [x] `compute_metrics(y_true, y_prob)` — Accuracy/P/R/F1/AUROC (scikit-learn)
  - [x] `metrics.json` 저장 유틸
- [x] 의존성 및 문서 (AC: 6)
  - [x] `requirements.txt` 작성, README에 설치/실행 가이드
- [x] Unit test 작성 (AC: 7)
  - [x] `tests/test_config.py`, `tests/test_seed.py`, `tests/test_metrics.py`

### Dev Notes

- **Source Tree 정합** [Source: architecture.md#Source Tree]: 최종 트리는 `src/data`, `src/vision`, `src/text`, `src/matching`, `src/fusion`(내 `baseline.py`), `src/utils`, `scripts/`, `experiments/`, `demo/`, `tests/`이다. Epic 1의 공통 Trainer/평가 코드는 `src/utils/` 또는 `src/training`·`src/evaluation`에 두되, Phase 1 baseline 모델은 architecture대로 `src/fusion/baseline.py`(및 인코더 모듈)에 배치해 이후 Epic과 충돌하지 않게 한다. 스캐폴딩 시 architecture 트리를 기준으로 삼고, 편차가 필요하면 architecture.md를 함께 갱신할 것(Coding Standards의 스키마/문서 동기화 원칙).
- **아키텍처 방향** [Source: prd.md#Technical Assumptions]: Monorepo, 모듈 구성은 최종적으로 `detection → visual_entity → text_entity → matching → fusion → demo`로 확장된다. Epic 1에서는 그 토대(공통 학습/평가/실험 관리)만 구축하되, 모듈이 독립적으로 학습·평가·교체 가능해야 ablation이 성립함(NFR6)을 염두에 두고 인터페이스를 느슨하게 설계할 것.
- **Framework**: PyTorch + HuggingFace Transformers + torchvision. 단일 GPU(Colab Pro/로컬) 제약(NFR1) — config에 batch size / max epoch / AMP(mixed precision) 옵션을 노출해 GPU 메모리에 맞게 조정 가능하게.
- **실험 관리**(NFR3): config yaml + seed 고정 + 결과 로깅(CSV/TensorBoard 중 택1). 실험별 출력은 `outputs/<exp_name>/`에 config 사본과 함께 저장하여 어떤 설정으로 얻은 결과인지 추적 가능하게 한다.
- **평가 지표**(NFR4): 최종 분류기 지표는 Accuracy/Precision/Recall/F1/AUROC. 이 metric 모듈이 Epic 5 ablation table의 공통 기준이 되므로 여기서 한 번만 구현하고 전 실험이 재사용한다.

### Testing

- `tests/` 하위 pytest 기반 unit test: config 로딩(누락 키 에러 포함), seed 고정 후 동일 난수 재현, metric 계산 결과가 scikit-learn 기대값과 일치.
- 수동 검증: 더미 데이터로 `train.py` 1 epoch smoke run이 에러 없이 checkpoint/metrics.json을 생성.

### Dev Agent Record

구현 완료 (2026-08-10). 검증 결과:

- **pytest 24개 전부 통과** — `tests/test_config.py`(8), `test_seed.py`(5), `test_metrics.py`(8), `test_trainer_smoke.py`(3).
- **AC3 재현성 실측**: 동일 config 2회 실행(`repro_a`/`repro_b`) → `metrics.json` 완전 일치.
- **AC4 학습 동작 실측**: 더미 데이터에서 val_f1 0.0 → 1.0 수렴 확인 (학습 루프가 실제로 학습함).
- **AC2/AC5 실측**: `outputs/<exp_name>/`에 `config.yaml`·`best.pt`·`history.csv`·`metrics.json` 생성 확인.

구현 시 결정 사항:

- 공통 학습/평가 코드는 Dev Notes가 허용한 두 위치 중 `src/training/`·`src/evaluation/`을 선택 (역할이 드러나 Epic 2~5에서 찾기 쉬움). architecture Source Tree의 `src/data`·`src/fusion`·`src/utils`는 그대로 유지.
- **모듈 교체 계약**(NFR6 ablation 전제): 모든 모델은 `forward(batch: dict) -> logits [B,2]`, 라벨은 batch의 `label` 키. 이 계약만 지키면 Trainer 수정 없이 Phase 1~4 모델을 갈아끼울 수 있다. 데이터셋/모델은 각각 `src/data/registry.py`·`src/fusion/registry.py`의 레지스트리에 등록해 config `data.name`/`model.name`으로 선택한다.
- config는 `_base_` 상속과 `--set key=value` 오버라이드를 지원 (ablation config를 base 상속으로 최소 diff 작성 가능).
- 스캐폴딩만으로 end-to-end 실행을 검증할 수 있도록 `dummy` 데이터셋/모델을 등록해 두었다. Story 1.2에서 `fakeddit`, 1.3~1.4에서 실제 모델이 추가된다.
- README의 프로젝트 구조·실행 절을 실제 구현 인터페이스에 맞게 갱신.

---

## Story 1.2: Fakeddit 데이터 파이프라인

### Status

Ready for Review

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

- [x] 데이터 다운로드/배치 문서화 및 준비 스크립트 (AC: 1)
  - [x] `scripts/download_fakeddit.py`(또는 README 절차) — 메타데이터 tsv, 이미지 준비
  - [x] `data/` .gitignore 처리, 디렉토리 규약 문서화
- [x] 전처리 스크립트 구현 (AC: 2)
  - [x] `scripts/preprocess_fakeddit.py` — 2-way label 정리, 결측/손상 이미지 필터링(PIL open 검증), 텍스트 필드(`clean_title`) 정리
  - [x] 클린 manifest(csv) 출력
- [x] 분할 생성 (AC: 3)
  - [x] 공식 train/val/test 분할 매핑 또는 seed 고정 stratified split
  - [x] 분할 파일 저장 + 분포 통계 리포트 출력
- [x] 금융 subset 필터링 (AC: 4)
  - [x] `scripts/filter_financial.py` — config 키워드 리스트 기반(대소문자 무시, 단어 경계 매칭) subset manifest 생성
  - [x] subset 통계 리포트 (규모 부족 리스크 조기 감지용)
- [x] `FakedditDataset` 구현 (AC: 5, 6)
  - [x] `src/data/fakeddit.py` — image transform, tokenizer(max_length, truncation), 모달 선택(text/image/both), subsampling 옵션
  - [x] collate function (padding 포함)
- [x] Unit test (AC: 7)
  - [x] 키워드 필터 매칭, 분할 재현성(동일 seed → 동일 분할), Dataset이 올바른 shape/type 반환

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

### Dev Agent Record

구현 완료 (2026-08-10, James/Dev). **실데이터 미보유 상태에서 구현** — 네트워크 다운로드는 시도하지 않았고,
모든 로직은 합성(synthetic) manifest/이미지로 검증했다.

#### 검증 결과

- **pytest 99개 전부 통과** (Story 1.1의 24개 + 신규 75개). 신규: `test_labels.py`(8), `test_preprocess.py`(13),
  `test_splits.py`(11), `test_financial_filter.py`(20), `test_fakeddit_dataset.py`(23).
- **수동 스모크(합성 데이터 300건)**: 손상 이미지 5건·이미지 결측 5건을 심어 두고 전체 파이프라인 실행 →
  `preprocess` 통계 `{raw:300, dropped_missing_image:5, dropped_corrupt_image:5, kept:290}`,
  stratified 분할 `train 232 / val 28 / test 30`(모두 fake_ratio 0.50 유지),
  `filter_financial` subset 287건 + 키워드별 히트 리포트 및 "규모 부족" 경고 출력 확인.
- **Story 1.1 회귀 없음**: `dummy` 데이터셋 등록·Trainer 계약(batch dict, 라벨 키 `label`) 테스트 통과.

#### 구현 시 결정 사항

- **라벨 매핑을 단일 모듈로 격리**: `src/data/labels.py`의 `FAKEDDIT_2WAY_TO_INTERNAL = {1: 0, 0: 1}`.
  Fakeddit 원본은 `1 = true(real)`이고 내부 규약은 `1 = FAKE`이므로 매핑은 **항등이 아니라 반전**이다.
  `test_labels.py`가 "항등이면 실패"까지 명시적으로 검사해 라벨 반전 회귀를 차단한다.
- **로직/스크립트 분리**: 전처리·분할·필터 로직은 `src/data/{preprocess,splits,financial}.py`에 DataFrame
  in/out 함수로 두고, `scripts/*.py`는 얇은 CLI 래퍼로만 구성 — 실데이터 없이 단위 테스트가 가능하다.
- **manifest 스키마**: `sample_id, image_path, title, body, text, label, source` (architecture `NewsSample` 호환).
  `image_path`는 데이터 루트 기준 상대경로로 저장하고 config `data.image_root`로 복원 — 경로 이식성 확보.
- **분할 재현성**: 라벨 그룹별 `np.random.default_rng(seed + label)` 순열 + `sample_id` 정렬 시작점을 사용해
  **입력 행 순서에 불변**이다(테스트로 고정). 공식 분할 tsv가 있으면 `--split-mode official`이 우선.
- **오프라인 tokenizer 경로**: `data.tokenizer.name: null`이면 tokenizer 없이 raw text를 배치 `text` 키로 전달.
  테스트는 `conftest.FakeTokenizer`를 주입해 padding/truncation/collate까지 네트워크 없이 검증한다.
- **collate 소유권**: `FakedditDataset`이 자기 `collate_fn`(pad_token_id 반영)을 들고 있고, `registry.build_dataloaders`가
  `getattr(dataset, "collate_fn", None)`로 자동 사용 — dummy 등 기존 데이터셋 경로는 변경 없이 동작.
- `data.subsample` / `data.subsample_per_split`로 층화 subsampling(NFR1), `data.modality`로 text/image/both 선택(Story 1.3).
- 다운로드 스크립트는 **원본을 재배포하지 않으며**, 이미지 다운로드는 `--download-images` opt-in으로만 동작한다.

#### 실데이터 확보 후 남은 작업 (Story 1.3 착수 전 필수)

1. **`2_way_label` 실제 인코딩 재확인** — 배포본에서 `2_way_label`과 `clean_title`을 눈으로 대조해
   `FAKEDDIT_2WAY_TO_INTERNAL`이 맞는지 검증한다. 다르면 상수 + `test_labels.py` + `docs/DATA.md`를 함께 수정.
2. **실 manifest/분할 생성 및 통계 기록** — AC3의 "샘플 수·label 분포 문서화"는 실행 산출물
   (`processed/split_report.json`)로 채워야 완료된다. 공식 분할 tsv 파일명이 배포본과 다르면
   `OFFICIAL_SPLIT_FILES` 조정 필요.
3. **금융 subset 실제 규모 확인** (FR11 최대 리스크) — 현재 키워드 리스트로 실데이터에서 몇 건이 나오는지 측정.
   부족하면 키워드 확장 또는 subset 단독 학습 대신 "금융 subset 평가만" 전략으로 축소 검토.
4. **이미지 다운로드 성공률/손상률 실측** — dead link 비율에 따라 확보 가능한 멀티모달 샘플 수가 결정된다.
5. **실 tokenizer(`bert-base-uncased`) 경로 검증** — 본 환경에서는 네트워크 부재로 HuggingFace 다운로드 경로를
   실행하지 못했다 (가짜 tokenizer로 인터페이스만 검증).

---

## Story 1.3: 단일 모달 baseline (BERT only / ResNet only)

### Status

Ready for Review

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

- [x] BERT 분류기 구현 (AC: 1, 5)
  - [x] `src/models/text_encoder.py` (또는 architecture 트리 내 상응 위치) — `AutoModel` 기반, [CLS](pooled) feature 추출 메서드 + classification head
  - [x] `configs/bert_only.yaml` (max_length, lr, epochs, batch_size 등)
- [x] ResNet 분류기 구현 (AC: 2, 5)
  - [x] `src/models/image_encoder.py` (또는 architecture 트리 내 상응 위치) — torchvision ResNet, fc 교체, pooled feature 추출 메서드
  - [x] `configs/resnet_only.yaml` (image size, augmentation, frozen/unfrozen 옵션)
- [x] 공통 Trainer 연결 (AC: 3, 6)
  - [x] 모델 팩토리(config `model.type` → 모델 생성) 구현
  - [x] AMP, gradient accumulation 옵션 확인
- [x] 학습 및 평가 실행 (AC: 4, 7) — *실행 경로 구현·스모크 완료, **실데이터 학습은 미실행** (아래 Dev Agent Record 참조)*
  - [x] BERT only 학습 → test 평가 → metrics.json 기록 *(합성 데이터 스모크만; 실 metric 미기록)*
  - [x] ResNet only 학습 → test 평가 → metrics.json 기록 *(합성 데이터 스모크만; 실 metric 미기록)*
  - [x] 동일 seed 재실행으로 재현성 확인 *(모델 초기화·forward 수준)*
- [x] Unit test
  - [x] 모델 forward shape 검증(더미 입력), feature 추출 인터페이스 검증

### Dev Notes

- **모델 선택** [Source: prd.md#Technical Assumptions]: Fakeddit은 영어이므로 BERT 계열 영어 모델(`bert-base-uncased` 기본, 여유 시 DeBERTa 비교 가능). 이미지 측은 pretrained ResNet 활용(NFR1 — pretrained 최대 활용).
- **Feature 추출 인터페이스가 핵심**: Story 1.4의 late fusion과 Epic 2 이후 fusion 모델이 이 인코더들을 재사용한다. `forward()`와 별도로 `encode()`(feature 반환) 메서드를 분리해 두면 모듈 교체·ablation이 쉬워진다(NFR6).
- **학습 팁(단일 GPU)**: backbone freeze + head만 학습하는 옵션을 config에 두면 빠른 iteration 가능. 전체 fine-tuning은 lr 2e-5(BERT), 1e-4~1e-3(head) 수준 권장.
- **ablation 위치**: 이 두 결과는 최종 ablation table(FR10)의 "BERT only", "ResNet only" 행이 된다 — 이후 재학습하지 않도록 checkpoint와 config를 보존한다.

### Testing

- pytest: 더미 배치 forward output shape (logits [B,2] 또는 [B,1]), `encode()` feature dim 검증 (768 / 2048).
- 모델 검증: held-out test set 평가, AUROC ≥ 0.60 확인, 2회 실행 재현성 확인.

### Dev Agent Record

구현 완료 (2026-08-10, James/Dev). **실데이터·사전학습 가중치 없이 구현** — 네트워크 접근이 없는 환경이라
HuggingFace/torchvision 가중치 다운로드는 시도하지 않았고, 모든 경로는 랜덤 초기화 인코더와
Story 1.2의 합성(synthetic) manifest/이미지로 검증했다.

#### 구현 파일

- `src/fusion/encoders.py` (신규) — `TextEncoder`(BERT 계열), `ImageEncoder`(torchvision ResNet), `ClassificationHead`
- `src/fusion/baseline.py` (신규) — `SingleModalClassifier` + `@register_model("text_only" | "image_only")`
- `src/fusion/registry.py` — `_ensure_builtin_models()` 지연 import 추가 (`dummy` 등록 경로는 그대로)
- `src/fusion/__init__.py` — 인코더 공개 export
- `scripts/train.py` — `build_optimizer()` 추가 (`train.head_lr`가 있으면 인코더/head lr 분리)
- `configs/bert_only.yaml`, `configs/resnet_only.yaml` (신규)
- `tests/test_baseline_models.py` (신규, 37개)

#### 구현 시 결정 사항

- **인코더와 분류기를 분리했다** (AC5 / NFR6의 핵심): 인코더는 분류 head를 갖지 않고
  `forward(batch) -> features [B, D]` + `output_dim` + `freeze()/unfreeze()`만 제공한다.
  Story 1.4의 late fusion은 `build_text_encoder(cfg)` / `build_image_encoder(cfg)`를 그대로 호출해
  두 feature를 concat하기만 하면 된다(테스트 `test_config_builders_are_reusable_for_late_fusion`이 이를 고정).
  분류기 쪽 `encode(batch)`는 penultimate feature를, `forward(batch)`는 logits [B,2]를 반환한다.
- **Trainer 계약 유지**: 모델은 `forward(batch: dict) -> logits [B, 2]`, 라벨 키 `label`. Story 1.1의
  Trainer/metrics/`scripts/{train,evaluate}.py`를 한 줄도 바꾸지 않고 `model.name`만 교체해 학습된다.
- **오프라인 안전 경로**: `model.text.pretrained: false`면 `BertConfig`로 랜덤 초기화(모델 id 조회 없음),
  `model.image.pretrained: false`면 `weights=None` — 두 경우 모두 네트워크 접근이 발생하지 않는다.
  `pretrained: true`(shipped config 기본값)가 실가중치 경로이며, `local_files_only`로 캐시 전용 로딩도 가능.
  테스트는 랜덤 초기화 + 주입 backbone(`encoder=`) 두 방식으로만 검증한다.
- **feature dim 정책**: 기본은 네이티브 dim(BERT [CLS] **768**, ResNet-50 pooled **2048**)을 그대로 노출한다 —
  Story 1.4의 concat 2816-dim 요구와 정합. 다만 두 인코더 모두 `proj_dim`을 지원해
  `proj_dim: 768`로 두면 이미지 쪽도 768로 정렬되어 Epic 2 이후 cross-attention(공통 768 hidden)에 바로 꽂힌다.
  두 모드 모두 테스트로 고정했다.
- **frozen backbone**(AC6, NFR1): `freeze: true`면 `requires_grad=False` + `train()` 호출 시에도 backbone은
  `eval()` 유지(BatchNorm running stat/Dropout 고정). `param_groups()`는 frozen 파라미터를 optimizer에서 제외한다.
- **lr 차등**: config `train.lr`(인코더) / `train.head_lr`(head). `scripts/train.py`가 모델에
  `param_groups`가 있을 때만 사용하므로 `dummy` 등 기존 모델 경로는 영향 없음.
- **모달 선택은 데이터 쪽 config로**: `configs/bert_only.yaml`은 `data.modality: text`(이미지 I/O 생략),
  `configs/resnet_only.yaml`은 `data.modality: image` + `tokenizer.name: null`(토큰화 생략) — 불필요한 I/O 제거.
- **인코더 checkpoint 재사용**: `save_encoder()` / `load_encoder()`로 인코더 가중치만 따로 저장·로딩
  (Story 1.4 Task "Story 1.3 인코더 checkpoint 로딩 지원"의 대응 지점).

#### 검증 결과

- **pytest 136개 전부 통과** (Story 1.1~1.2의 99개 회귀 없음 + 신규 37개), 네트워크 접근 0회.
  신규 커버리지: 인코더 forward shape/output_dim(768, 512, 2048), projection, mean-pooling의 padding 무시,
  누락 키 에러 메시지, freeze/unfreeze 및 frozen backbone 무-grad, head MLP 변형, `encode()`/`predict_proba()`
  범위, param group 분리, 인코더 checkpoint round-trip, 레지스트리 등록(+`dummy` 회귀), shipped config 파싱,
  Story 1.2 `FakedditDataset`(text/image 모드) + 공통 Trainer 연동 학습 1 epoch, 동일 seed 재현성.
- **CLI 스모크**(합성 `dummy` 데이터셋, CPU): `python scripts/train.py --config configs/base.yaml --set
  model.name=text_only ...` 및 `model.name=image_only model.image.arch=resnet18 model.image.pretrained=false`
  둘 다 1 epoch 학습 → checkpoint/metrics.json 생성 확인. `train.head_lr` 지정 시 param group 분리 경로도 실행됨.

#### 실데이터/실가중치 확보 후 남은 작업 (미검증 항목)

1. **AC1/AC2의 "config로 학습된다"의 실제 실행** — `configs/bert_only.yaml`, `configs/resnet_only.yaml`은
   Fakeddit manifest(`data/fakeddit/processed/*.csv`)와 `bert-base-uncased`/ImageNet 가중치를 요구하므로
   본 환경에서 end-to-end 실행하지 못했다. 검증한 것은 config 파싱·모델 조립·학습 루프까지다.
2. **AC4 (test AUROC ≥ 0.60) 미검증** — 실데이터가 없어 `outputs/<exp_name>/metrics.json`의 test 지표를
   생성하지 못했다. 데이터 확보 후 `scripts/train.py` → `scripts/evaluate.py --split test`로 기록하고,
   미달 시 원인(lr, frozen 여부, subsample 규모)을 이 절에 문서화할 것.
3. **사전학습 로딩 경로 미실행** — `AutoModel.from_pretrained("bert-base-uncased")`와
   torchvision `weights="DEFAULT"` 경로는 다운로드가 필요해 실행하지 못했다. 실가중치에서 hidden 768 /
   pooled 2048 가정이 유지되는지(다른 backbone으로 교체 시 특히) 최초 1회 확인 필요.
4. **AMP / 단일 GPU 학습 시간·메모리 미측정** — CPU에서만 검증했다. `train.amp: true` 경로는 CUDA에서만
   활성화되므로 GPU 환경에서 별도 확인이 필요하다.
5. **AC7 재현성은 모델 초기화·forward 수준에서만 확인** — 동일 seed 2회 실행의 *metric* 일치는
   실데이터 학습 후 재확인해야 한다.


---

## Story 1.4: Late fusion baseline (ResNet + BERT)

### Status

Changes Requested (PO, 2026-08-10)

### Story

As a **ML 엔지니어(개발자)**,
I want **ResNet image feature와 BERT text feature를 concat하여 분류하는 late fusion baseline**,
so that **"BERT+Image" ablation 기준선을 기록하고, 이후 Object Detection·Entity consistency 추가가 이 기준선 대비 개선되는지 입증할 토대를 만든다**.

### Acceptance Criteria

1. Late fusion 모델이 구현된다: Story 1.3의 BERT 인코더([CLS] 768-dim)와 ResNet 인코더(pooled 2048-dim)의 feature를 concat(2816-dim) → MLP classification head → Fake/Real 확률 출력 (FR1).
2. 두 가지 학습 모드를 config로 지원한다: (a) 양쪽 인코더 frozen + fusion head만 학습, (b) end-to-end fine-tuning. 최소 (a)는 반드시 완료하고, GPU 여건에 따라 (b)를 수행한다.
3. Story 1.2의 동일 분할로 학습·평가되며, test set Accuracy/Precision/Recall/F1/AUROC가 기록된다.
4. Late fusion(BERT+Image) F1이 단일 모달 baseline(BERT only, ResNet only) 각각의 F1과 비교 기록되며, 비교 결과(개선 여부 포함)가 문서화된다.
5. ablation 기준선 문서(`docs/bmad/results/ablation.md` 또는 `outputs/ablation_table.csv`)가 생성되어 BERT only / ResNet only / BERT+Image 3개 행의 metric이 동일 test set 기준으로 정리된다 (FR10). 또한 표의 각 행에는 metric과 함께 해당 실험의 config 사본, git commit hash, split 파일 해시(manifest·train/val/test), 환경 정보(`requirements.txt` 버전 pin, CUDA/cuDNN 버전)가 기록된다. 이 메타데이터가 없는 행은 `status=unverified`로 표시한다.
6. 단일 샘플(이미지+텍스트) 추론 함수가 제공되어 fake probability를 반환하며, GPU 기준 수 초 이내에 완료된다 (NFR2 사전 검증).
7. 동일 config + seed 재실행 시 metric이 재현된다. **실데이터 학습 2회 실행의 test metric 일치를 최소 1회 실증하고 그 결과를 Dev Agent Record에 기록한다** — 합성/dummy 데이터 재현성은 이 AC를 충족하지 않는다.

### Tasks / Subtasks

- [x] Late fusion 모델 구현 (AC: 1)
  - [x] `src/fusion/baseline.py` [Source: architecture.md#Source Tree] — text/image encoder 조합, concat + MLP head (hidden dim, dropout config화)
  - [x] Story 1.3 인코더 checkpoint 로딩 지원 (pretrained 인코더 재사용)
- [x] 학습 모드 지원 (AC: 2)
  - [x] config `freeze_encoders: true/false` 분기, param group별 lr 설정
  - [x] `configs/late_fusion.yaml` 작성
- [x] 학습/평가 실행 (AC: 3, 7) — *실행 경로 구현·스모크 완료, **실데이터 학습은 미실행** (아래 Dev Agent Record 참조)*
  - [x] frozen 모드 학습 → test 평가 → metrics.json *(합성 데이터 스모크만; 실 metric 미기록)*
  - [x] (여건 시) end-to-end 모드 학습 → 비교 *(경로만 구현·테스트, 실학습 미실행)*
  - [x] seed 재실행 재현성 확인 *(모델 초기화·forward 수준)*
- [x] Ablation 기준선 기록 (AC: 4, 5)
  - [x] BERT only / ResNet only / BERT+Image metric을 하나의 표로 취합하는 스크립트(`scripts/collect_ablation.py`) 또는 문서 작성
  - [x] 개선 폭 분석 코멘트 기록 *(비교 로직 구현 + 표 자동 생성; 실수치는 실데이터 학습 후 채워짐)*
- [x] 단일 샘플 추론 (AC: 6)
  - [x] `predict(image_path, text) -> fake_prob` 함수 + 추론 시간 측정
- [x] Unit test
  - [x] fusion forward shape, concat dim 검증, frozen 모드에서 encoder grad 미갱신 검증

### Dev Notes

- **Late fusion 정의** [Source: prd.md#Epic 1]: 각 모달을 독립 인코딩 후 feature 수준에서 concat하여 분류 — Epic 2의 cross-modal attention(early/intermediate fusion)과 대비되는 가장 단순한 결합. 이 단순함이 baseline으로서의 가치다.
- **이 스토리의 산출물이 프로젝트 핵심 가설의 비교 기준**: PRD 핵심 입증 목표는 "+Object Detection, +Entity consistency가 F1을 추가 개선"이므로, 여기서 기록한 BERT+Image 기준선의 실험 조건(분할, seed, 전처리, metric 코드)이 이후 전 실험에서 동결되어야 한다. checkpoint/config/metrics를 `outputs/`에 보존할 것.
- **구현 팁**: frozen 모드에서는 인코더 feature를 사전 추출·캐싱하면 학습이 수 분 내로 단축된다 (단일 GPU 제약 NFR1 대응). end-to-end 모드는 모달별 lr 차등(인코더 낮게, head 높게) 권장.
- **모듈성**(NFR6): fusion head는 입력 feature dim만 맞으면 인코더 교체가 가능하도록 dim을 config에서 주입 — Epic 2에서 region feature 시퀀스로 확장될 때 재작성 최소화.
- **checkpoint 보관 정책(PO 확정, 2026-08-10)**: `.gitignore`의 `outputs/`·`*.pt`·`checkpoints/`는 **유지**한다. git에 커밋하는 것은 metric·config 사본·git hash·split 해시·환경 정보뿐이다. 단 "seed+config만 있으면 재학습으로 복원 가능하므로 checkpoint는 불필요"라는 가정은 **아직 검증되지 않았다** — 현재 재현성 증빙은 dummy 데이터 2회 실행 일치와 모델 초기화/forward 일치까지이며(Story 1.1·1.4 Dev Agent Record), 실데이터·실가중치·GPU 경로는 한 번도 실행되지 않았고 AMP/cudnn 비결정성도 미측정이다. 따라서 **AC7의 실데이터 재현성 실증이 끝나기 전까지 baseline 3종(`bert_only`/`resnet_only`/`late_fusion`)의 best checkpoint는 로컬 보존 의무**이며 삭제·덮어쓰기를 금한다. 보존 위치(공유 스토리지/개인 로컬)는 프로젝트 오너 결정 사항.

### Testing

- pytest: concat 차원(768+2048=2816) 및 output shape 검증, `freeze_encoders=true` 시 인코더 `requires_grad=False` 확인, `predict()` 반환 범위(0~1) 검증.
- 모델 검증: 동일 test set에서 3개 구성 비교 평가, 재현성 2회 실행 확인.
- 수동 검증: 대표 샘플 수 건으로 단일 추론 실행 및 시간 측정(수 초 이내).

### Dev Agent Record

구현 완료 (2026-08-10, James/Dev). **실데이터·실가중치 없이 구현** — 네트워크 접근이 없는 환경이라
HuggingFace/torchvision 가중치 다운로드는 시도하지 않았고, 모든 경로는 랜덤 초기화(`pretrained: false`)
인코더 또는 주입된 가짜 인코더와 Story 1.2의 합성 manifest/이미지로 검증했다.

#### 구현 파일

- `src/fusion/baseline.py` — `LateFusionClassifier` + `@register_model("late_fusion")`,
  `build_single_sample_batch()` / `predict_single()` (AC6), `DEFAULT_FUSION_HIDDEN_DIMS`
- `src/evaluation/ablation.py` (신규) — `AblationRow`/`EPIC1_ROWS`, `collect_ablation()`,
  `f1_comparison()`, `render_markdown()`, `write_ablation()` (AC4, AC5)
- `src/evaluation/__init__.py`, `src/fusion/__init__.py` — 공개 export 갱신(fusion은 지연 import로 순환 회피)
- `configs/late_fusion.yaml` (신규)
- `scripts/collect_ablation.py` (신규), `scripts/predict.py` (신규)
- `docs/bmad/results/ablation.md` (신규, 자동 생성) + `outputs/ablation_table.csv`(gitignore 대상)
- `tests/test_late_fusion.py` (신규, 47개), README 실행 절 갱신

#### 구현 시 결정 사항

- **인코더를 새로 만들지 않았다**: `build_text_encoder(cfg)` / `build_image_encoder(cfg)`(Story 1.3)를
  그대로 호출하고 `LateFusionClassifier`는 두 feature를 `torch.cat`할 뿐이다. 인코더에 요구하는 계약은
  `forward(batch) -> [B, D]` + `output_dim` 두 개뿐 — Epic 2에서 region feature 인코더로 갈아끼워도
  head는 재작성이 필요 없다(NFR6).
- **Trainer 계약 유지**: `forward(batch: dict) -> logits [B, 2]`, 라벨 키 `label`. Story 1.1의
  Trainer/metrics/`scripts/{train,evaluate}.py`를 한 줄도 바꾸지 않고 `model.name=late_fusion`만으로 학습된다.
  기존 등록(`dummy`/`text_only`/`image_only`)과 데이터셋 등록도 회귀 테스트로 고정했다.
- **concat dim 정책**: 기본은 네이티브 dim — BERT [CLS] 768 + ResNet-50 pooled 2048 = **2816**
  (`test_native_dims_concat_to_2816`이 가중치 다운로드 없이 이를 고정). `proj_dim`을 주면 dim을 맞출 수 있어
  Epic 2의 공통 768 hidden으로 정렬하는 경로도 열려 있다.
- **fusion head 기본값**: config에 `model.head.hidden_dims`가 **없으면** `[512]`(2816→512→2 MLP),
  명시적으로 `null`을 주면 단일 Linear. 단일 모달 baseline(기본 단일 Linear)과 다른 기본값인데,
  concat 2816-dim에서는 비선형 head가 late fusion의 최소 요건이라고 판단했다.
- **학습 모드 스위치(AC2)**: `model.freeze_encoders`가 개별 인코더의 `model.{text,image}.freeze`보다
  **나중에** 적용된다(상위 스위치). `true`면 두 backbone 모두 `requires_grad=False` + `train()`에서도
  eval 유지(BN/Dropout 고정), `param_groups()`가 frozen 파라미터를 optimizer에서 제외한다.
  shipped config 기본값은 (a) frozen — 단일 GPU 제약(NFR1) 대응.
- **인코더 checkpoint 재사용**: `load_encoders(text_path=, image_path=)`는 Story 1.3
  `SingleModalClassifier.save_encoder()` 산출물(`{"encoder_state": ...}`)과 raw state_dict를 모두 받는다.
  config `model.text.checkpoint` / `model.image.checkpoint`로도 지정 가능하며, freeze 적용 **전에** 로딩한다.
- **단일 추론은 모델 비의존 함수로**: `predict_single(model, image_path, text, tokenizer=...)`는
  `predict_proba`가 아니라 `model(batch)`만 요구하므로 `text_only`/`image_only`/`late_fusion` 어디에도 쓴다.
  전처리는 `FakedditDataset`과 동일한 `build_image_transform`/tokenizer 규약을 재사용해 학습-추론 불일치를 막았고,
  반환값에 `elapsed_sec`(전처리+forward)를 넣어 NFR2를 상시 측정한다.
- **ablation 표는 "비어 있음"을 드러내도록 설계**: 학습하지 않은 실험은 지표 NaN + `status=missing`으로
  남는다. 실수로 빈 표를 "결과"로 오독하는 것을 막고, `f1_comparison()`은 비교 불가 시 `improved=None`을 낸다.
  Epic 2+ 행은 `AblationRow` 추가 또는 `--row LABEL=EXP_NAME`으로 확장한다.

#### 검증 결과

- **pytest 183개 전부 통과** (Story 1.1~1.3의 136개 회귀 없음 + 신규 47개), 네트워크 접근 0회.
  신규 커버리지: concat dim(2816 포함)·forward shape, 두 모달이 모두 출력에 영향을 주는지,
  head hidden dim 변형, freeze/unfreeze·frozen 인코더 무-grad·optimizer step 후 가중치 불변,
  param group 분리, config 빌더(기본/`null` head, freeze 플래그, checkpoint 경로), 기존 등록 회귀,
  A1~A3 인터페이스 동일성, Story 1.3 인코더 checkpoint round-trip, 단일 추론(확률 범위·모드 복원·결정성·
  tensor 이미지·tokenizer 미주입), 합성 Fakeddit `modality=both` + 공통 Trainer 1 epoch 학습·test 평가,
  동일 seed 재현성, ablation 취합(missing 표시·개선/미개선 판정·csv/markdown 생성·CLI main).
- **CLI 스모크**(합성 `dummy` 데이터셋, CPU): `python scripts/train.py --config configs/base.yaml --set
  model.name=late_fusion model.freeze_encoders=true train.head_lr=0.001 ...` 1 epoch 학습
  (val_f1 0.875) → `scripts/evaluate.py --split test`로 `metrics_test.json` 생성 → `scripts/collect_ablation.py`로
  `outputs/ablation_table.csv` + `docs/bmad/results/ablation.md` 생성 확인. 스모크 산출물은 정리했다.

#### 실데이터/실가중치 확보 후 남은 작업 (미검증 항목)

1. **AC3/AC4/AC5의 실수치 미기록** — `docs/bmad/results/ablation.md`의 3개 행은 현재 모두 `missing`이다.
   실데이터 학습 후 세 config를 학습→`evaluate --split test`→`collect_ablation.py` 순으로 재실행해 채울 것.
2. **AC4의 "개선 여부 문서화"** — late fusion F1이 단일 모달을 실제로 넘는지는 실행해야 알 수 있다.
   `f1_comparison()`이 자동 판정하지만, **개선되지 않았을 때의 원인 분석(모달 불균형, frozen 여부, head 용량)**은
   사람이 `--notes`로 표에 남겨야 한다.
3. **AC2 (b) end-to-end 모드 실학습 미실행** — 분기·param group·grad 흐름은 테스트로 고정했으나
   실제 fine-tuning의 수렴/메모리는 GPU에서 확인이 필요하다.
4. **AC6의 "GPU 기준 수 초 이내"** — CPU + 소형 랜덤 인코더에서만 측정했다(테스트는 5초 상한만 검사).
   실 BERT-base + ResNet-50 + GPU에서 재측정 필요.
5. **AC7 재현성은 모델 초기화·forward 수준에서만 확인** — 동일 seed 2회 학습의 *metric* 일치는 실데이터 후 재확인.

#### Epic 1 잔여 검증 항목 (1.1~1.4 통합)

실데이터(Fakeddit 원본)와 사전학습 가중치(HuggingFace/torchvision)가 **모두 없는 오프라인 환경**에서
Epic 1을 완료했다. 아래는 Epic 2 착수 전 또는 데이터 확보 즉시 처리해야 할 항목을 한데 모은 것이다.

**A. 데이터 (Story 1.2)**

1. `2_way_label` 실제 인코딩 재확인 — `FAKEDDIT_2WAY_TO_INTERNAL = {1: 0, 0: 1}`(반전)이 배포본과 맞는지
   눈으로 대조. 다르면 상수 + `test_labels.py` + `docs/DATA.md`를 함께 수정. **라벨 반전은 전 실험을 무효화한다.**
2. 실 manifest/분할 생성 및 통계 기록(`processed/split_report.json`) — AC3의 "샘플 수·label 분포 문서화" 완료 조건.
   공식 분할 tsv 파일명이 다르면 `OFFICIAL_SPLIT_FILES` 조정.
3. 금융 subset 실제 규모 측정 (FR11 최대 리스크) — 부족하면 키워드 확장 또는 "금융 subset 평가만" 전략으로 축소.
4. 이미지 다운로드 성공률/손상률 실측 — 확보 가능한 멀티모달 샘플 수가 여기서 결정된다.

**B. 사전학습 가중치 로딩 (Story 1.3, 1.4)**

5. `AutoModel.from_pretrained("bert-base-uncased")` / torchvision `weights="DEFAULT"` 경로 최초 1회 실행 —
   hidden 768 / pooled 2048 가정과 concat 2816이 실가중치에서도 유지되는지 확인.
6. 실 tokenizer(`bert-base-uncased`) 경로 검증 — 현재는 `conftest.FakeTokenizer`로 인터페이스만 검증했다.
   `data.tokenizer.max_length: 128` 절단 비율도 실 텍스트에서 점검할 것.

**C. 학습 결과 (Story 1.3, 1.4 — ablation table의 실제 내용)**

7. 세 config(`bert_only` / `resnet_only` / `late_fusion`) 실학습 + `evaluate --split test` 실행 →
   `collect_ablation.py`로 `docs/bmad/results/ablation.md` 3행 채우기 (현재 전부 `missing`).
8. Story 1.3 AC4의 **test AUROC ≥ 0.60** 달성 여부 확인 — 미달 시 원인(lr, frozen 여부, subsample 규모)을
   해당 스토리 Dev Agent Record에 문서화.
9. Story 1.4 AC4의 **F1 개선 여부** 판정 및 코멘트 — 이 수치가 Epic 2~4 개선 주장의 기준선이므로,
   확정 후 분할/seed/전처리/metric 코드를 **동결**하고 checkpoint·config를 `outputs/`에 보존할 것.
10. late fusion (b) end-to-end 모드 학습 및 (a) frozen 대비 비교.

**D. 환경/성능 (Story 1.1, 1.3, 1.4)**

11. AMP(`train.amp: true`) 경로 — CUDA에서만 활성화되므로 GPU에서 별도 확인. 단일 GPU 학습 시간·메모리 미측정.
12. 재현성(NFR3)의 *metric* 수준 검증 — 현재는 dummy 데이터 2회 실행 일치 + 모델 초기화/forward 일치까지만
    확인했다. 실데이터 2회 학습으로 최종 확인 필요.
13. NFR2 추론 지연 — 실 BERT-base + ResNet-50 + GPU에서 단일 샘플 `predict` 시간 재측정.

**검증된 것 (재확인 불필요)**: config 상속/오버라이드, seed 유틸, Trainer 루프·checkpoint·early stopping,
metric 계산, 전처리/분할/필터 로직, Dataset·collate·subsampling, 인코더/분류기 조립과 계약,
freeze/param group, 인코더 checkpoint 재사용, 단일 추론 인터페이스, ablation 취합 —
모두 pytest 183개로 네트워크 없이 고정되어 있다.

#### Change Log

- 2026-08-10 (PO Sarah): AC5에 실험 메타데이터(config/git hash/split 해시/환경) 기록 요건 추가, AC7에 실데이터 2회 학습 metric 일치 실증 요건 명시, Dev Notes에 checkpoint 보관 정책 확정. Status를 Ready for Review → Changes Requested로 조정.

---

## Story 1.5: FinFact-Eval 수집 착수 게이트

### Status

Approved

### Story

As a **프로젝트 오너/과제 수행자**,
I want **FinFact-Eval(자체 금융 평가셋)의 출처·라이선스 정책을 확정하고 시드 20건을 실제로 수집한 상태**,
so that **Epic 6이 번호 때문에 마지막으로 밀려 Epic 5 ablation의 `custom_fin` 평가 열이 통째로 비는 사고를 구조적으로 막는다**.

### Acceptance Criteria

1. 수집 출처·라이선스 정책 문서(`docs/data/finfact_eval_sources.md`)가 작성되고 확정된다: 허용 출처 목록, 출처별 재배포 허용 범위(본문 전문/발췌만/이미지 가능 여부), 그리고 금지 사항(자동 크롤링·스크래핑 스크립트 작성, robots.txt 무시, 로그인 벽 우회)이 명문화된다 (NFR5).
2. 샘플 기록 스키마가 확정된다: `id, source_url, publisher, license, collected_at, image_sha256, title, body_excerpt, image_path, lang` — Epic 6 Story 6.1의 `sources.jsonl`과 동일 스키마여야 하며, 이 스토리에서 확정한 스키마가 단일 소스다.
3. **시드 샘플 20건**이 위 정책에 따라 수동 수집되어 `data/finfact_eval/raw/`에 배치되고 `sources.jsonl`에 기록된다. 20건은 Real 기준이며, 금융 entity 2종 이상 + 관련 대표 이미지 요건(Story 6.1 AC3)을 만족한다.
4. 검증 스크립트(`scripts/validate_eval_sources.py`)가 필수 필드 누락·중복 id·이미지 해시 불일치를 검출하고, 시드 20건이 이를 통과한다.
5. 20건 수집에 소요된 실제 시간(건당 분)이 측정·기록되어, Epic 6의 300건 목표가 3~7주차 일정 안에 가능한지 판단 근거가 된다. 추정 총 소요가 가용 인력 시간을 초과하면 **규모 축소 제안을 오너 결정 안건으로 올린다**(수집 방식을 바꾸지 않는다 — NFR5).
6. 자동 수집 스크립트가 작성되지 않았음이 리포지토리 상태로 확인 가능하다. 허용되는 보조 도구는 사람이 확정한 URL 1건을 받는 단건 다운로드 유틸(`scripts/fetch_eval_asset.py --url ... --id ...`)까지다.

### Tasks / Subtasks

- [ ] 출처·라이선스 정책 확정 (AC: 1)
  - [ ] 재사용 허용 후보 조사(공개 라이선스 뉴스 아카이브, 기관 보도자료, 공시, CC 이미지 아카이브)
  - [ ] 애매한 출처는 "발췌 + URL만" 보수적 기준 채택, 표로 정리
- [ ] 기록 스키마 및 검증기 (AC: 2, 4)
  - [ ] `sources.jsonl` 스키마 확정 (Epic 6과 공유)
  - [ ] `scripts/validate_eval_sources.py` + `tests/test_eval_sources.py`
- [ ] 시드 20건 수집 (AC: 3, 5, 6)
  - [ ] 수동 선별·다운로드, 건당 소요 시간 기록
  - [ ] 단건 다운로드 유틸만 사용, 배치 크롤링 금지
- [ ] 페이스 리포트 (AC: 5)
  - [ ] 건당 평균 시간 × 300건 추정치 산출, 일정 충돌 시 오너 결정 안건 작성

### Dev Notes

- **이 스토리의 존재 이유는 순서 강제다**: Epic 6은 번호가 6이지만 실행은 3~7주차로 Epic 2~4와 병행한다. "문서에 순서 주석을 달아 두면 지켜진다"는 가정은 성립하지 않으므로, Epic 1 안에 착수 앵커를 두고 Story 2.2 완료 조건과 Story 5.1 착수 전제조건으로 이중 게이트를 건다.
- **Epic 6과의 관계**: 본 스토리는 Story 6.1의 **선행 부분집합**이다. 여기서 확정한 정책 문서와 스키마를 6.1이 그대로 승계하며, 시드 20건은 6.1의 Real 150건에 포함 계상한다(중복 수집 금지).
- **NFR5가 최우선 제약**: 수집 속도가 문제라면 규모를 줄일지언정 수집 방식(크롤링 금지)은 바꾸지 않는다.
- **언어 정책**: 주 실험은 영어이므로 시드도 영어 기본. 한국어 샘플은 데모용으로 `lang: ko` 태그로만 보관하고 주 지표 집계에서 제외한다.
- **핵심 가설 검증과의 역할 분리(미팅 확정)**: A5 대비 A6 개선 여부는 Fakeddit 금융 subset(5k~20k, 3-seed)에서 판정한다. FinFact-Eval은 통계 검정력이 아니라 **FR16 matching 판정 정확도의 유일한 정답 소스 + 정성 근거 + 데모**를 담당한다 — 이 역할 차이를 수집 설계에 반영할 것(mismatch 유형 annotation 품질 > 건수).

### Testing

- Unit: `validate_eval_sources.py` — 필드 누락/중복 id/해시 불일치 fixture 검출.
- 수동: 시드 20건 전수에 대해 `source_url` 접속 → 기사·이미지 일치 및 라이선스 표기 재확인.

#### Change Log

- 2026-08-10 (PO Sarah): PM John 제안에 따라 Epic 6 착수 앵커로 신설. Status Approved (즉시 착수 가능).
