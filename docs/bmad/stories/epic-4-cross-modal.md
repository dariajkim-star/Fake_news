# Epic 4: Cross-modal Consistency & 최종 모델 — User Stories

> 작성자: Bob (BMAD Scrum Master) · 작성일: 2026-08-10
> 근거 문서: `docs/bmad/prd.md` (FR3, FR6, FR7, FR8, FR12), `docs/ARCHITECTURE.md` (§2.1.2, §2.3, §2.4, §3)

---

## Story 4.1: Visual Entity Recognition 모듈

### Status

Approved

### Story

**As a** ML 엔지니어,
**I want** Object Detection이 검출한 bbox crop을 구체적 entity 값(인물 이름, 기업명, 수치/날짜)으로 인식하는 Visual Entity Recognition 모듈,
**so that** 이미지 속 객체를 텍스트 entity와 대조 가능한 구조화된 visual entity로 변환할 수 있다.

### Acceptance Criteria

1. `visual_entity` 모듈이 detection 출력(`region_id, class_name, bbox, conf, crop`)을 입력받아 ARCHITECTURE.md §2.1.2 출력 스키마(`region_id, entity_type, value, score, source, embedding, raw`)의 visual entity 리스트를 반환한다.
2. PERSON crop: 얼굴 인식(InsightFace ArcFace embedding — architecture Tech Stack 확정 사항) + 사전 구축한 유명 인물 gallery와의 cosine similarity로 인물 이름을 예측한다 (`source: "face_recognition"`).
3. LOGO crop: CLIP zero-shot 분류(후보 기업명 프롬프트 예: `"a logo of {ORG}"`)로 기업명을 예측하고, top-k candidates와 score를 `raw.candidates`에 기록한다 (`source: "clip_zero_shot"`).
4. CHART/DOCUMENT/TEXT_REGION crop: OCR(PaddleOCR 우선, 대안 EasyOCR — architecture Tech Stack 준수)로 텍스트를 추출하고, 정규식 기반 파서로 `MONEY`(₩/조/억/$/%), `DATE`, `ORG` 등 다중 entity를 파생한다 (`source: "ocr"`). 수치는 `normalized` 필드로 정규화한다.
5. 모든 인식 결과에 confidence score가 있고, config의 threshold(`score < τ_rec`) 미달 시 `value: null, source: "unrecognized"`로 출력하여 matching 단계의 UNKNOWN 후보가 된다 (FR12).
6. 각 visual entity에 CLIP image embedding(`float32[512]`)이 포함되어 matching 단계에서 embedding 유사도 계산에 사용 가능하다.
7. 모듈은 detection 없이도 단독 실행 가능한 CLI/함수 인터페이스(`recognize(image, detections) -> List[VisualEntity]` — architecture Components §3 시그니처와 동일 인자 순서)를 제공하며, config(yaml)로 threshold·후보 리스트·모델 선택을 제어한다.

### Tasks / Subtasks

- [ ] Task 1: 모듈 스캐폴딩 및 스키마 정의 (AC: 1, 7)
  - [ ] `src/vision/entity_recognizer.py` 생성 (architecture Source Tree 준수), `VisualEntity` dataclass/pydantic 스키마 정의 (ARCHITECTURE §2.1.2 준수)
  - [ ] `configs/visual_entity.yaml`에 threshold, 후보 ORG/PERSON 리스트 경로, OCR 엔진 선택 추가
  - [ ] `recognize(detections, image)` 진입점 함수 + class_name→recognizer 라우팅 구현
- [ ] Task 2: PERSON 얼굴/인물 인식 (AC: 2, 5)
  - [ ] 얼굴 detection+embedding 모델 통합 (InsightFace ArcFace — architecture Tech Stack 확정, facenet-pytorch는 설치 이슈 시 fallback으로만)
  - [ ] 금융 도메인 유명 인물 gallery 구축 스크립트 (인물 리스트 + 대표 이미지 → embedding 저장, `assets/face_gallery.pkl`)
  - [ ] gallery cosine similarity 기반 top-1 예측 + threshold 미달 시 unrecognized 처리
- [ ] Task 3: LOGO CLIP zero-shot 분류 (AC: 3, 5, 6)
  - [ ] CLIP(ViT-B/32) 로드 및 crop 전처리(224x224) 유틸
  - [ ] 후보 기업명 리스트(금융/테크 주요 기업, `assets/org_candidates.txt`) + 프롬프트 템플릿으로 zero-shot 분류
  - [ ] top-k candidates/score 기록, threshold 처리, image embedding 반환
- [ ] Task 4: OCR 기반 수치/날짜/텍스트 추출 (AC: 4, 5)
  - [ ] PaddleOCR(우선, 대안 EasyOCR) 통합, CHART/DOCUMENT/TEXT_REGION crop에 적용
  - [ ] 정규식 파서: MONEY(₩, 조, 억, $, %, 쉼표 숫자), DATE(YYYY-MM-DD, N월 N일 등), ORG 키워드 → entity 파생 + `normalized` 값 산출
  - [ ] OCR confidence 전파 및 threshold 처리
- [ ] Task 5: 단위 테스트 및 샘플 검증 (AC: 1~7)
  - [ ] 정규식 파서/threshold 로직/스키마 유효성 unit test
  - [ ] 대표 샘플 이미지 10장 이상에 대한 수동 검증 노트북/스크립트, 인식 정확도 기록

### Dev Notes

- **입출력 스키마**: ARCHITECTURE.md §2.1.2를 단일 소스로 삼을 것. `entity_type` 매핑: `PERSON→PERSON`, `LOGO→ORG`, `PRODUCT→PRODUCT`, `CHART/DOCUMENT/TEXT_REGION→OCR` 파생(MONEY/DATE/ORG 다중 가능). 하나의 region에서 여러 entity가 파생될 수 있으므로 반환 타입은 flat list.
- **CLIP zero-shot**: `openai/clip-vit-base-patch32` (HuggingFace) 사용, 후보 리스트는 config로 교체 가능해야 Fakeddit 영어 도메인/한국어 금융 도메인 전환이 쉬움. 프롬프트 앙상블(`"a logo of {}"`, `"the {} company logo"`) 평균 허용.
- **얼굴 인식**: gallery는 수십 명 규모로 시작(단일 GPU 제약, NFR1). gallery에 없는 인물은 당연히 unrecognized → UNKNOWN이며 이는 정상 동작.
- **오류 전파 제한(FR12)**: threshold `τ_rec`는 유형별로 분리 config (face: 0.5 — architecture Components §3의 "InsightFace 인물 DB 매칭 threshold 0.5" 준수, clip: 0.5, ocr: 0.4 초기값) — Story 4.2의 UNKNOWN 판정과 연동. 조정 시 architecture 문서 갱신 필요.
- **성능**: 모든 모델 frozen(학습 없음). 배치 추론 지원하여 단일 샘플 수 초 이내(NFR2).
- **Region Encoder(Epic 2 Story 2.3)와 CLIP 인스턴스 공유**하여 메모리 절약 — 동일 ViT-B/32 재사용.

### Testing

- Unit: 정규식 파서(MONEY/DATE 정규화 케이스 표), threshold→unrecognized 분기, 스키마 검증. `pytest tests/test_visual_entity.py`
- Integration: detection mock 입력 → 전체 recognize 파이프라인 스모크 테스트
- 수동: 대표 샘플(로고/인물/차트 각 포함) 정성 평가 및 결과 기록

---

## Story 4.2: Cross-modal Matching (Entity Alignment)

### Status

Approved

### Story

**As a** ML 엔지니어,
**I want** visual entity 집합과 textual entity 집합을 정렬하여 각 pair를 MATCH / MISMATCH / UNKNOWN으로 판정하는 matching 모듈,
**so that** "이미지 속 인물/로고와 기사 주장이 불일치"하는 object-level 증거를 산출할 수 있다 (FR6).

### Acceptance Criteria

1. `matching` 모듈이 visual entities(Story 4.1 출력)와 textual entities/events(Epic 3 출력)를 입력받아 ARCHITECTURE.md §2.3 출력 스키마(`alignments, unmatched_visual, unmatched_textual`)를 반환한다.
2. type-compatible pair(PERSON↔PERSON, ORG↔ORG, PRODUCT↔PRODUCT, MONEY↔MONEY, DATE↔DATE)에 대해서만 similarity를 계산한다.
3. MONEY/DATE는 normalized 값의 exact/허용오차(tol) 비교로 sim ∈ {0, 1}, 그 외 유형은 공통 공간 projection 후 cosine similarity에 name similarity(별칭 사전 + fuzzy string matching)를 max 결합한다.
4. score matrix에 대해 Hungarian matching(또는 1:1 greedy)으로 최적 정렬을 수행한다.
5. 판정 로직이 ARCHITECTURE.md §2.3 표를 따른다: type 일치 ∧ sim ≥ τ_match(0.7) → MATCH; type 일치 ∧ sim < τ_mismatch(0.4) ∧ 양쪽 confidence ≥ 0.6 → MISMATCH; 그 외(중간 유사도, 인식 실패, 대응 entity 없음) → UNKNOWN. threshold는 모두 config화.
6. 텍스트에만 존재하는 entity는 UNKNOWN("무증거")으로, MISMATCH와 명확히 구분된다.
7. matching 결과로부터 consistency feature vector `c ∈ R^11`(n_match, n_mismatch, n_unknown, mean_sim_matched, min_sim, 유형별 mismatch flag 5-dim, global_clip_sim)을 계산하는 함수를 제공한다.
8. rule 기반 판정 로직에 대한 unit test가 edge case(빈 집합, 전부 unrecognized, 동률 similarity)를 포함해 통과한다.

### Tasks / Subtasks

- [ ] Task 1: 모듈 및 스키마 (AC: 1)
  - [ ] `src/matching/` 패키지, `Alignment`/`MatchResult` 스키마 정의
  - [ ] `configs/matching.yaml`: τ_match=0.7, τ_mismatch=0.4, conf_min=0.6, tol(MONEY 상대오차, DATE 일 단위) 정의
- [ ] Task 2: Similarity 계산 (AC: 2, 3)
  - [ ] type compatibility 매트릭스 구현
  - [ ] projection layer: CLIP 512-dim ↔ BERT 768-dim → 공통 공간 Linear projection (초기: CLIP text encoder로 textual entity string embedding하여 동일 공간 사용, 대안으로 학습 projection)
  - [ ] name similarity: 별칭 사전(`assets/alias.json`, 예: "엔비디아"↔"NVIDIA") + `rapidfuzz` fuzzy ratio, embedding sim과 max 결합
  - [ ] MONEY/DATE normalized 비교기(tol 적용)
- [ ] Task 3: Assignment 및 판정 (AC: 4, 5, 6)
  - [ ] `scipy.optimize.linear_sum_assignment` 기반 Hungarian matching (greedy fallback 옵션)
  - [ ] verdict 판정 함수(표 로직) + unmatched entity UNKNOWN 처리
- [ ] Task 4: Consistency feature 계산 (AC: 7)
  - [ ] `compute_consistency_vector(match_result, global_clip_sim) -> np.ndarray[11]` 구현 (개수 정규화 포함)
- [ ] Task 5: 테스트 (AC: 8)
  - [ ] 판정 표 전 분기 unit test, edge case test
  - [ ] 합성 시나리오 test: "로고 NVIDIA vs 텍스트 TSMC → MISMATCH", "텍스트 전용 MONEY → UNKNOWN"

### Dev Notes

- **핵심 로직 소스**: ARCHITECTURE.md §2.3의 pseudo-code와 판정 표를 그대로 구현. 이 rule 기반 부분은 PRD Testing Requirements에서 unit test 최우선 대상으로 명시됨.
- **soft matching 철학(FR12)**: 인식 실패나 애매한 similarity를 억지로 MATCH/MISMATCH로 밀지 않고 UNKNOWN으로 흡수 — MISMATCH는 "양쪽 모두 confident한데 다르다"일 때만 발화하는 강한 신호여야 함. τ_match/τ_mismatch 사이 gray zone이 존재하는 것이 의도된 설계.
- **embedding 공간 정렬**: 가장 단순한 안은 textual entity string을 CLIP text encoder로 embedding하여 visual CLIP embedding과 직접 cosine — 별도 학습 없이 동작(단일 GPU 제약에 유리). 학습 projection(`proj_v`, `proj_t`)은 Story 4.3 fusion 학습 시 함께 fine-tuning하는 확장 옵션으로 남김.
- **consistency vector `c` 정의(§2.4)**: `[n_match, n_mismatch, n_unknown, mean_sim_matched, min_sim, mismatch_flag_{PERSON,ORG,PRODUCT,MONEY,DATE}, global_clip_sim]` → R^11. 개수는 전체 pair 수로 정규화. 빈 결과일 때 기본값(0, sim 통계는 0.5 등) 명시적으로 정의할 것.
- **global_clip_sim**: 전체 이미지 CLIP embedding vs 전체 텍스트 CLIP text embedding cosine — Story 4.1의 CLIP 인스턴스 재사용.

### Testing

- Unit: verdict 표 전 분기, normalized 비교기(tol 경계값), consistency vector 산출, 빈 집합/동률 edge case. `pytest tests/test_matching.py`
- Integration: Story 4.1 + Epic 3 실제 출력 연결 스모크 테스트
- 정성: 합성 MISMATCH 샘플(이미지-텍스트 entity 교차 조작)로 판정 sanity check

---

## Story 4.3: Consistency Feature 결합 최종 모델 학습

### Status

Approved

### Story

**As a** ML 엔지니어,
**I want** cross-attention fusion 출력에 consistency feature vector를 결합한 최종 fake 분류기를 학습,
**so that** ablation table에서 "+Entity consistency" 구성이 F1을 추가 개선함을 입증할 수 있다 (FR7, FR10).

### Acceptance Criteria

1. Fusion 모델이 ARCHITECTURE.md §2.4 구조를 구현한다: 양방향 cross-attention 2 stream(text→image, image→text; heads=8, d=768, 2 layers, FFN 3072) → mean pooling → `h_t2v, h_v2t ∈ R^768`.
2. Consistency vector `c ∈ R^11` → `Linear(11→64)` → `c' ∈ R^64`가 fusion 출력과 concat되어 `z ∈ R^1600` (768+768+64)을 구성한다.
3. 최종 MLP head: `Linear(1600→512)→GELU→Dropout(0.3)→Linear(512→128)→GELU→Linear(128→2)→softmax`, Cross-Entropy loss(+class weight)로 학습된다.
4. 학습 시 YOLOv8/OCR/Face는 frozen이고, 학습 대상은 projection layers, cross-attention, consistency Linear, MLP head로 제한된다 (NFR1).
5. 학습/평가는 Epic 1의 공통 루프·config·seed 체계를 사용하며, held-out test set에서 Accuracy/Precision/Recall/F1/AUROC를 산출한다 (NFR3, NFR4).
6. Epic 2의 "+Object Detection" 구성(consistency feature 없음, `z ∈ R^1536`) 대비 동일 데이터·seed 조건에서 "+Entity consistency" 결과가 ablation 결과 파일에 기록된다 — consistency feature는 config flag로 on/off 가능해야 한다 (NFR6).
7. 학습 데이터 전처리로 matching 결과(consistency vector)를 사전 계산·캐싱하는 스크립트가 제공된다 (매 epoch 재계산 방지).

### Tasks / Subtasks

- [ ] Task 1: Fusion 모델 확장 (AC: 1, 2, 3)
  - [ ] Epic 2의 cross-attention fusion 모델에 consistency branch(`Linear(11→64)`) 추가
  - [ ] `use_consistency: bool` config flag로 concat 차원 분기(1536 vs 1600)
  - [ ] MLP head 차원 수정 및 forward 검증(§3.3 텐서 차원 표 준수)
- [ ] Task 2: 학습 데이터 사전 계산 (AC: 7)
  - [ ] train/val/test 전체에 대해 detection→visual entity→matching 실행, consistency vector + region feature 캐시 저장 스크립트 (`scripts/precompute_consistency.py`, npz/parquet)
  - [ ] 캐시 무결성 체크(샘플 수, seed, config hash)
- [ ] Task 3: 학습 실행 (AC: 4, 5)
  - [ ] `configs/exp_entity_consistency.yaml` 작성(lr, batch, epoch, class weight, frozen 모듈 명시)
  - [ ] 학습 스크립트 실행, 조기 종료/best checkpoint 저장, 학습 곡선 로깅
- [ ] Task 4: Ablation 기록 (AC: 6)
  - [ ] "+Object Detection"(flag off)와 "+Entity consistency"(flag on) 동일 seed 3회 반복 평가
  - [ ] `results/ablation.csv`에 평균±표준편차 기록, F1 개선 여부 요약
- [ ] Task 5: 테스트 (AC: 1, 2, 3)
  - [ ] 모델 forward shape unit test (batch 포함, R 가변 padding/mask)
  - [ ] flag on/off 시 파라미터 수·출력 차원 검증 test

### Dev Notes

- **Epic 2 Story 2.4 모델을 확장**하는 story — 새 모델을 만들지 말고 기존 fusion 클래스에 consistency branch를 조건부 추가하여 ablation 성립(NFR6). region feature는 `V' ∈ R^{(R+1)×768}` (CLIP 512 → Linear 768, global image token 포함, R ≤ 16 padding+attention mask).
- **2-stage 학습 전략(§4)**: Stage 1에서 각 모듈(YOLO, NER, RE)은 이미 개별 학습 완료 상태(Epic 2·3 산출물). 본 story는 Stage 2 — frozen backbone 위 fusion end-to-end fine-tuning. BERT/CLIP encoder는 frozen 또는 마지막 layer만 unfreeze(GPU 메모리에 따라 config로 결정).
- **consistency vector는 gradient가 흐르지 않는 수작업 feature** — 사전 계산·캐싱이 학습 속도에 결정적. 캐시 파일에 config hash를 심어 threshold 변경 시 재계산 강제.
- **class weight**: Fakeddit 금융 subset의 label 불균형에 대응, `n_samples / (n_classes * n_class_samples)` 기본식.
- **핵심 리스크(PRD 리스크 2·3)**: Visual Entity Recognition 노이즈로 consistency feature가 무정보일 가능성 — F1 개선이 미달하면 UNKNOWN 비율·mismatch flag 분포를 함께 기록하여 Epic 5 error analysis 입력으로 넘길 것.
- **재현성(NFR3)**: seed 고정(3개 seed), config yaml 전체를 결과 디렉터리에 복사 저장.

### Testing

- Unit: forward shape(가변 R, mask), flag 분기, consistency Linear 파라미터 검증. `pytest tests/test_fusion.py`
- 학습 검증: 소규모 subset overfit sanity check(loss → 0 확인)
- 평가: held-out test Accuracy/P/R/F1/AUROC + ablation 대조 기록

---

## Story 4.4: End-to-end 추론 파이프라인 통합

### Status

Approved

### Story

**As a** 데모 개발자(Epic 5),
**I want** 이미지+텍스트 입력 한 번으로 fake score와 mismatch entity 근거를 구조화 JSON으로 반환하는 단일 추론 파이프라인,
**so that** Gradio/Streamlit 데모와 평가 스크립트가 동일한 진입점을 호출할 수 있다 (FR1, FR8).

### Acceptance Criteria

1. `pipeline` 모듈이 ARCHITECTURE.md §3.1 시퀀스를 구현한다: vision branch(YOLOv8 → Visual Entity Recognition + region feature)와 text branch(NER/RE + token feature)를 실행 후 matching → fusion 순으로 결합한다.
2. 출력이 ARCHITECTURE.md §3.2 최종 JSON 스키마를 준수한다: `fake_probability, label, threshold, evidence[](kind: entity_mismatch/entity_match/no_visual_evidence, visual/textual 상세, sim, verdict, weight), scores(global_clip_similarity, n_match, n_mismatch, n_unknown)`.
3. evidence의 `weight`는 fusion cross-attention weight와 consistency feature 기여도 기반으로 산출되어 중요도 내림차순 정렬된다.
4. 단일 샘플 end-to-end 추론이 GPU 기준 수 초 이내에 완료된다 (NFR2) — 모델은 초기화 시 1회 로드 후 재사용.
5. `FinFactPipeline(config).predict(image, title, body) -> dict` 형태의 Python API와 `python -m src.pipeline --image ... --text ...` CLI를 모두 제공한다 (architecture Source Tree의 `src/pipeline.py` 단일 모듈 준수).
6. 개별 모듈 실패(detection 0건, OCR 실패, NER 무결과 등) 시에도 파이프라인이 예외 없이 degrade하여 결과를 반환한다 — detection 0건이면 global feature만으로 fusion, evidence는 UNKNOWN 중심.
7. 대표 샘플 세트(진짜/가짜/MISMATCH 포함 5개 이상)에 대한 end-to-end 스모크 테스트가 통과한다.

### Tasks / Subtasks

- [ ] Task 1: 파이프라인 클래스 (AC: 1, 4, 5)
  - [ ] `src/pipeline.py` (architecture Source Tree 준수): 모든 모듈(YOLO, VisualEntity, NER/RE, Matcher, Fusion) lazy 1회 로드 + `predict()` 구현
  - [ ] vision/text branch 실행 (동기 순차로 우선 구현, §3.1의 par는 최적화 옵션)
  - [ ] CLI entry point + config 경로 인자
- [ ] Task 2: Evidence JSON 조립 (AC: 2, 3)
  - [ ] alignments → evidence 항목 변환(MATCH/MISMATCH/UNKNOWN → kind 매핑, bbox/span 포함)
  - [ ] weight 산출: 해당 region/token의 cross-attention weight 평균 × consistency 기여(mismatch flag gradient 대용으로 verdict별 가중) — 산식은 Dev Notes 참조
  - [ ] JSON schema 검증(pydantic) 및 정렬
- [ ] Task 3: Graceful degradation (AC: 6)
  - [ ] 각 branch try/except + 빈 결과 기본값 정의(빈 detection → global token만, 빈 NER → UNKNOWN)
  - [ ] degrade 발생 시 결과 JSON에 `warnings` 필드 기록 (§3.2 Prediction 스키마의 optional 확장 필드 — 스키마 필수 키는 항상 유지)
- [ ] Task 4: 성능 및 스모크 테스트 (AC: 4, 7)
  - [ ] 대표 샘플 5+개 fixture 구성(진짜 2, 가짜 2, 인위적 MISMATCH 1)
  - [ ] end-to-end 스모크 테스트 + 추론 시간 측정 로그
  - [ ] Epic 5 데모/평가 스크립트에서 사용할 사용 예시 docstring 정리

### Tasks 참고

Epic 5 (Story 5.3 데모)가 이 파이프라인을 그대로 소비하므로, 인터페이스 변경은 이 story에서 확정한다.

### Dev Notes

- **모듈 wiring만 하는 story** — 새 모델 학습 없음. Epic 2(detection/region encoder), Epic 3(NER/RE), Story 4.1~4.3 산출물을 조립. 각 모듈은 NFR6에 따라 독립 인터페이스를 가지므로 pipeline은 어댑터 역할.
- **최종 JSON 스키마는 ARCHITECTURE.md §3.2가 단일 소스** — Epic 5 데모의 bbox 하이라이트(MATCH 녹색/MISMATCH 적색/UNKNOWN 회색)와 텍스트 span 하이라이트가 이 스키마를 직접 렌더링하므로 `bbox`, `span` 좌표는 원본 이미지/텍스트 좌표계로 반환할 것 (letterbox 640 좌표 → 원본 역변환 필요).
- **weight 산식(초기 구현)**: `weight = α · attn(region↔span 평균 cross-attention) + β · verdict_prior` (MISMATCH 0.5, MATCH 0.1, UNKNOWN 0.05), α=β=0.5 후 정규화. attention weight는 fusion forward에서 `output_attentions=True`로 추출. 정교화(gradient 기반 기여도)는 Epic 5 error analysis 이후 개선 항목.
- **체크포인트 로딩**: Story 4.3 best checkpoint 경로를 config에 명시. `use_consistency=True` 모델을 기본으로 하되, ablation 비교용으로 checkpoint 교체 가능하게.
- **성능(NFR2)**: 모델 로드는 생성자에서 1회, `torch.inference_mode()` 사용, YOLO/CLIP/BERT 순차 실행으로도 단일 GPU 수 초 내 충분. OCR이 최대 병목 — crop 수 상한(R ≤ 16)으로 제어.

### Testing

- Unit: evidence 변환 로직, bbox 좌표 역변환, degrade 분기, JSON schema 검증. `pytest tests/test_pipeline.py`
- Integration: 대표 샘플 fixture end-to-end 스모크(라벨 방향성 + 스키마 + 시간 측정)
- 수동: 데모 시나리오 리허설 — MISMATCH 샘플에서 evidence 최상위에 mismatch가 노출되는지 확인
