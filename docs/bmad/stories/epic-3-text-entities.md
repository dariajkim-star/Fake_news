# Epic 3: Text Entity Extraction — User Stories

> 작성자: Bob (BMAD Scrum Master) · 작성일: 2026-08-10 · 기반 문서: docs/bmad/prd.md (v1.0), docs/bmad/architecture.md (v1.0)
> PO 검토: Sarah · 2026-08-10 · 3.1/3.2/3.3 Approved — 모듈 경로(`src/text/`)·출력 스키마(TextualEntity/Event)·relation 8종을 architecture와 정합화
>
> **Epic Goal**: 텍스트 측 주장을 entity/event 단위 구조화 데이터로 변환하는 NLP 파이프라인을 완성한다. (PRD FR4, FR5 / NFR3, NFR4, NFR6 대응)

---

## Story 3.1: NER Fine-tuning

### Status

Approved

### Story

**As a** ML 엔지니어,
**I want** BERT 계열 Transformer(Fakeddit 영어 데이터: DeBERTa 계열, 한국어 금융 subset: KLUE-BERT 또는 KoELECTRA)를 fine-tuning하여 뉴스 텍스트에서 PERSON/ORG/PRODUCT/LOCATION/DATE/MONEY entity를 추출하고,
**so that** 이미지에서 검출된 visual entity와 대조할 수 있는 텍스트 측 entity 목록을 신뢰할 수 있는 품질(entity-level F1 측정)로 확보할 수 있다.

### Acceptance Criteria

1. 텍스트 entity 모듈(`src/text/`, architecture Source Tree 기준)에 token classification 기반 NER 학습 스크립트(`scripts/train_ner.py`)가 존재하고, config(yaml, `configs/ner.yaml`)로 backbone 모델(`klue/bert-base`, `monologg/koelectra-base-v3-discriminator`, `microsoft/deberta-v3-base` 중 선택), learning rate, epoch, batch size, max_length, seed를 제어할 수 있다.
2. NER 라벨 스키마는 BIO tagging 기반으로 PERSON, ORG, PRODUCT, LOCATION, DATE, MONEY 6개 entity type을 지원한다 (총 13개 라벨: B-/I- x 6 + O).
3. 공개 NER 데이터셋(영어: CoNLL-2003 + 금융 도메인 보강, 한국어: KLUE-NER)을 프로젝트 라벨 스키마로 매핑하는 전처리 스크립트가 존재하며, 매핑 규칙이 코드/문서로 명시된다 (예: KLUE-NER의 PS→PERSON, OG→ORG, LC→LOCATION, DT→DATE, QT 중 통화 표현→MONEY).
4. 학습된 모델은 held-out test set에서 entity-level F1(seqeval 기준: micro-F1 및 entity type별 F1)이 산출되어 CSV/로그로 기록된다.
5. 추론 함수는 raw text를 입력받아 architecture의 `TextualEntity` 스키마를 따르는 구조화 리스트를 반환한다 — `[{"entity_id": 0, "type": "ORG", "text": "삼성전자", "span": [0, 4], "embedding": float32[768], "normalized": None, "confidence": 0.97}, ...]` (embedding: span token mean pooling, normalized: MONEY/DATE 정규화 값 — 정규화 유틸은 3.2에서 구현되므로 이 스토리에서는 필드 자리만 확보하고 None 허용). confidence는 token logits의 softmax 평균으로 산출하며, token hidden states `T ∈ R^{256×768}`도 함께 반환한다 (fusion 입력, architecture NER Module 인터페이스).
6. 모든 실험은 seed 고정으로 재현 가능하고(NFR3), 학습은 단일 GPU에서 완료된다(NFR1).

### Tasks / Subtasks

- [ ] Task 1: NER 데이터 준비 (AC: 2, 3)
  - [ ] `src/data/` 및 `src/text/` 에 NER 데이터셋 로더 작성 (HuggingFace `datasets` 사용, architecture Source Tree 준수)
  - [ ] KLUE-NER / CoNLL-2003 라벨 → 프로젝트 6-type BIO 스키마 매핑 함수 구현 및 매핑 테이블 문서화
  - [ ] MONEY/PRODUCT 부족분 보강: 금융 키워드 규칙(₩/조/억/원, $, %, 종목명 사전) 기반 weak labeling 스크립트 작성
  - [ ] train/val/test 분할 스크립트(seed 고정) 및 통계 리포트(문장 수, entity type 분포) 생성
- [ ] Task 2: Fine-tuning 학습 파이프라인 구현 (AC: 1, 6)
  - [ ] `AutoModelForTokenClassification` + `AutoTokenizer` 기반 학습 스크립트 작성 (subword alignment: 첫 subword에만 라벨, 나머지 -100)
  - [ ] yaml config로 backbone/hyperparameter 제어 (Epic 1의 config 시스템 재사용)
  - [ ] early stopping(val entity F1 기준) 및 best checkpoint 저장
- [ ] Task 3: 평가 구현 (AC: 4)
  - [ ] seqeval 기반 entity-level F1 (micro + per-type) 평가 스크립트 작성
  - [ ] 결과를 `results/ner_eval.csv` 로 기록 (모델, seed, F1 per type 포함)
- [ ] Task 4: 추론 인터페이스 구현 (AC: 5)
  - [ ] `src/text/ner.py` 에 `extract(title, body) -> List[TextualEntity]` 인터페이스 구현 (span 병합, confidence·span embedding 산출, token hidden states 반환 포함 — architecture NER Module 계약)
  - [ ] confidence threshold config 옵션 추가 (기본 0.5, threshold 미만 entity 제외)
- [ ] Task 5: 테스트 작성 (AC: 2, 3, 5)
  - [ ] 라벨 매핑 함수 unit test, BIO 인코딩/디코딩 round-trip test, 추론 출력 스키마 test

### Dev Notes

- **모델 선택**: architecture Tech Stack은 `klue/bert-base`(KLUE-BERT)를 표준 Text Backbone으로 고정한다 (relation extraction과 backbone 공유, KLUE-NER label 체계 호환) — 기본 config는 KLUE-BERT로 한다. 다만 PRD Technical Assumptions에 따라 Fakeddit(영어) 실험 시 `microsoft/deberta-v3-base`, KoELECTRA를 config 교체만으로 쓸 수 있게 backbone-agnostic 하게 구현할 것. [Source: architecture.md#Tech Stack, prd.md#Technical Assumptions]
- **라벨 스키마**: `PERSON, ORG, PRODUCT, LOCATION, DATE, MONEY` — Epic 2의 visual class(PERSON/LOGO/PRODUCT/CHART/DOCUMENT/TEXT_REGION)와의 매칭(Epic 4)을 염두: PERSON↔PERSON(얼굴), ORG↔LOGO, PRODUCT↔PRODUCT, DATE/MONEY↔OCR 추출값. [Source: prd.md#FR3, FR4]
- **KLUE-NER 매핑 주의**: KLUE-NER의 QT(quantity)는 통화 단위 포함 여부로 MONEY 여부를 판별(정규식: `[₩$€¥]|원|달러|조|억` 등). PRODUCT는 KLUE-NER에 직접 대응 라벨이 없으므로 weak labeling + 수작업 검수 소량 세트로 보강.
- **Tokenizer alignment**: `is_split_into_words=True` + `word_ids()` 사용, 첫 subword 라벨 / 나머지 -100 방식이 seqeval 평가와 정합.
- **평가 지표**: NFR4 — NER은 entity-level F1이 공식 지표. token-level accuracy는 보조로만 기록.
- **컴퓨팅**: base 모델 + max_length 256 + batch 16~32, 3~5 epoch이면 단일 GPU(Colab) 학습 가능. fp16(amp) 사용 권장.
- **모듈 경계**: `src/text/` 내부에서 자체 완결(NFR6) — Epic 4 matching 모듈은 이 모듈의 출력 스키마(`TextualEntity`)에만 의존해야 한다. [Source: architecture.md#Source Tree, #Data Models]

### Testing

- Unit test: 라벨 매핑, BIO 인코딩/디코딩, NER 출력 스키마(pytest, `tests/test_ner.py` — architecture Test Strategy 준수)
- 모델 검증: held-out test set entity-level F1 (seqeval), per-type F1 리포트
- 회귀 기준: 대표 금융 문장 10개에 대한 golden output snapshot test (entity type/span 고정 확인)

---

## Story 3.2: Event/Relation Extraction

### Status

Approved

### Story

**As a** ML 엔지니어,
**I want** rule + model hybrid 방식으로 텍스트에서 관계 트리플(예: 삼성전자 —CONTRACT_WITH→ NVIDIA, AMOUNT: ₩20조)을 추출하고,
**so that** "누가 누구와 무엇을 얼마에" 수준의 event 주장을 구조화하여 Epic 4의 Event consistency 점수 산출에 사용할 수 있다.

### Acceptance Criteria

1. relation 스키마가 architecture Event 스키마의 8종과 일치하게 정의된다 — `CONTRACT_WITH`(ORG-ORG), `ACQUIRE`(ORG-ORG), `PARTNER_WITH`(ORG-ORG), `INVEST_IN`(ORG-ORG), `CEO_OF`(PERSON-ORG), `EARNINGS_OF`(ORG), `PRICE_OF`(ORG/PRODUCT), `NO_RELATION`(분류기 negative class); event attribute(args): `AMOUNT`(MONEY), `DATE`(DATE). [Source: architecture.md#Data Models]
2. rule 기반 extractor가 구현된다 — Story 3.1의 NER 출력과 trigger 키워드 사전(예: 계약/수주/인수/투자/파트너십, contract/acquire/invest)을 결합하여 문장 단위로 트리플을 추출한다.
3. model 기반 보강이 구현된다 — 문장 내 entity pair에 대해 relation을 분류하는 경량 분류기(BERT 계열 [CLS] 또는 entity marker 기반) 또는 zero-shot NLI 방식 중 하나를 config로 선택 가능하며, rule 결과와 union/confidence 우선 방식으로 병합한다.
4. 출력은 architecture의 Event 스키마를 따른다 — `[{"event_id": 0, "relation": "CONTRACT_WITH", "subject": <entity_id>, "object": <entity_id>, "args": {"AMOUNT": <entity_id>, "DATE": <entity_id>}, "score": 0.8, "source": "rule|model"}]` (subject/object/args는 3.1 `TextualEntity`의 entity_id 참조; `source`는 rule/model 병합 추적용 확장 필드).
5. 수작업 라벨링한 평가 세트(최소 100문장)에서 relation extraction의 Precision/Recall/F1이 산출되어 기록된다.
6. NER 오류 전파를 제한한다 — head/tail entity의 NER confidence가 threshold 미만이면 해당 트리플의 confidence를 감쇠하거나 제외한다 (FR12 정신).

### Tasks / Subtasks

- [ ] Task 1: relation 스키마 및 trigger 사전 정의 (AC: 1, 2)
  - [ ] `src/text/relation.py` 에 relation type 8종, 허용 entity type pair, attribute(args) 정의 (architecture Source Tree 준수)
  - [ ] 금융 도메인 trigger 키워드 사전(한/영) 작성 — relation type별 동사/명사 패턴
- [ ] Task 2: rule 기반 extractor 구현 (AC: 2, 4, 6)
  - [ ] 문장 분리(kss 또는 nltk) 후 문장 내 NER entity + trigger 매칭으로 트리플 후보 생성
  - [ ] head/tail 결정 휴리스틱(주어-목적어 어순, 조사/전치사 패턴) 구현
  - [ ] 같은 문장의 MONEY/DATE entity를 attribute로 부착
- [ ] Task 3: model 기반 relation 분류기 구현 (AC: 3)
  - [ ] entity marker 삽입([E1]...[/E1], [E2]...[/E2]) 방식 문장 인코딩 + relation 분류 head 구현
  - [ ] 학습 데이터: rule 추출 결과 기반 distant supervision + 수작업 검수 세트
  - [ ] rule/model 병합 로직(confidence 우선, source 필드 기록) 구현
- [ ] Task 4: 평가 세트 구축 및 평가 (AC: 5)
  - [ ] 금융 subset에서 100+ 문장 샘플링, 트리플 수작업 라벨링 (라벨링 가이드 문서 포함)
  - [ ] triple-level P/R/F1 평가 스크립트 (`results/relation_eval.csv`)
- [ ] Task 5: 테스트 작성 (AC: 2, 4, 6)
  - [ ] 대표 문장별 rule extractor unit test, 출력 스키마 validation test, confidence 감쇠 로직 test

### Dev Notes

- **hybrid 근거**: PRD FR5 + Story 2 정의 — 순수 model 학습 데이터가 부족하므로 rule을 기본으로, model은 rule이 못 잡는 표현 보강용. 8~10주 일정상 rule 우선 완성 → model은 시간 여유 시 고도화 (rule-only도 Epic 4 진입 가능해야 함).
- **NER 의존성**: 입력은 반드시 Story 3.1의 `TextualEntity` 출력 스키마를 사용 — raw text 재분석 금지 (모듈 경계, NFR6).
- **entity marker 방식**: 한국어는 KLUE-BERT/KoELECTRA tokenizer에 special token으로 `[E1]`/`[/E1]`/`[E2]`/`[/E2]` 추가 후 embedding resize. 문장당 entity pair 조합이 많으면 같은 문장 내 인접 pair만 후보로 제한하여 추론 비용 관리(NFR2).
- **zero-shot 대안**: 학습 데이터 확보가 어려우면 NLI 모델(예: multilingual DeBERTa NLI)로 "「head」가 「tail」와 계약을 맺었다" 가설 검증 방식 허용 — config 스위치로 실험 비교.
- **Epic 4 연계**: AMOUNT/DATE attribute는 OCR로 차트/문서에서 추출한 수치·날짜와 직접 대조되므로 정규화(₩20조 → 2.0e13 또는 canonical string) 유틸을 `src/text/normalize.py` 로 이 스토리에서 구현하고, 3.1 `TextualEntity.normalized` 필드 채움에도 재사용할 것 (architecture Source Tree의 normalize.py, Test Strategy의 Normalize edge case 대응). [Source: prd.md#FR3, FR5; architecture.md#Source Tree]
- **backbone 공유**: model 기반 분류기는 architecture Relation Extractor 정의대로 KLUE-BERT backbone을 NER과 공유하여 단일 GPU 메모리 부담을 줄인다. [Source: architecture.md#Components §6]

### Testing

- Unit test: trigger 매칭, head/tail 휴리스틱, attribute 부착, MONEY/DATE 정규화 함수 (pytest)
- 모델 검증: 수작업 라벨 100+ 문장 triple-level P/R/F1
- Edge case test: entity 0~1개 문장(트리플 없음), 다중 relation 문장, 부정문("계약을 맺지 않았다" — 최소한 known limitation으로 문서화)

---

## Story 3.3: 텍스트 Entity 파이프라인 통합

### Status

Approved

### Story

**As a** 파이프라인 개발자,
**I want** NER(3.1)과 Event/Relation Extraction(3.2)을 하나의 모듈로 통합하여 raw text 입력 → 구조화 entity/event 리스트를 단일 호출로 출력하고,
**so that** Epic 4의 Cross-modal Matching과 Epic 5의 데모가 안정된 인터페이스 하나만 의존하도록 만들 수 있다.

### Acceptance Criteria

1. `TextEntityPipeline` 클래스가 구현된다 — `pipeline.process(title: str, body: str) -> TextEntityResult` 단일 진입점(architecture NER 인터페이스 `extract(title, body)`와 정합), 내부에서 NER → relation extraction 순차 실행.
2. `TextEntityResult`는 직렬화 가능한(dataclass + `to_dict()`/JSON) 스키마로 `entities`(3.1 `TextualEntity` 리스트), `events`(3.2 Event 리스트), `meta`(모델 버전, 처리 시간, config hash)를 포함한다. token hidden states `T`는 tensor 접근 API로 별도 제공(캐시 JSON에는 embedding/hidden states를 npz 등 바이너리 사이드카로 저장하거나 제외 가능 — fusion 학습 요구에 맞춰 문서화).
3. config(yaml) 하나로 파이프라인 전체(모델 checkpoint 경로, threshold, rule/model 모드)를 구성할 수 있으며, Epic 1의 config 시스템과 통합된다.
4. 배치 처리 API(`process_batch(texts)`)가 제공되고, Fakeddit 금융 subset 전체에 대해 entity/event를 사전 추출하여 캐시(jsonl)로 저장하는 스크립트가 존재한다 (fusion 학습 시 재계산 방지).
5. 단일 텍스트 처리 latency가 GPU 기준 1초 이내로 측정·기록된다 (end-to-end 수 초 예산 내 배분, NFR2).
6. 파이프라인 smoke test가 CI 수준에서 실행 가능하다 — 대표 샘플 5개 입력 시 예외 없이 스키마 유효 출력 확인.

### Tasks / Subtasks

- [ ] Task 1: 통합 인터페이스 설계 및 구현 (AC: 1, 2)
  - [ ] `src/text/pipeline.py` 에 `TextEntityPipeline` 구현 (lazy model loading, device 지정; architecture Source Tree의 `src/text/` 하위 유지, 전체 orchestration용 `src/pipeline.py` 와 구분)
  - [ ] `TextEntityResult` dataclass + JSON 직렬화/역직렬화 구현, 스키마 버전 필드 포함
- [ ] Task 2: config 통합 (AC: 3)
  - [ ] `configs/text_entity.yaml` 작성 — NER checkpoint, relation 모드(rule/hybrid), threshold 일원화
  - [ ] Epic 1 config loader와 연결, config hash를 meta에 기록 (재현성, NFR3)
- [ ] Task 3: 배치 추출 및 캐시 (AC: 4)
  - [ ] `scripts/extract_text_entities.py` — 금융 subset 전체 → `data/cache/text_entities.jsonl`
  - [ ] 캐시 무효화 기준(config hash 불일치 시 재추출) 구현
- [ ] Task 4: 성능 측정 (AC: 5)
  - [ ] 샘플 100개 평균/최대 latency 측정 스크립트, `results/text_entity_latency.csv` 기록
  - [ ] 병목 시 완화책 적용(batch tokenization, fp16, max_length 조정)
- [ ] Task 5: 테스트 및 문서 (AC: 6)
  - [ ] smoke test(pytest) — 대표 샘플 5개 + 빈 문자열/초장문 edge case
  - [ ] 모듈 README: 입출력 스키마, 사용 예시, Epic 4 소비자를 위한 인터페이스 계약 명시

### Dev Notes

- **모듈 경계(NFR6)**: Epic 4의 matching 모듈은 `TextEntityResult` 스키마에만 의존해야 하며, NER/relation 내부 구현 교체(예: KoELECTRA→KLUE-BERT)가 소비자 코드에 영향을 주지 않아야 ablation이 성립한다.
- **캐시 설계 이유**: fusion 학습(Epic 4)과 ablation(Epic 5)에서 동일 데이터에 반복 접근하므로, 텍스트 entity 추출을 사전 계산해 두면 단일 GPU 학습 시간을 크게 절약한다 (NFR1).
- **meta 필드**: 모델 checkpoint hash + config hash를 기록해 두면 ablation 표 작성 시 "어떤 NER 버전으로 뽑은 feature인지" 추적 가능 (NFR3 재현성).
- **오류 처리**: NER/relation 개별 실패 시 파이프라인 전체가 죽지 않고 빈 리스트 + meta.error 기록으로 degrade — FR12의 오류 전파 제한 원칙을 텍스트 측에도 적용.
- **데모 연계(Epic 5)**: 텍스트 entity 하이라이트를 위해 entity의 start/end offset이 raw text 기준으로 정확해야 한다 — tokenizer offset mapping(`return_offsets_mapping=True`) 사용 필수.

### Testing

- Unit test: 직렬화 round-trip, config hash, 캐시 무효화 로직
- Smoke test: 대표 샘플 5개 + edge case(빈 문자열, 이모지/URL 포함 텍스트, 512 token 초과 장문)
- 통합 test: 캐시 파일 생성 스크립트를 소규모 subset(10건)으로 실행하여 jsonl 스키마 검증
- 성능 test: latency 1초/샘플 이내 확인 (GPU 환경)

---

## Epic 3 완료 기준 (Definition of Done)

- [ ] 3.1~3.3 모든 스토리의 AC 충족 및 테스트 통과
- [ ] NER entity-level F1, relation P/R/F1 결과가 `results/`에 기록됨 (NFR4)
- [ ] 금융 subset 전체 텍스트 entity 캐시 생성 완료 — Epic 4 착수 가능 상태
- [ ] 모든 실험 config + seed 기록으로 재현 가능 (NFR3)
