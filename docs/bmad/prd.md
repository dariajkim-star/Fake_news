# FinFact Product Requirements Document (PRD)

> Object-Level Multimodal Financial Misinformation Detection
> 작성자: John (BMAD PM) · 작성일: 2026-08-10 · 버전: v1.0

---

## Goals and Background Context

### Goals

- 금융 뉴스 게시물(이미지 + 텍스트)에 대해 Fake/Real 확률과 판단 근거를 출력하는 object-level multimodal 탐지 모델을 완성한다.
- 단일 global CLIP similarity 방식이 놓치는 세부 불일치("사진 속 인물은 A인데 기사는 B", "차트 수치·날짜가 기사 주장과 불일치")를 word-region pair 수준의 cross-modal matching으로 탐지한다.
- **핵심 입증 목표**: ablation table(A1~A6: BERT only / ResNet only / late fusion / +Object Detection / +Textual entity features / +Entity consistency)에서 entity consistency feature 추가(A5→A6)가 최종 분류 F1을 추가 개선함을 실험으로 입증한다.
- **금융 도메인 특화 입증**: Fakeddit 키워드 필터 subset과 별개로, 금융 뉴스 기반 자체 held-out 평가셋(FinFact-Eval)을 구축해 금융 도메인 성능과 entity mismatch 판정 정확도를 직접 측정한다.
- 어떤 entity가 MISMATCH인지 근거를 시각적으로 제시하는 데모(Gradio/Streamlit)를 제공하여 explainability를 확보한다.
- 재현 가능한 코드, 실험 결과, 발표 자료를 과제 deliverable로 산출한다.

### Background Context

금융 도메인에서는 조작된 이미지(가짜 계약서, 합성 인물 사진, 조작 차트)와 자극적인 텍스트를 결합한 게시물이 주가조작·투자 유인에 악용된다. 기존 멀티모달 탐지의 다수는 이미지 전체 embedding과 텍스트 전체 embedding의 similarity 한 번(single CLIP similarity)에 의존하기 때문에, 이미지의 전반적 주제만 유사하면 통과되어 object-level 불일치를 놓친다. 금융 가짜뉴스는 인물(CEO), 기업 로고, 계약서/공시 문서, 차트 수치 등 구체적 entity 수준의 사실 주장으로 구성되므로 entity 단위 대조가 특히 유효하다.

FinFact는 Object Detection으로 이미지를 객체 단위 증거로 분해하고, NER/Event Extraction으로 텍스트 주장을 entity 단위로 추출한 뒤, 이를 하나씩 대조하는 object-level entity consistency 접근을 취한다. 연구 근거는 EM-FEND(arXiv:2108.10509), CFFN(arXiv:2311.01807), Event-Radar(ACL 2024), PROPOR 2026 financial fake news framework, Fakeddit 데이터셋(arXiv:1911.03854)이다. 본 프로젝트는 8~10주 딥러닝 과제로, 단일 GPU 환경에서 Phase 1~4 단계별 완성 전략으로 진행한다.

### Change Log

| Date | Version | Description | Author |
|---|---|---|---|
| 2026-08-10 | v1.0 | 최초 PRD 작성 (project-brief 기반) | John (PM) |
| 2026-08-10 | v1.1 | 문서 리뷰 갭 3건 반영 — 상세는 하단 "Revision Notes (v1.1)" 참조 | John (PM) |
| 2026-08-10 | v1.2 | 팀 리뷰 반영 — 통계적 주장 수준 하향(FR10·FR13·FR16), κ 한계·제3자 스팟체크, FR11 조건부 트리거, fallback ladder, 사람 시간 경합 조정. 상세는 "Revision Notes (v1.2)" 참조 | John (PM) |

---

## Requirements

### Functional Requirements

- **FR1**: 시스템은 이미지 + 텍스트 쌍을 입력받아 Fake/Real 이진 분류 확률(fake score)을 출력해야 한다.
- **FR2**: 시스템은 Object Detection(YOLOv8 또는 DETR)으로 이미지에서 PERSON, LOGO, PRODUCT, CHART, DOCUMENT, TEXT_REGION 클래스의 bbox를 검출해야 한다.
- **FR3**: 시스템은 검출된 bbox crop에 대해 Visual Entity Recognition을 수행해야 한다 — PERSON→얼굴/인물 인식(예: "Jensen Huang"), LOGO→CLIP 기반/로고 분류기(예: "NVIDIA"), CHART/TEXT_REGION/DOCUMENT→OCR로 수치·날짜 추출.
- **FR4**: 시스템은 Transformer(KoELECTRA / KLUE-BERT / DeBERTa) fine-tuning으로 텍스트에서 NER(PERSON, ORG, PRODUCT, LOCATION, DATE, MONEY)을 수행해야 한다.
- **FR5**: 시스템은 텍스트에서 Event/Relation Extraction을 수행해야 한다 (예: 삼성전자 —CONTRACT_WITH→ NVIDIA, AMOUNT ₩20조).
- **FR6**: 시스템은 visual entity와 textual entity를 정렬하는 Cross-modal Matching을 수행하여 각 entity pair를 MATCH / MISMATCH / UNKNOWN으로 분류해야 한다.
- **FR7**: 시스템은 Cross-modal Attention fusion(region feature × token feature)으로 Entity consistency, Event consistency, Image/Text similarity 점수를 산출하고 최종 fake probability에 결합해야 한다.
- **FR8**: 추론 파이프라인은 fake score와 함께 판단 근거(어떤 entity가 mismatch인지)를 구조화된 형태로 출력해야 한다.
- **FR9**: 데모 UI(Gradio 또는 Streamlit)는 이미지+텍스트 입력을 받아 fake score와 mismatch evidence(이미지 bbox 하이라이트 + 텍스트 entity 하이라이트)를 시각화해야 한다.
- **FR10**: 시스템은 baseline(ResNet+BERT late fusion)부터 최종 모델까지 **A1~A6 6개 구성**을 동일 평가 셋·동일 split·동일 seed에서 비교하는 ablation 실험을 지원해야 한다. 구성 정의는 DEV_PLAN §6을 정본으로 한다:
  - **A1** BERT only (text)
  - **A2** ResNet only (image)
  - **A3** A1+A2 late fusion
  - **A4** A3 + YOLO object regions + Cross-modal Attention
  - **A5** A4 + Textual entity features (NER/event entity embedding 단독 기여)
  - **A6** A5 + Entity/Event consistency score (최종)
  - A6의 개선폭은 반드시 **A5 대비**(A4 대비가 아님)로 보고해야 한다.
  - **핵심 가설(A5→A6 개선)의 검증 셋은 Fakeddit 금융 subset(5k~20k)이며, 3-seed 평균으로 판정한다.** FinFact-Eval(n=300)은 가설 검증의 1차 근거가 아니다 — n=300에서 F1 1%p는 샘플 3건에 해당하고, McNemar 검정으로 유의성을 얻으려면 8~10%p 수준의 점프가 필요한데 이는 기대 가능한 효과 크기가 아니다. FinFact-Eval의 A1~A6 수치는 **도메인 이전의 정성적 확인용 보조 표**로만 보고하며, 유의성 주장을 하지 않는다.
- **FR11**: 학습/평가 데이터로 Fakeddit을 사용하며, 키워드 필터링으로 금융 subset을 구성할 수 있어야 한다. **금융 subset의 실규모에 따라 그 용도를 조건부로 결정한다** — 판정 시점은 1주차 필터링 스크립트 실행 직후(Epic 2 착수 전)이며, 기준은 필터링 후 **train split 샘플 수**다:
  - **≥ 5k**: 현행 유지 — 금융 subset을 주 학습·평가 셋으로 사용.
  - **3k ~ 5k**: 전체 Fakeddit pretrain → 금융 subset fine-tuning의 **2-stage 학습**으로 전환 (DEV_PLAN §7 대응).
  - **< 3k**: FR11을 **평가 전용으로 축소** — 금융 subset은 학습에 사용하지 않고 A1~A6 공통 평가 셋으로만 쓰며, 학습은 전체 Fakeddit으로 수행한다. 이 경우 "금융 특화 학습" 주장을 철회하고 "일반 도메인 학습 + 금융 도메인 평가"로 프레임을 정정해 발표 자료에 명시한다.
  - 어느 분기든 실측 숫자와 선택한 분기를 `docs/results/`에 기록한다.
- **FR12**: Visual Entity Recognition 신뢰도가 threshold 미만인 경우 UNKNOWN으로 처리하여 오류 전파를 제한해야 한다 (soft matching / rule + model hybrid 허용).
- **FR13**: 프로젝트는 Fakeddit과 독립된 **자체 금융 held-out 평가셋(FinFact-Eval)** 을 구축해야 한다 — 공개 금융 뉴스 기사(텍스트 + 대표 이미지) 기반 Real 샘플과, 이미지·텍스트를 의도적으로 불일치시킨 Fake 샘플로 구성하며, 학습에 사용하지 않고 평가 전용으로 고정한다. FinFact-Eval의 역할은 **다음 3가지로 한정**한다:
  - **(i) FR16 matching 판정 정확도의 유일한 정답 소스** — Fakeddit에는 entity pair annotation이 없으므로 이 셋 없이는 FR16 자체가 측정 불가다. 이것이 FinFact-Eval의 1차 존재 이유다.
  - **(ii) 도메인 이전의 정성적 확인** — Fakeddit 금융 subset에서 학습·검증된 모델이 실제 금융 기사 분포에서도 동작하는지 보조 표로 확인 (유의성 주장 없음, FR10 참조).
  - **(iii) 데모·발표 예시 공급** — Story 5.3의 pre-loaded 예시와 Story 5.2의 대표 사례.
  - 즉 이 셋은 "핵심 가설을 검정하는 통계 표본"이 아니라 "**entity 대조 능력을 측정할 수 있는 유일한 정답지 + 도메인 증거**"다. 이 역할 정의 하에서 300건은 과소 규모가 아니라 적정 규모다.
- **FR14**: FinFact-Eval의 모든 Fake 샘플은 **mismatch 유형 taxonomy**로 annotation되어야 한다 — `PERSON_MISMATCH`(인물 오류), `ORG_LOGO_MISMATCH`(기업/로고 불일치), `NUMBER_MISMATCH`(수치·금액 모순), `DATE_MISMATCH`(시점 모순), `EVENT_MISMATCH`(사건/관계 오류), `NONE`(Real). 샘플당 primary 유형 1개 + secondary 유형 0개 이상, 그리고 근거가 되는 entity pair(visual 값, textual 값)를 함께 기록한다.
- **FR15**: FinFact-Eval annotation은 품질 관리 절차를 거쳐야 한다 — 작성된 annotation guideline, blind 재라벨링 기반 일치도(Cohen's κ) 측정, κ 미달 항목에 대한 adjudication 및 최종 라벨 확정 기록.
- **FR16**: 시스템은 FinFact-Eval의 entity pair annotation을 정답으로 하여 **Cross-modal Matching 판정 정확도**를 측정할 수 있어야 한다. 결론적으로 보고하는 지표는 **전체 단위 1세트**로 한정한다 — MATCH/MISMATCH/UNKNOWN 3-way accuracy와 MISMATCH 클래스 Precision/Recall/F1(pair 총계 기준). **mismatch 유형별 지표 표는 산출·보고하지 않는다** — 유형당 25~35건 표본에서는 Wilson 95% CI 폭이 약 27%p(예: n=30, recall 0.80 → [0.63, 0.90])라 유형 간 차이가 통계적으로 구분되지 않기 때문이다. 유형별 특성은 지표 대신 **유형당 정성 사례 3~5건**(Story 5.2 error analysis)으로 보고한다. 유형별 표본 수와 원자료(per-pair 판정)는 파일로 남기되, 표로 제시하지 않는다.
- **FR17**: 시스템은 FinFact-Eval의 OCR 검증 subset(차트/문서/TEXT_REGION crop과 정답 문자열 쌍)에 대해 **OCR CER(Character Error Rate)** 을 측정할 수 있어야 한다. 해당 subset이 최소 규모(50 crop)에 미달하여 측정이 불가한 경우, OCR 지표는 CER 대신 "수치 추출 정확도(정성 샘플 검증)"로 대체하고 그 사실을 결과 리포트에 명시한다.

### Non-Functional Requirements

- **NFR1**: 모든 학습은 단일 GPU(Colab/로컬 수준)에서 수행 가능해야 한다 — pretrained model(YOLO, CLIP, BERT 계열) 최대 활용, 모듈별 독립 학습 후 fusion 단계 결합.
- **NFR2**: 단일 샘플(이미지+텍스트) end-to-end 추론은 데모 시연에 적합한 시간(GPU 기준 수 초 이내)에 완료되어야 한다.
- **NFR3**: 실험은 재현 가능해야 한다 — random seed 고정, config 파일 기반 실험 관리, 데이터 전처리/분할 스크립트 버전 관리.
- **NFR4**: 컴포넌트별 평가 지표를 산출해야 한다 — Object Detection: mAP@50/Precision/Recall, NER: entity-level F1, 최종 분류기: Accuracy/Precision/Recall/F1/AUROC, Cross-modal Matching: 판정 정확도(FR16), OCR: CER(FR17). 지표는 반드시 측정 스토리를 가져야 하며, 측정 계획이 없는 지표는 문서에서 제거한다.
- **NFR5**: 공개 데이터셋만 사용한다 (Fakeddit + 자체 필터링/수작업 라벨링 금융 subset). 크롤링 기반 실시간 데이터 수집은 범위 외. FinFact-Eval 수집 역시 **자동 크롤링을 사용하지 않는다** — 공개 라이선스 데이터셋/아카이브, 기관 보도자료·공시 등 재사용이 허용된 공개 자료, 그리고 사람이 수동으로 선별해 URL·출처·라이선스를 기록한 기사에 한정한다.
- **NFR6**: 코드는 모듈화되어야 한다 — detection, entity recognition, NER, matching, fusion이 독립적으로 학습·평가·교체 가능해야 ablation이 성립한다.
- **NFR7**: Out of scope 준수 — 실시간 크롤링/서비스 배포, 다국어 동시 지원, 딥페이크/forgery 자체 탐지, 분산 학습, 사용자 계정 시스템은 구현하지 않는다.
- **NFR8**: FinFact-Eval은 **held-out 무결성**을 보장해야 한다 — 어떤 구성(A1~A6)의 학습·하이퍼파라미터 튜닝·threshold 선택에도 사용하지 않으며, 샘플 id 목록을 고정 파일로 관리해 학습 split과의 교집합이 0임을 자동 검증한다.
- **NFR9**: FinFact-Eval은 **재배포 가능한 형태**로 산출해야 한다 — repo에는 원문 이미지 대신 출처 URL·라이선스·해시와 annotation(JSONL)을 포함하고, 재배포가 허용된 자산만 release에 첨부한다. 저작권 확인이 안 된 이미지는 셋에서 제외한다.

---

## UI Design Goals

### Overall UX Vision

발표 청중과 과제 평가자가 "어떤 entity가 불일치했는지"를 한눈에 이해할 수 있는 단일 화면 데모. 복잡한 설정 없이 이미지 업로드 + 텍스트 입력 → 결과 확인의 최소 흐름.

### Core Screens and Views

- **입력 영역**: 이미지 업로드 + 뉴스 텍스트 입력창 + 분석 버튼, 예시 샘플(pre-loaded examples) 제공
- **결과 영역**:
  - Fake probability 게이지/점수 표시
  - 이미지 위 검출 bbox 오버레이 — MATCH(녹색)/MISMATCH(적색)/UNKNOWN(회색) 색상 구분
  - 텍스트 entity 하이라이트 — 동일 색상 체계로 대응 visual entity와 매칭 표시
  - Entity consistency / Event consistency / Image-Text similarity 세부 점수 테이블

### Accessibility & Branding

- 과제 데모 수준: 별도 branding 없음, 색상 외 라벨 텍스트 병기(MATCH/MISMATCH)로 색각 접근성 보완

### Target Device and Platforms

- Web Responsive 불요 — 데스크톱 브라우저에서 실행되는 Gradio 또는 Streamlit 로컬 앱(발표 시연용)

---

## Technical Assumptions

### Repository Structure: Monorepo

단일 저장소에 데이터 처리, 각 모델 모듈, 학습/평가 스크립트, 데모 앱을 포함.

### Service Architecture

배포용 서비스 아키텍처 없음 — 로컬 ML 파이프라인 + 데모 앱(Monolith script 기반). 모듈 구성: `detection` → `visual_entity` → `text_entity` → `matching` → `fusion` → `demo`.

### Testing Requirements

- Unit test: 데이터 전처리, matching 로직(rule 기반 부분) 중심
- 모델 검증: 컴포넌트별 held-out 평가(mAP, NER F1) + end-to-end 평가 + ablation
- 데모: 수동 시연 테스트(대표 샘플 세트)

### Additional Technical Assumptions and Requests

- **Framework**: PyTorch 기반, HuggingFace Transformers(BERT 계열 fine-tuning, NER), Ultralytics(YOLOv8)
- **Vision**: YOLOv8(우선) 또는 DETR, CLIP embedding(로고/시각 entity), 얼굴 인식 모델, 딥러닝 기반 OCR(EasyOCR/PaddleOCR 등)
- **NLP**: KoELECTRA / KLUE-BERT / DeBERTa 중 데이터 언어에 맞게 선택 (Fakeddit은 영어 → DeBERTa 계열 우선, 금융 한국어 셋 확보 시 KLUE-BERT/KoELECTRA)
- **데이터셋**: Fakeddit(100만+ text-image pair, 2/3/6-way label) + 키워드 필터링 기반 자체 금융 subset(부족 시 수작업 라벨링 보완)
- **컴퓨팅**: 단일 GPU(Colab Pro/로컬) — batch size, epoch, 모델 크기 제약 하에 설계
- **실험 관리**: config 기반(yaml) + seed 고정, 결과 로깅(간단한 CSV/TensorBoard/W&B 중 택1)

---

## Epic List

- **Epic 1: Baseline 파이프라인** — 프로젝트 스캐폴딩, Fakeddit 데이터 파이프라인, ResNet+BERT late fusion 이진 분류 baseline 및 평가 체계 확립
- **Epic 2: Object Detection 통합** — 6-class detector 학습/적용, region encoder, cross-modal attention으로 baseline 확장
- **Epic 3: Text Entity Extraction** — Transformer fine-tuning NER + Event/Relation Extraction, 텍스트 entity 파이프라인 구축
- **Epic 4: Cross-modal Consistency & 최종 모델** — Visual Entity Recognition, entity matching(MATCH/MISMATCH/UNKNOWN), consistency feature 결합 최종 모델 완성
- **Epic 5: 평가·Ablation·데모** — 전체 ablation 실험, error analysis, Gradio/Streamlit 데모, 발표 자료
- **Epic 6: FinFact-Eval — 금융 도메인 특화 held-out 평가셋** — 공개 출처 금융 뉴스 수집, mismatch 유형 taxonomy annotation, 품질 관리, matching 판정 정확도·OCR CER 측정 하네스 (Epic 2~4와 **병행** 진행하며 Epic 5 착수 전 완료)

> **Epic 6 번호와 실행 순서에 대한 주의**: Epic 1~5는 PO 승인 완료 상태이므로 번호를 재배열하지 않고 Epic 6으로 이어 붙였다. 그러나 Epic 6은 마지막에 하는 일이 아니라 **DEV_PLAN §5의 3주차(수집 시작)~7주차(셋 완성) 일정에 병행**되는 데이터 트랙이며, Epic 5의 Story 5.1(AC 6: `custom_fin` 평가 셋)·5.2(AC 3: mismatch 유형별 정확도)·5.3(AC 4: 예시 샘플)이 Epic 6 산출물에 의존한다.

---

## Epic Details

### Epic 1: Baseline 파이프라인

**Goal**: Fakeddit 기반 데이터 파이프라인과 ResNet+BERT late fusion baseline을 완성하고, 이후 모든 ablation의 비교 기준이 되는 평가 체계(Accuracy/P/R/F1/AUROC)를 확립한다.

- **Story 1.1**: 프로젝트 스캐폴딩 — repo 구조, config 시스템, seed 고정, 학습/평가 공통 루프 구축
- **Story 1.2**: Fakeddit 데이터 파이프라인 — 다운로드/전처리/2-way label 정리, train/val/test 분할, 금융 키워드 필터링 subset 스크립트
- **Story 1.3**: 단일 모달 baseline — BERT only(텍스트), ResNet only(이미지) 분류기 학습 및 평가
- **Story 1.4**: Late fusion baseline — ResNet feature + BERT feature concat → 분류기, ablation 기준선 기록

### Epic 2: Object Detection 통합

**Goal**: 이미지를 객체 단위 증거로 분해하는 detector를 확보하고, region feature를 cross-modal attention으로 텍스트와 결합해 baseline 대비 개선을 확인한다.

- **Story 2.1**: Detection 데이터 준비 — PERSON/LOGO/PRODUCT/CHART/DOCUMENT/TEXT_REGION 라벨 전략(pretrained 클래스 매핑 + 부분 수작업 라벨링)
- **Story 2.2**: YOLOv8 학습/적용 — fine-tuning 또는 pretrained 활용, mAP@50/P/R 평가
- **Story 2.3**: Region Encoder — bbox crop feature 추출(CNN/CLIP), region feature 시퀀스 구성
- **Story 2.4**: Cross-modal Attention fusion — region feature × token feature attention 모델 학습, "+Object Detection" ablation 결과 기록

### Epic 3: Text Entity Extraction

**Goal**: 텍스트 측 주장을 entity/event 단위 구조화 데이터로 변환하는 NLP 파이프라인을 완성한다.

- **Story 3.1**: NER fine-tuning — BERT 계열 모델로 PERSON/ORG/PRODUCT/LOCATION/DATE/MONEY 추출, entity-level F1 평가
- **Story 3.2**: Event/Relation Extraction — rule + model hybrid로 관계 트리플 추출(예: ORG—CONTRACT_WITH→ORG, AMOUNT)
- **Story 3.3**: 텍스트 entity 파이프라인 통합 — 입력 텍스트 → 구조화 entity/event 리스트 출력 모듈화

### Epic 4: Cross-modal Consistency & 최종 모델

**Goal**: visual entity와 textual entity의 대조로 consistency feature를 산출하고 최종 fake 분류 모델을 완성한다.

- **Story 4.1**: Visual Entity Recognition — 얼굴/인물 인식, CLIP 기반 로고 분류, OCR 수치·날짜 추출, confidence threshold + UNKNOWN 처리
- **Story 4.2**: Cross-modal Matching — visual entity vs textual entity 정렬(soft matching, embedding 기반 + rule hybrid) → MATCH/MISMATCH/UNKNOWN
- **Story 4.3**: Consistency feature 결합 최종 모델 — Entity consistency/Event consistency/Image-Text similarity 점수를 fusion 분류기에 결합, "+Entity consistency" 학습
- **Story 4.4**: 추론 파이프라인 통합 — 이미지+텍스트 입력 → fake score + mismatch entity 근거 출력 end-to-end 모듈

### Epic 5: 평가·Ablation·데모

**Goal**: 전체 ablation 실험으로 핵심 가설(entity consistency의 F1 추가 개선)을 검증하고, 데모와 발표 자료로 결과를 전달한다.

- **Story 5.1**: Ablation 실험 — A1~A6(BERT only / ResNet only / late fusion / +Object Detection / +Textual entity features / +Entity consistency) 전 구성 동일 조건 평가, Fakeddit 금융 subset과 FinFact-Eval 두 셋 모두에서 결과 표 작성 (FR10)
- **Story 5.2**: Error analysis — MISMATCH 오탐/미탐 사례 분석, 컴포넌트별 오류 전파 분석(개선 실패 시 원인 분석 자체를 기여로 제시)
- **Story 5.3**: 데모 앱 — Gradio/Streamlit UI(이미지+텍스트 입력 → fake score, bbox/entity 하이라이트, consistency 점수 표), 예시 샘플 탑재
- **Story 5.4**: 발표 자료 및 재현 문서 — 실험 결과 정리, README/재현 가이드, 발표 슬라이드

### Epic 6: FinFact-Eval — 금융 도메인 특화 held-out 평가셋

**Goal**: "금융 특화"라는 프로젝트 정체성을 검증할 수 있는 독립 평가 자산을 만든다 — 공개 출처만으로 수집한 금융 뉴스 Real 샘플과 통제된 mismatch 조작 Fake 샘플, mismatch 유형 taxonomy annotation, 그리고 이를 소비해 금융 도메인 성능·matching 판정 정확도·OCR CER을 산출하는 평가 하네스까지. (FR13~FR17, NFR5, NFR8, NFR9)

**규모 결정 (8~10주 단일 인원 제약)**: 총 **300건(Real 150 / Fake 150)** 을 커밋 목표로 한다(stretch 400건). Fake 150건은 mismatch 유형 5종에 각 25~35건씩 배분하는데, 이는 **유형별 지표를 보고하기 위해서가 아니라** 특정 유형에만 쏠린 셋이 되지 않도록 커버리지를 확보하기 위해서다(유형별 지표를 보고하지 않는 이유는 FR16 참조). OCR 검증 subset은 CHART/DOCUMENT/TEXT_REGION crop 50~80개로 별도 구성한다. DEV_PLAN §2.3의 "실제 기사 200~500건" 상한은 단일 인원 기준 비현실적이므로 하한(총 300건)을 정본으로 채택한다.

**규모 축소 fallback ladder (사전 합의, 6주차 말 실측 건수로 자동 판정)**: 수동 수집·라벨링은 사람 시간 제약이 커서 목표 미달 가능성이 실재한다. 오너가 마감 직전에 급하게 결정하지 않도록 **축소 단계와 각 단계에서 포기하는 주장을 미리 고정**한다.

| 규모 | 성격 | 보고 가능한 것 | 포기하는 주장 |
|---|---|---|---|
| **300** (Real 150 / Fake 150) | held-out 정량 평가 | 전체 MISMATCH P/R/F1(pair 총계, n≈300+ pair → CI ±6~8%p), 3-way accuracy, 유형별 정성 사례 3~5건 | 유형별 정량 비교(애초에 주장하지 않음, FR16) |
| **180** (90 / 90) | 정량, 단 정밀도 하락 | 전체 MISMATCH P/R/F1(CI ±9~11%p로 확대 — 리포트에 CI 병기 필수) | 소수점 단위 비교. "0.72 vs 0.78" 같은 차이 주장 불가 |
| **100** (50 / 50) | 탐색적 평가 | matching 정확도 총계 1개 + 정성 사례 | 금융 도메인 **분류 성능 비교**(A1~A6 표) 철회. FinFact-Eval 역할 (ii) 포기 |
| **40** (case study) | 정성 사례집 | 데모 예시, 대표 mismatch 사례 서술 | FR16 정량 지표 전부 철회. 발표에서 "matching 정확도 측정 못 함"을 한계로 명시 |

- **컷오프 규칙**: 6주차 말 실측 수집·annotation 완료 건수로 위 표의 해당 행을 자동 선택한다. 사후 재량 판단을 두지 않는다.
- **어느 단계로 내려가든 (i) FR16 정답 소스 역할이 최우선**이다 — 규모를 줄일 때 Real/Fake 건수를 줄이더라도 entity pair annotation 품질(Story 6.3~6.4)은 축소하지 않는다. 품질을 줄이면 남은 지표마저 무의미해진다.

**실행 순서 강제 장치 (Epic 번호 6이 실행 순서 6으로 오해되는 것 방지)**: 문서에 "3주차 병행"이라고 적는 것만으로는 강제력이 없다는 지적(Analyst)을 수용해, 다음을 합의했다.
- **Epic 1에 Story 1.5 "FinFact-Eval 수집 착수 게이트"를 신설한다** — 내용: Story 6.1의 출처·라이선스 정책 확정 + 시드 20건 수집. Epic 1 안에 앵커가 있어야 무시되지 않는다. 단 `docs/bmad/stories/epic-1-baseline.md`는 PO 관할 문서이므로 **실제 스토리 작성은 PO(Sarah)에게 이관**하며, 본 PRD는 그 신설 의도와 범위만 기록한다.
- **Epic 2 착수 시 체크**: 3주차 말 Real 누적 30건 미달이면 Epic 2를 중단하지는 않되, 해당 주 GPU 대기시간을 수집에 배정한다(사전 합의).
- **주간 체크포인트 고정 지표**: "FinFact-Eval 누적 수집/annotation 건수" 한 줄을 매주 기록한다.

**사람 시간 경합 조정 (Story 2.1 bbox 수작업 vs Epic 6 수집)**: 두 작업 모두 GPU가 아니라 오너의 손 시간을 쓰며, DEV_PLAN §5 기준 **4주차에 정면 충돌**한다(2.1 bbox annotation 4주차, 6.1 수집 3~7주차). 조정안:
- **자체 bbox annotation을 300장 → 150장으로 축소**하고 부족분은 공개 데이터 재활용(LogoDet-3K subset 등)으로 대체한다. CHART/DOCUMENT의 낮은 mAP는 이미 허용된 리스크(DEV_PLAN §7)이므로 여기서 비용을 깎는다. *(Story 2.1 본문 반영은 PO 관할 — 본 PRD는 결정만 기록)*
- **수작업 작업을 주차별로 번갈아 배치**한다 — 4주차는 bbox, 5주차는 FinFact-Eval 수집. 사람 시간은 병렬화되지 않으므로 스케줄에서 분리한다.

- **Story 6.1**: 수집 소스·라이선스 정책 확정 및 Real 샘플 수집 — 공개 라이선스 뉴스 아카이브/기관 보도자료/공시 등 재사용 허용 출처 목록 확정, 자동 크롤링 금지(NFR5), 샘플별 출처 URL·라이선스·수집일·이미지 해시 기록, Real 150건 확보
- **Story 6.2**: 통제된 Fake 샘플 생성 — 수집한 Real 샘플을 기반으로 mismatch 유형별 조작 생성(인물 사진 교체, 로고/기업 교체, 금액·수치 변조 headline, 날짜 변경, 사건/관계 오도), 조작 전/후 원본 링크(`source_real_id`)와 조작 스크립트/절차 기록 → Fake 150건
- **Story 6.3**: mismatch taxonomy 및 annotation guideline — FR14의 6개 라벨 정의·경계 사례·판정 규칙 문서화, entity pair(visual 값 / textual 값) 기록 스키마와 JSONL 포맷 확정, 샘플 20건으로 파일럿 annotation 후 guideline 개정
- **Story 6.4**: annotation 수행 및 품질 관리 — 전 샘플 annotation, 20% 표본에 대해 최소 7일 간격 blind 재라벨링으로 intra-annotator Cohen's κ 산출(목표 κ ≥ 0.8), 불일치 항목 adjudication 및 사유 기록, **그리고 제3자 스팟체크 30건을 별도 품질 게이트로 수행**. κ는 일관성 지표이지 정확성 지표가 아니므로(같은 사람이 같은 실수를 반복하면 κ는 높은데 라벨은 틀림) κ 단독으로는 게이트를 통과시키지 않는다
- **Story 6.5**: 데이터셋 패키징 및 held-out 무결성 검증 — `data/finfact_eval/` 스키마 확정(annotations.jsonl + manifest + splits), 학습 split과의 id 교집합 0 자동 검증 테스트(NFR8), 라이선스 미확인 이미지 제외 및 재배포 패키지 구성(NFR9)
- **Story 6.6**: 평가 하네스 — FinFact-Eval 기반 지표 산출 — (a) 금융 도메인 분류 성능(Accuracy/P/R/F1/AUROC)을 A1~A6에 대해 **보조 표**로 산출(유의성 주장 없음, FR10), (b) Cross-modal Matching 3-way 판정 정확도 및 MISMATCH P/R/F1을 **전체 단위 1세트**로 산출하고 CI를 병기(유형별 표는 산출하지 않음, FR16), (c) OCR CER 산출 및 subset 미달 시 대체 지표 명시(FR17)

---

## Checklist Results Report

PM 체크리스트 자체 점검 결과:

| 항목 | 상태 | 비고 |
|---|---|---|
| Problem definition & context | PASS | project-brief 기반, 금융 도메인 특수성 명시 |
| MVP scope definition | PASS | Phase 1~4 In/Out of Scope 명확 |
| User experience requirements | PASS | 데모 중심 최소 UI, explainability 요구 반영 |
| Functional requirements | PASS | FR1~FR12, 파이프라인 단계별 커버 |
| Non-functional requirements | PASS | 단일 GPU/재현성/평가 지표 명시 |
| Epic & story structure | PASS | 6 Epic, 각 Epic이 배포 가능한 증분(ablation 단계 + 평가 자산)과 일치 |
| Domain-specific evaluation asset | PASS (v1.1) | Epic 6 신설로 FinFact-Eval 300건 구축·annotation·측정 하네스 계획 확보 |
| Metric measurability | PASS (v1.1) | matching 판정 정확도(FR16)·OCR CER(FR17)에 Story 6.6이라는 측정 주체 부여, 미달 시 대체 규칙 명시 |
| Technical guidance | PASS | PyTorch/HuggingFace/Ultralytics, 데이터셋 확정 |
| Cross-functional requirements | PARTIAL | Detection 라벨링 데이터 규모·비용은 Epic 2 착수 시 구체화 필요 |
| Clarity & communication | PASS | 한국어 + 기술 용어 영어 유지 |

**주요 리스크(모니터링)**: (1) Fakeddit 금융 subset 규모 부족, (2) Visual Entity Recognition 정확도로 인한 consistency 노이즈, (3) entity consistency의 F1 개선 미달 가능성 — Epic별 중간 평가로 조기 감지, error analysis를 대안 기여로 준비.

---

## Next Steps

### UX Expert Prompt

이 PRD의 UI Design Goals 섹션을 기반으로 Gradio/Streamlit 데모 화면(입력 영역, fake score 표시, bbox/entity 하이라이트, consistency 점수 표)의 front-end spec을 작성해 주세요. 발표 시연용 단일 화면이며 색상 외 텍스트 라벨 병기가 필요합니다.

### Architect Prompt

이 PRD를 입력으로 FinFact의 architecture 문서를 작성해 주세요. 핵심 결정 사항: (1) 모듈 구조(detection/visual_entity/text_entity/matching/fusion/demo)와 인터페이스 정의, (2) 단일 GPU 제약 하 모듈별 독립 학습 + fusion 결합 전략, (3) config 기반 ablation 실험 프레임, (4) Fakeddit 전처리 및 금융 subset 파이프라인 설계, (5) FinFact-Eval(Epic 6) 데이터 스키마와 평가 하네스가 ablation 러너에 두 번째 eval set으로 꽂히는 구조.

---

## Revision Notes (v1.1) — 2026-08-10, John (PM)

문서 간 정합성 리뷰에서 확인된 갭 3건을 반영했다. **Epic 1~5의 기존 구조·스토리 개요와 FR1~FR12 번호 체계는 유지**했고, 확장은 FR13~FR17 / NFR8~NFR9 / Epic 6으로만 이어 붙였다.

### 1. 금융 도메인 특화 데이터 스토리 부재 → Epic 6 신설 (FR13~FR17, NFR5 보강, NFR8~NFR9)

- **문제**: README §5와 DEV_PLAN §2.3이 "자체 금융 셋 300건+, mismatch 유형 annotation, 교차검증"을 프로젝트 정체성으로 내세우는데, Epic 1~5 어디에도 이 데이터를 만드는 스토리가 없었다. 결과적으로 "금융 특화"의 근거가 키워드 필터(FR11) 하나뿐이었고, held-out 금융 평가셋은 계획상 존재하지 않았다. (Story 5.1 AC 6과 5.2 AC 3은 이미 이 셋을 **전제**하고 있었으므로 dangling dependency이기도 했다.)
- **판단 — 기존 Epic 추가가 아닌 Epic 신설**: 이 작업은 수집·조작 생성·taxonomy·annotation·QA·패키징·측정 하네스로 6스토리 분량이고, Epic 1~4 어느 하나의 목표(모델 증분)에도 속하지 않는 독립 데이터 트랙이다. 기존 Epic에 끼워 넣으면 PO 승인된 Epic의 goal이 흔들린다. 다만 **번호 6 = 실행 순서 6이 아니며**, DEV_PLAN §5의 3~7주차에 Epic 2~4와 병행하고 Epic 5 착수 전에 완료해야 한다는 점을 Epic List에 명시했다.
- **(a) 출처·합법성**: NFR5를 확장해 자동 크롤링을 명시적으로 금지하고, 재사용 허용 공개 자료 + 사람이 수동 선별한 기사로 출처를 한정했다. 샘플별 URL·라이선스·수집일·이미지 해시 기록(Story 6.1)과 라이선스 미확인 이미지 제외·URL 기반 재배포(NFR9, Story 6.5)를 요구사항으로 못박았다.
- **(b) taxonomy**: FR14에 PERSON / ORG_LOGO / NUMBER / DATE / EVENT / NONE 6라벨 + primary/secondary + 근거 entity pair 스키마를 정의했다. DEV_PLAN §2.3의 4유형(PERSON/LOGO/NUMBER/EVENT)에 DATE를 추가한 이유는, FR3·FR4가 이미 DATE를 visual(OCR)·textual(NER) 양쪽에서 추출하도록 요구하므로 시점 모순이 별도 유형으로 측정 가능하기 때문이다.
- **(c) annotation 절차·품질 관리**: FR15와 Story 6.3~6.4. **"2인 교차검증"은 단일 인원 과제에서 성립하지 않으므로** 채택하지 않고, guideline 문서화 → 20건 파일럿 → 전수 annotation → 20% 표본 blind 재라벨링(7일 간격) intra-annotator Cohen's κ ≥ 0.8 → adjudication 기록으로 대체했다. 제3자(동료/조교) 2차 검토는 확보 가능하면 추가 증거로만 사용한다. *(→ v1.2에서 개정: 제3자 스팟체크 30건이 선택이 아닌 필수 게이트가 되었다. v1.2 §2 참조)*
- **(d) 측정 대상**: Story 6.6이 (i) A1~A6 전 구성의 금융 특화 분류 성능, (ii) matching 판정 정확도, (iii) OCR CER을 산출한다.
- **규모 판단**: 총 300건(Real 150 / Fake 150), Fake는 유형당 25~35건, OCR subset 50~80 crop. DEV_PLAN §2.3의 "실제 기사 200~500건 + Fake 별도"는 8~10주 단일 인원(모델 개발과 병행)에는 과하다. 유형별 지표에 최소한의 의미를 주려면 유형당 25건 이상이 필요하고 유형이 5종이므로 Fake 150이 하한이며, 클래스 균형을 위해 Real 150을 맞춰 총 300건이 된다 — 즉 300은 편의상의 숫자가 아니라 "유형별 분해 지표를 보고할 수 있는 최소치"다. 400건은 stretch로만 둔다. *(→ **v1.2에서 이 근거는 철회되었다.** 유형당 25건으로는 유형별 지표를 보고할 수 없다(CI 폭 27%p). 300건의 정당화는 "유형별 보고 최소치"가 아니라 "FR16 정답 소스 + 전체 단위 지표 + 커버리지"로 대체되었다. v1.2 §1 참조)*

### 2. Ablation 행 정의 불일치 → **DEV_PLAN 기준(A1~A6) 채택** (FR10 개정)

- **문제**: DEV_PLAN §6은 A1~A6 6행(A5 = textual entity features 단독 기여)인데, PRD FR10과 Epic 5 개요는 5구성이라 A5가 빠져 있었다. 핵심 주장의 통제군이 문서마다 달랐다. (Story 5.1은 이미 A1~A6로 작성되어 있어 PRD 쪽이 유일한 이탈이었다.)
- **판단 — 6행이 옳다**: A5가 없으면 A4→A6의 개선이 "textual entity feature를 넣은 효과"인지 "consistency(불일치 대조) 효과"인지 분리되지 않는다. 그런데 Epic 3(텍스트 entity)과 Epic 4(consistency)는 서로 다른 Epic의 산출물이고, 본 프로젝트의 핵심 주장은 후자다. A5는 그 주장의 통제군이므로 생략하면 핵심 가설이 입증 불가능해진다. 따라서 FR10을 A1~A6로 개정하고, A6의 개선폭은 **A5 대비**로 보고하도록 명시했다. Epic 5 Story 5.1 개요 문장도 동일하게 정정했다(스토리 파일 본문은 이미 A1~A6이므로 변경 불필요).

### 3. 측정 계획 없는 지표 → FR16 / FR17 + Story 6.6으로 측정 주체 부여

- **문제**: DEV_PLAN §4의 "Cross-modal matching MATCH/MISMATCH 판정 정확도"와 README §3의 "OCR CER"이 어떤 스토리의 AC에도 없어 측정 불가였다.
- **matching 판정 정확도(FR16)**: FinFact-Eval의 entity pair annotation(FR14)을 정답으로 삼아 3-way accuracy + MISMATCH P/R/F1 + 유형별 분해로 정의했다. Story 6.6이 산출하고 Story 5.2 AC 3이 소비한다. *(→ v1.2에서 **유형별 분해는 제거**되었다. 전체 단위 지표 + CI 병기 + 유형별 정성 사례로 대체. v1.2 §1 참조)*
- **OCR CER(FR17)**: FinFact-Eval에 CHART/DOCUMENT/TEXT_REGION crop과 정답 문자열 쌍으로 구성된 OCR 검증 subset(50~80건)을 두고 CER을 측정한다. **subset이 50 crop 미달이면 CER은 측정 불가로 판정하고 지표에서 제외**하며, DEV_PLAN §4의 "수치 추출 정확도(정성 샘플 검증)"로 대체하고 그 사실을 결과 리포트에 명시하도록 탈출 규칙을 함께 정의했다.
- 일반 규칙으로 NFR4에 "측정 스토리가 없는 지표는 문서에서 제거한다"를 추가했다.

### 후속 조치 (본 PRD 범위 밖, 담당자에게 이관)

- `docs/DEV_PLAN.md` §2.3의 규모(200~500건)·"2인 교차검증" 문구를 본 개정의 300건·intra-annotator κ 절차와 맞출 것.
- `README.md` §3의 OCR CER 항목에 FR17의 대체 규칙 각주를 달 것.
- Epic 6 스토리 상세는 `docs/bmad/stories/epic-6-financial-eval-set.md`에 작성됨 — PO 승인 대기 상태(Draft).

---

## Revision Notes (v1.2) — 2026-08-10, John (PM)

팀 리뷰(Analyst/PO/Dev/리드)에서 제기된 미해결 충돌 2건과 합의 사항을 반영했다. FR/NFR 번호는 추가 없이 **기존 FR10·FR11·FR13·FR16의 내용만 개정**했고, Epic 1~5 스토리 파일은 건드리지 않았다(해당 변경은 PO에게 이관).

### 1. 통계적 주장 수준 하향 — Analyst의 CI 논거 수용 (FR16, FR13, FR10, Epic 6 규모/ladder)

v1.1은 "유형당 25건이면 유형별 지표를 보고할 수 있다"고 전제했다. **이는 틀렸다.** 근거(문서에 보존하는 이유는 나중에 "왜 유형별 표가 없냐"는 질문에 답하기 위해서다):

> 유형당 n=30에서 MISMATCH recall 0.80을 관측하면 Wilson 95% CI는 **[0.63, 0.90], 폭 27%p**다. PERSON 0.85와 NUMBER 0.70이 나와도 두 CI는 완전히 겹친다. 즉 "어느 유형에서 약한가"는 이 표본에서 **통계적으로 판별 불가능**하다. "유형당 25건이 최소치"라는 v1.1의 표현은 *셀을 채운다*는 뜻이었지 *비교 가능하다*는 뜻이 아니었고, 그 둘을 혼동한 것이 오류다.

대응은 셋을 키우는 것이 아니라(단일 인원 제약상 불가) **주장 수준을 낮추는 것**이다:
- **FR16**: 결론적 보고 지표를 **전체 단위 1세트**(3-way accuracy + MISMATCH P/R/F1, pair 총계)로 한정하고, **유형별 지표 표는 산출·보고하지 않는다**고 명시. 유형별 특성은 **정성 사례 3~5건**(Story 5.2)으로 전환. per-pair 원자료는 파일로 남기되 표로 제시하지 않는다. CI 근거를 FR16 본문과 이 절에 이중으로 남겼다.
- **유형 배분 25~35건의 의미 정정**: 지표 산출용이 아니라 **커버리지 확보용**(특정 유형 쏠림 방지). Epic 6 규모 결정 문단에 반영.
- **FR10 — 핵심 가설 검증 셋 재지정**: A5→A6 검증은 **Fakeddit 금융 subset(5k~20k, 3-seed 평균)** 에서 한다. n=300에서 F1 1%p는 샘플 3건이고 McNemar 유의성에는 8~10%p 점프가 필요한데 이는 기대 가능한 효과 크기가 아니다. FinFact-Eval의 A1~A6 수치는 **유의성 주장 없는 보조 표**로 격하.
- **FR13 — FinFact-Eval 역할 3개로 재배치**: (i) FR16 판정 정확도의 **유일한** 정답 소스(Fakeddit엔 entity pair annotation이 없다 — 이것이 1차 존재 이유), (ii) 도메인 이전의 정성 확인, (iii) 데모·발표 예시. 이 역할 정의 하에서 **300건은 과소가 아니라 적정**이다. v1.1이 이 셋을 은연중 "가설 검정 표본"으로 취급한 것이 규모 논쟁의 원인이었다.

### 2. κ의 한계 — Dev 지적 수용 (Epic 6 Story 6.3·6.4)

κ는 **일관성**이지 **정확성**이 아니다. 같은 사람이 같은 실수를 두 번 반복하면 κ는 0.9가 나오면서 라벨은 틀려 있다. v1.1의 품질 게이트는 intra-annotator κ 단독이라 이 실패 모드에 무방비였다. 반영:
- (a) annotation guideline을 **라벨링 시작 전에 확정하고 버전을 고정**(v1.0 freeze). 라벨링 도중 규칙이 바뀌면 앞뒤 라벨의 기준이 달라져 κ도 정확성도 의미를 잃는다.
- (b) **제3자 스팟체크 30건을 별도 품질 게이트로 추가** — κ와 병렬로 통과해야 한다. 불일치 발견 시 guideline 개정 → 해당 범위 재라벨링.
- (c) **제3자 확보 불가 시 대안**: 정답이 명확한 seed 케이스 15건을 guideline에 박아두고(작성 시점에 판정 근거까지 명시), 라벨링 중간·종료 시 두 차례 자기 점검하여 seed 정확도를 보고한다. 이는 스팟체크의 완전한 대체가 아니며 그 한계를 리포트에 명시한다.
- (d) "κ는 정확성을 담보하지 못한다"는 한계 자체를 품질 리포트에 명문화.

### 3. FR11 조건부 트리거 (Dev 제안 수용)

"금융 subset이 작으면 학습이 아니라 평가 전용으로"를 **조건부로 수용**했다. 지금 문구를 바꾸지 않고 트리거를 건 이유는, 실규모를 보기 전에 학습 전략을 확정하면 5k 이상일 때 불필요하게 손해를 보기 때문이다. 판정 시점은 **1주차 필터링 직후·Epic 2 착수 전**, 기준은 필터링 후 train split 샘플 수 — ≥5k 현행, 3k~5k 2-stage, <3k 평가 전용 축소 + "금융 특화 학습" 주장 철회. 분기 결과는 `docs/results/`에 기록.

### 4. 실행 순서 강제 장치 (Analyst 지적 수용)

"Epic 번호 6이라 실행도 마지막으로 밀린다"는 지적은 타당하다. 문서에 "3주차 병행"이라 적는 건 강제력이 없다. **Epic 번호는 재배치하지 않는다**(승인된 번호를 흔들면 Story 5.1/5.2/5.3의 상호 참조가 깨진다). 대신 (a) **Epic 1에 Story 1.5 "FinFact-Eval 수집 착수 게이트"(출처 정책 확정 + 시드 20건) 신설** — `epic-1-baseline.md`는 PO 관할이므로 **작성은 PO에게 이관**하고 본 PRD에는 의도·범위만 기록, (b) Epic 2 착수 시 3주차 말 Real 30건 미달이면 그 주 GPU 대기시간을 수집에 배정, (c) 주간 체크포인트에 누적 건수 한 줄 고정.

### 5. 사람 시간 경합 (PO 지적 수용)

Story 2.1의 수작업 bbox와 Epic 6 수집은 둘 다 GPU가 아닌 **오너의 손 시간**을 쓰고 DEV_PLAN §5 기준 4주차에 충돌한다. 조정: **자체 bbox annotation 300장 → 150장 축소**(부족분은 LogoDet-3K 등 공개 데이터 재활용, CHART/DOCUMENT 낮은 mAP는 이미 허용된 리스크), 그리고 **4주차 bbox / 5주차 수집으로 번갈아 배치**. Story 2.1 본문 반영은 PO 관할이므로 결정만 기록한다.

### 후속 조치 (v1.2 추가분)

- **PO**: Story 1.5 신설(`epic-1-baseline.md`), Story 2.1 bbox 300→150 축소 반영(`epic-2-object-detection.md`), Story 5.1 AC 6·5.2 AC 3의 "유형별 정확도 표"를 "전체 지표 + 유형별 정성 사례"로 정정(`epic-5-eval-demo.md`).
- **DEV_PLAN**: §5 주차 배치(4주차 bbox / 5주차 수집), §6 발표 포인트에 "가설 검증은 Fakeddit 금융 subset 기준" 명기.
