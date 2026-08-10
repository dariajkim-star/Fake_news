# FinFact Product Requirements Document (PRD)

> Object-Level Multimodal Financial Misinformation Detection
> 작성자: John (BMAD PM) · 작성일: 2026-08-10 · 버전: v1.0

---

## Goals and Background Context

### Goals

- 금융 뉴스 게시물(이미지 + 텍스트)에 대해 Fake/Real 확률과 판단 근거를 출력하는 object-level multimodal 탐지 모델을 완성한다.
- 단일 global CLIP similarity 방식이 놓치는 세부 불일치("사진 속 인물은 A인데 기사는 B", "차트 수치·날짜가 기사 주장과 불일치")를 word-region pair 수준의 cross-modal matching으로 탐지한다.
- **핵심 입증 목표**: ablation table(BERT only / ResNet only / BERT+Image / +Object Detection / +Entity consistency)에서 entity consistency feature 추가가 최종 분류 F1을 추가 개선함을 실험으로 입증한다.
- 어떤 entity가 MISMATCH인지 근거를 시각적으로 제시하는 데모(Gradio/Streamlit)를 제공하여 explainability를 확보한다.
- 재현 가능한 코드, 실험 결과, 발표 자료를 과제 deliverable로 산출한다.

### Background Context

금융 도메인에서는 조작된 이미지(가짜 계약서, 합성 인물 사진, 조작 차트)와 자극적인 텍스트를 결합한 게시물이 주가조작·투자 유인에 악용된다. 기존 멀티모달 탐지의 다수는 이미지 전체 embedding과 텍스트 전체 embedding의 similarity 한 번(single CLIP similarity)에 의존하기 때문에, 이미지의 전반적 주제만 유사하면 통과되어 object-level 불일치를 놓친다. 금융 가짜뉴스는 인물(CEO), 기업 로고, 계약서/공시 문서, 차트 수치 등 구체적 entity 수준의 사실 주장으로 구성되므로 entity 단위 대조가 특히 유효하다.

FinFact는 Object Detection으로 이미지를 객체 단위 증거로 분해하고, NER/Event Extraction으로 텍스트 주장을 entity 단위로 추출한 뒤, 이를 하나씩 대조하는 object-level entity consistency 접근을 취한다. 연구 근거는 EM-FEND(arXiv:2108.10509), CFFN(arXiv:2311.01807), Event-Radar(ACL 2024), PROPOR 2026 financial fake news framework, Fakeddit 데이터셋(arXiv:1911.03854)이다. 본 프로젝트는 8~10주 딥러닝 과제로, 단일 GPU 환경에서 Phase 1~4 단계별 완성 전략으로 진행한다.

### Change Log

| Date | Version | Description | Author |
|---|---|---|---|
| 2026-08-10 | v1.0 | 최초 PRD 작성 (project-brief 기반) | John (PM) |

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
- **FR10**: 시스템은 baseline(ResNet+BERT late fusion)부터 최종 모델까지 각 구성(BERT only / ResNet only / BERT+Image / +Object Detection / +Entity consistency)을 동일 평가 셋에서 비교하는 ablation 실험을 지원해야 한다.
- **FR11**: 학습/평가 데이터로 Fakeddit을 사용하며, 키워드 필터링으로 금융 subset을 구성할 수 있어야 한다.
- **FR12**: Visual Entity Recognition 신뢰도가 threshold 미만인 경우 UNKNOWN으로 처리하여 오류 전파를 제한해야 한다 (soft matching / rule + model hybrid 허용).

### Non-Functional Requirements

- **NFR1**: 모든 학습은 단일 GPU(Colab/로컬 수준)에서 수행 가능해야 한다 — pretrained model(YOLO, CLIP, BERT 계열) 최대 활용, 모듈별 독립 학습 후 fusion 단계 결합.
- **NFR2**: 단일 샘플(이미지+텍스트) end-to-end 추론은 데모 시연에 적합한 시간(GPU 기준 수 초 이내)에 완료되어야 한다.
- **NFR3**: 실험은 재현 가능해야 한다 — random seed 고정, config 파일 기반 실험 관리, 데이터 전처리/분할 스크립트 버전 관리.
- **NFR4**: 컴포넌트별 평가 지표를 산출해야 한다 — Object Detection: mAP@50/Precision/Recall, NER: entity-level F1, 최종 분류기: Accuracy/Precision/Recall/F1/AUROC.
- **NFR5**: 공개 데이터셋만 사용한다 (Fakeddit + 자체 필터링/수작업 라벨링 금융 subset). 크롤링 기반 실시간 데이터 수집은 범위 외.
- **NFR6**: 코드는 모듈화되어야 한다 — detection, entity recognition, NER, matching, fusion이 독립적으로 학습·평가·교체 가능해야 ablation이 성립한다.
- **NFR7**: Out of scope 준수 — 실시간 크롤링/서비스 배포, 다국어 동시 지원, 딥페이크/forgery 자체 탐지, 분산 학습, 사용자 계정 시스템은 구현하지 않는다.

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

- **Story 5.1**: Ablation 실험 — BERT only / ResNet only / BERT+Image / +Object Detection / +Entity consistency 전 구성 동일 조건 평가, 결과 표 작성
- **Story 5.2**: Error analysis — MISMATCH 오탐/미탐 사례 분석, 컴포넌트별 오류 전파 분석(개선 실패 시 원인 분석 자체를 기여로 제시)
- **Story 5.3**: 데모 앱 — Gradio/Streamlit UI(이미지+텍스트 입력 → fake score, bbox/entity 하이라이트, consistency 점수 표), 예시 샘플 탑재
- **Story 5.4**: 발표 자료 및 재현 문서 — 실험 결과 정리, README/재현 가이드, 발표 슬라이드

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
| Epic & story structure | PASS | 5 Epic, 각 Epic이 배포 가능한 증분(ablation 단계)과 일치 |
| Technical guidance | PASS | PyTorch/HuggingFace/Ultralytics, 데이터셋 확정 |
| Cross-functional requirements | PARTIAL | Detection 라벨링 데이터 규모·비용은 Epic 2 착수 시 구체화 필요 |
| Clarity & communication | PASS | 한국어 + 기술 용어 영어 유지 |

**주요 리스크(모니터링)**: (1) Fakeddit 금융 subset 규모 부족, (2) Visual Entity Recognition 정확도로 인한 consistency 노이즈, (3) entity consistency의 F1 개선 미달 가능성 — Epic별 중간 평가로 조기 감지, error analysis를 대안 기여로 준비.

---

## Next Steps

### UX Expert Prompt

이 PRD의 UI Design Goals 섹션을 기반으로 Gradio/Streamlit 데모 화면(입력 영역, fake score 표시, bbox/entity 하이라이트, consistency 점수 표)의 front-end spec을 작성해 주세요. 발표 시연용 단일 화면이며 색상 외 텍스트 라벨 병기가 필요합니다.

### Architect Prompt

이 PRD를 입력으로 FinFact의 architecture 문서를 작성해 주세요. 핵심 결정 사항: (1) 모듈 구조(detection/visual_entity/text_entity/matching/fusion/demo)와 인터페이스 정의, (2) 단일 GPU 제약 하 모듈별 독립 학습 + fusion 결합 전략, (3) config 기반 ablation 실험 프레임, (4) Fakeddit 전처리 및 금융 subset 파이프라인 설계.
