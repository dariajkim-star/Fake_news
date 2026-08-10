# Project Brief: FinFact — Object-Level Multimodal Financial Misinformation Detection

> Detecting Financial Fake News through Visual–Textual Entity Inconsistency
> 작성자: Mary (BMAD Analyst) · 작성일: 2026-08-10

---

## Executive Summary

FinFact는 금융 뉴스 게시물(이미지 + 텍스트)에서 **이미지 속 시각적 증거와 기사 텍스트의 주장이 서로 다른 사실을 말하는지**(entity/event 불일치)를 탐지하여 Fake/Real 확률과 판단 근거를 출력하는 딥러닝 과제용 프로젝트다. 기존의 이미지 전체–텍스트 전체 비교(single CLIP similarity) 방식과 달리, Object Detection으로 이미지를 객체 단위 증거로 분해하고 NER/Event Extraction으로 텍스트 주장을 entity 단위로 추출한 뒤, 이를 하나씩 대조(cross-modal matching)하는 **object-level entity consistency** 접근을 취한다. 대상 도메인은 M&A 루머, 실적 조작, CEO 관련 가짜뉴스, 주가조작성 SNS 게시물 등 금융 가짜뉴스이며, 핵심 성과 목표는 ablation 실험을 통해 entity consistency 정보가 최종 분류 F1을 추가로 개선함을 입증하는 것이다.

## Problem Statement

- **금융 멀티모달 가짜뉴스의 확산**: 금융 도메인에서는 조작된 이미지(가짜 계약서, 합성된 인물 사진, 조작 차트)와 자극적인 텍스트를 결합한 게시물이 주가조작·투자 유인에 악용된다. 텍스트만 보는 탐지 모델은 "이미지가 증거처럼 보이는" 게시물에 취약하다.
- **기존 접근의 한계 — 이미지·텍스트 전체 비교**: 기존 멀티모달 탐지의 다수는 이미지 전체 embedding과 텍스트 전체 embedding의 similarity(예: CLIP 한 번의 비교)에 의존한다. 이 방식은 이미지의 전반적 주제가 텍스트와 유사하기만 하면 통과되어, "사진 속 인물은 A인데 기사는 B라고 주장", "차트의 수치·날짜가 기사 주장과 불일치" 같은 **세부(object-level) 불일치**를 놓친다.
- **금융 뉴스의 특수성**: 금융 가짜뉴스는 인물(CEO), 기업 로고, 계약서/공시 문서, 차트 수치 등 **구체적 entity 수준의 사실 주장**으로 구성되므로, entity 단위 대조가 특히 유효한 도메인이다.

## Proposed Solution

Object-level visual–textual entity inconsistency 탐지 파이프라인:

1. **IMAGE → Object Detection** (YOLOv8 또는 DETR): PERSON, LOGO, PRODUCT, CHART, DOCUMENT, TEXT_REGION 클래스의 bbox 검출
2. **bbox crop → Visual Entity Recognition**: PERSON→얼굴/인물 인식(예: "Jensen Huang"), LOGO→CLIP/로고 분류기(예: "NVIDIA"), CHART/TEXT_REGION→OCR로 수치·날짜 추출
3. **TEXT → NLP** (KoELECTRA / KLUE-BERT / DeBERTa fine-tuning): NER(PERSON, ORG, PRODUCT, LOCATION, DATE, MONEY) + Event/Relation Extraction(예: 삼성전자 —CONTRACT_WITH→ NVIDIA, AMOUNT ₩20조)
4. **Cross-modal Matching**: visual entity vs textual entity 정렬 → MATCH / MISMATCH / UNKNOWN
5. **Cross-modal Attention Fusion**: region feature × token feature attention → Entity consistency, Event consistency, Image/Text similarity 점수 → 최종 Fake probability

차별점: 단일 global similarity가 아닌 **word-region pair 수준의 대조**로, 어떤 entity가 mismatch인지 근거(explainability)까지 제시한다.

## Target Users

### 1차 사용자 (실제)
- **과제 평가자(교수/조교)**: 딥러닝 기법의 적용 지점(Object Detection, Transformer fine-tuning, Cross-modal Attention)과 실험 설계(ablation)의 타당성을 평가
- **발표 청중(수강생)**: 데모를 통해 "어떤 entity가 불일치했는지" 직관적으로 이해할 수 있어야 함

### 2차 사용자 (가상 시나리오)
- **금융정보 플랫폼 운영자**: 커뮤니티/뉴스 피드에 올라오는 이미지 첨부 게시물의 fake score를 자동 산출하여 모더레이션 우선순위 결정
- **개인투자자**: SNS에서 접한 "계약서 사진 첨부" 루머성 게시물의 신뢰도를 확인하고, mismatch 근거를 보고 판단

## Goals & Success Metrics

### Goals
- 금융 멀티모달 가짜뉴스에 대해 object-level entity consistency 기반 탐지 모델 완성
- **핵심 입증 목표**: ablation table에서 entity consistency feature 추가 시 F1이 추가 개선됨을 보여주는 것 (발표 포인트)

### Success Metrics
| 구성 요소 | 지표 |
|---|---|
| Object Detection | mAP@50, Precision, Recall |
| NER (Transformer fine-tuning) | Entity-level F1 |
| 최종 분류기 (Fusion) | Accuracy, Precision, Recall, F1, AUROC |
| Ablation | BERT only / ResNet only / BERT+Image / +Object Detection / +Entity consistency 단계별 F1 개선 확인 |

## MVP Scope

### In Scope
- **Phase 1 Baseline**: ResNet(이미지) + BERT(텍스트) → late fusion → Fake/Real 이진 분류 (데이터: Fakeddit)
- **Phase 2**: + YOLO object regions → Region Encoder → Cross-modal Attention
- **Phase 3**: + NER/Entity Extraction (텍스트 측)
- **Phase 4**: + Visual entity vs Textual entity consistency score → 최종 모델
- Ablation 실험 및 결과 표
- 추론 파이프라인(이미지+텍스트 입력 → fake score + mismatch entity 근거 출력) 및 데모(Streamlit/Gradio)

### Out of Scope
- 실시간 뉴스/SNS 크롤링 및 서비스 배포
- 다국어 지원(한국어/영어 중 데이터셋이 확보되는 언어에 집중)
- 딥페이크/이미지 forgery 자체 탐지 (entity 불일치만 다룸)
- 대규모 분산 학습, 모델 서빙 인프라, 사용자 계정/피드백 시스템

## Constraints

- **일정**: 과제 기간 8~10주 (Phase 1~4 단계별 완성 전략으로 리스크 분산)
- **컴퓨팅**: 단일 GPU 가정 (Colab/로컬 수준) — 모델 크기·batch size·학습 epoch 제한
- **데이터**: 공개 데이터셋만 사용 (Fakeddit 등). 금융 특화 라벨 데이터는 부분적 필터링/수작업 라벨링으로 보완
- **팀 규모/과제 성격**: 수업 과제 산출물 기준 — 재현 가능한 코드, 실험 결과, 발표 자료가 핵심 deliverable

## Risks

| 리스크 | 영향 | 완화 방안 |
|---|---|---|
| Fakeddit에 금융 도메인 샘플이 적음 | 도메인 특화 주장 약화 | 키워드 필터링으로 금융 subset 구성, 부족 시 일반 도메인 성능 + 금융 사례 study로 보완 |
| Visual Entity Recognition(인물/로고 식별) 정확도 낮음 | consistency score 노이즈 | UNKNOWN 클래스 허용, 신뢰도 threshold 적용, CHART/OCR 등 잘 되는 클래스 중심 분석 |
| Entity consistency가 F1 개선을 못 보일 가능성 | 핵심 발표 포인트 상실 | Phase별 중간 평가로 조기 감지, error analysis로 개선/실패 원인 분석 자체를 기여로 제시 |
| 단일 GPU로 다단계 파이프라인 학습 부담 | 일정 지연 | pretrained model 최대 활용(YOLO, CLIP, BERT 계열), 모듈별 독립 학습 후 fusion만 end-to-end |
| OCR/NER 오류 전파 | 최종 분류 성능 저하 | soft matching(embedding 기반), rule + model hybrid 매칭 |

## Technical Considerations (요약)

- **Vision**: YOLOv8 또는 DETR (detection), CLIP 임베딩(로고/시각 entity), 얼굴 인식 모델, 딥러닝 기반 OCR
- **NLP**: KoELECTRA / KLUE-BERT / DeBERTa fine-tuning — NER + Event/Relation Extraction
- **Fusion**: Cross-modal Attention (region feature × token feature), consistency feature를 최종 분류기에 결합
- **데이터**: Fakeddit (100만+ text-image pair, 2/3/6-way label)
- **데모/추론**: Streamlit 또는 Gradio, 입력(이미지+텍스트) → fake score + mismatch entity 시각화
- **평가**: 컴포넌트별(mAP, NER F1) + end-to-end(Accuracy/P/R/F1/AUROC) + ablation

## 관련 연구 (Related Work)

- **EM-FEND** (arXiv:2108.10509): visual/textual entity inconsistency를 가짜뉴스 탐지에 도입 — 본 프로젝트의 핵심 아이디어의 직접적 근거
- **CFFN** (arXiv:2311.01807): word-region consistency 기반 fine-grained cross-modal 대조 — Phase 2의 region-token attention 설계 근거
- **Event-Radar** (ACL 2024): event-level graph를 이용한 멀티모달 탐지 — Phase 3~4의 event consistency 개념 근거
- **PROPOR 2026**: financial fake news multimodal framework — 금융 도메인 적용의 선행 사례
- **Fakeddit** (arXiv:1911.03854): 100만+ text-image 멀티모달 가짜뉴스 데이터셋(2/3/6-way label) — 학습/평가 데이터 기반
