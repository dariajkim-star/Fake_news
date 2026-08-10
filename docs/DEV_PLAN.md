# FinFact 개발 계획 (DEV_PLAN)

**프로젝트**: FinFact — Object-Level Multimodal Financial Misinformation Detection
**목표**: 뉴스 이미지 속 시각적 증거(objects/entities)와 기사 텍스트의 주장(entities/events)을 객체 단위로 대조하여 금융 가짜뉴스(Fake/Real)를 탐지
**총 기간 가정**: 8~10주

---

## 1. 공통 기술 스택

| 구분 | 사용 기술 |
|---|---|
| Framework | PyTorch, PyTorch Lightning(선택) |
| 텍스트 모델 | HuggingFace Transformers — BERT, KLUE-BERT, KoELECTRA, DeBERTa |
| 이미지 모델 | torchvision ResNet-50, CLIP (openai/clip 또는 open_clip) |
| Object Detection | Ultralytics YOLOv8 (fallback: HuggingFace DETR) |
| OCR | EasyOCR 또는 PaddleOCR (딥러닝 기반) |
| 얼굴 인식 | InsightFace 또는 facenet-pytorch (제한된 유명인 클래스) |
| 데이터/실험 | pandas, scikit-learn, wandb(또는 TensorBoard) |
| 데모 | Streamlit 또는 Gradio |

---

## 2. 데이터 준비

### 2.1 Fakeddit 다운로드/전처리
- [ ] Fakeddit (arXiv:1911.03854) 메타데이터 TSV 다운로드 (train/val/test)
- [ ] `image_url` 기반 이미지 크롤링 스크립트 작성 (실패 URL 로그, 재시도, 최대 해상도 제한)
- [ ] multimodal 샘플만 필터링 (`hasImage == True`, 이미지 다운로드 성공분)
- [ ] 2-way label(Fake/Real) 기준 정리, 6-way label은 보조 분석용으로 보관
- [ ] 텍스트 클리닝(URL/이모지 제거), 이미지 무결성 검사(corrupt 제거), train/val/test split 고정(seed)

### 2.2 금융 서브셋 필터링 전략
- [ ] 키워드 사전 구축: 회사명(ticker 매핑), "stock, earnings, merger, acquisition, CEO, SEC, IPO, revenue, crypto" 등 금융 도메인 용어
- [ ] Fakeddit title에 키워드 매칭 + subreddit 기반 필터(예: 경제/비즈니스 관련 subreddit)
- [ ] 필터 결과가 부족할 경우: zero-shot 분류(BART-MNLI 등)로 "finance" 토픽 스코어링 → threshold 필터
- [ ] 최종 금융 서브셋 목표: 5k~20k 샘플, 클래스 비율 확인 및 stratified split

### 2.3 자체 소규모 금융 가짜뉴스 셋 구축
- [ ] 수집: 실제 금융 뉴스(경제지 기사 + 대표 이미지) 200~500건 → Real
- [ ] 합성 Fake 생성: 이미지-텍스트 mismatch 조작 (예: NVIDIA 로고 이미지 + 삼성전자 계약 기사, 인물 사진 교체, 수치 변조 headline)
- [ ] 라벨 스키마: fake/real + mismatch 유형(PERSON / LOGO(ORG) / NUMBER / EVENT) annotation
- [ ] 최소 300건 이상, 2인 교차 검증(라벨 합의) → held-out 평가 전용으로 사용

---

## 3. Phase별 계획

### Phase 1 — Baseline (Late Fusion)
**목표**: ResNet(이미지) + BERT(텍스트) → late fusion binary classifier로 end-to-end 파이프라인/평가 체계 확립

**작업**
- [ ] 데이터 로더 구현 (image + text pair, Fakeddit 2-way)
- [ ] ResNet-50 (torchvision, ImageNet pretrained) 이미지 인코더
- [ ] BERT/KLUE-BERT (HuggingFace) 텍스트 인코더 fine-tuning
- [ ] Late fusion: [CLS] embedding ⊕ image pooled feature → MLP classifier
- [ ] Text-only / Image-only 단일 modality baseline도 함께 학습 (ablation 행 확보)
- [ ] 학습/평가 루프, metric 로깅(wandb), checkpoint 저장 체계

**모델/라이브러리**: PyTorch, torchvision, HuggingFace Transformers
**성공 기준**: Fakeddit 2-way test Accuracy ≥ 80%, F1 ≥ 0.78; 재현 가능한 학습 스크립트 완성

### Phase 2 — Object Regions + Cross Attention
**목표**: YOLO 객체 검출 결과(region features)를 텍스트 토큰과 cross-modal attention으로 결합

**작업**
- [ ] YOLOv8 pretrained(COCO)로 PERSON 등 기본 클래스 검출 → 파이프라인 통합
- [ ] 커스텀 클래스(LOGO, CHART, DOCUMENT, TEXT_REGION) 소규모 fine-tuning: 오픈 데이터(LogoDet-3K subset, chart 데이터) + 자체 annotation 200~500장
- [ ] bbox crop → Region Encoder (ResNet 또는 CLIP visual encoder)로 region feature 추출 (top-K regions, K=5~10)
- [ ] Cross-modal Attention 모듈: region features × BERT token features (multi-head attention)
- [ ] Phase 1 대비 성능 비교 실험

**모델/라이브러리**: Ultralytics YOLOv8 (또는 DETR), CLIP, PyTorch attention
**성공 기준**: detector mAP@50 ≥ 0.6 (커스텀 클래스), 최종 분류 F1이 Phase 1 대비 +1~2%p

### Phase 3 — Textual Entity/Event Extraction
**목표**: 텍스트에서 NER + 간단한 event/relation 추출을 결합해 entity-aware 표현 확보

**작업**
- [ ] NER fine-tuning: KLUE-BERT/KoELECTRA(한국어) 또는 DeBERTa(영어, CoNLL/OntoNotes) — PERSON, ORG, PRODUCT, LOCATION, DATE, MONEY
- [ ] 금융 도메인 소량 annotation으로 NER 보강 (MONEY/ORG 중심)
- [ ] Event/Relation 추출: 규칙 기반 트리거(계약, 인수, 실적 발표 등) + dependency/템플릿 매칭 (경량화)
- [ ] Entity embedding을 fusion 입력에 추가 (entity type embedding 포함)

**모델/라이브러리**: HuggingFace Transformers (token classification), spaCy(보조)
**성공 기준**: NER F1 ≥ 0.85 (일반 벤치마크 기준), entity feature 추가 시 분류 성능 유지 또는 개선

### Phase 4 — Visual–Textual Entity Consistency (최종 모델)
**목표**: visual entity vs textual entity 정렬(MATCH/MISMATCH/UNKNOWN) → consistency score를 fusion에 통합, 최종 모델 및 근거 제시 기능 완성

**작업**
- [ ] Visual Entity Recognition: PERSON→얼굴 인식(제한된 유명인 리스트), LOGO→CLIP zero-shot/로고 분류기, CHART/TEXT_REGION→OCR 수치·날짜 추출
- [ ] Cross-modal matching 모듈: entity 이름 문자열/embedding 유사도 기반 정렬 → MATCH/MISMATCH/UNKNOWN
- [ ] Entity consistency score + Event consistency score + 전역 CLIP similarity를 최종 fusion에 concat
- [ ] 전체 ablation 실험 수행 및 결과 표 작성
- [ ] 추론 파이프라인 (입력: 이미지+텍스트 → fake score + mismatch entity 근거 출력)
- [ ] Streamlit/Gradio 데모, 자체 금융 셋으로 최종 평가

**모델/라이브러리**: CLIP, InsightFace/facenet-pytorch, EasyOCR/PaddleOCR
**성공 기준**: ablation에서 entity consistency 추가 시 F1/AUROC 유의미한 개선(+1%p 이상), 자체 금융 셋에서 mismatch 유형별 정성 사례 3개 이상 시연

---

## 4. 평가 지표

| 모듈 | 지표 |
|---|---|
| Object Detection | mAP@50, Precision, Recall |
| NER | Entity-level F1 |
| OCR | 수치 추출 정확도(정성/샘플 검증) |
| 최종 분류 | Accuracy, Precision, Recall, F1, AUROC |
| Cross-modal matching | MATCH/MISMATCH 판정 정확도 (자체 셋 annotation 기준) |

---

## 5. 주차별 스케줄 (10주 기준, 8주 시 괄호 항목 축소)

| 주차 | 내용 | Phase |
|---|---|---|
| 1주 | Fakeddit 다운로드, 전처리, 금융 서브셋 필터링, EDA | 데이터 |
| 2주 | Baseline 구현 (ResNet+BERT late fusion) 및 단일 modality 실험 | 1 |
| 3주 | Baseline 튜닝/평가 확정, 자체 금융 셋 수집 시작 | 1 |
| 4주 | YOLO 통합, 커스텀 클래스 annotation 및 fine-tuning | 2 |
| 5주 | Region Encoder + Cross Attention 구현/실험 | 2 |
| 6주 | NER fine-tuning, event/relation 추출 | 3 |
| 7주 | Entity feature fusion 실험, 자체 금융 셋 완성 | 3 |
| 8주 | Visual entity recognition (얼굴/로고/OCR) + matching 모듈 | 4 |
| 9주 | 최종 모델 학습, ablation 전체 실험 (8주 계획 시 8주차와 병행 압축) | 4 |
| 10주 | 데모, 리포트, 발표자료 작성 (8주 계획 시 축소) | 마무리 |

---

## 6. Ablation 실험 설계

| # | 구성 | 목적 |
|---|---|---|
| A1 | BERT only (text) | 텍스트 단독 baseline |
| A2 | ResNet only (image) | 이미지 단독 baseline |
| A3 | BERT + ResNet late fusion | 단순 multimodal 효과 (Phase 1) |
| A4 | A3 + YOLO object regions + Cross Attention | object-level 증거 효과 (Phase 2) |
| A5 | A4 + Textual entity features | entity-aware 표현 효과 (Phase 3) |
| A6 | A5 + Entity/Event consistency score (최종) | 불일치 정보의 추가 개선 (Phase 4) |

- 모든 실험: 동일 split/seed(3-seed 평균 권장), 지표 Accuracy/F1/AUROC
- 발표 포인트: A3→A4→A6에서 F1의 단계적 개선 곡선 제시

---

## 7. 리스크와 대응

| 리스크 | 영향 | 대응 |
|---|---|---|
| Visual entity recognition 난이도 (open-set 인물/로고 인식 불가능 수준) | Phase 4 지연 | 유명인 30~50명, 로고 50~100개로 **클래스 제한**; 미인식은 UNKNOWN 처리하고 consistency 계산에서 제외 |
| OCR 품질 저하 (저해상도 차트/합성 이미지) | 수치 대조 신뢰도 하락 | 이미지 upscale 전처리, confidence threshold, 숫자 정규화 규칙; OCR feature는 보조 신호로만 사용 |
| Fakeddit 이미지 URL 대량 유실 | 데이터 부족 | 초기(1주차)에 다운로드 성공률 확인, 부족 시 전체 셋에서 목표 수량 확보 후 금융 필터 완화 |
| 금융 서브셋 규모 부족 | 도메인 일반화 저하 | 전체 Fakeddit으로 pretrain 후 금융 서브셋 fine-tuning(2-stage); 자체 합성 셋으로 보강 |
| 커스텀 detector 클래스 annotation 비용 | Phase 2 지연 | LogoDet-3K 등 공개 데이터 재활용, annotation은 500장 이내로 제한, CHART/DOCUMENT는 낮은 mAP 허용 |
| Cross Attention 학습 불안정 / 과적합 | 성능 미개선 | region 수 K 제한, dropout/layer norm, 백본 freeze 후 head만 학습하는 warmup |
| 한국어/영어 데이터 혼재 | 모델 선택 혼란 | 주 실험은 영어(Fakeddit+DeBERTa/BERT)로 고정, 한국어(KoELECTRA)는 자체 셋 데모용으로 분리 |
| GPU 자원 부족 | 학습 시간 초과 | 백본 freeze + head fine-tuning, mixed precision(AMP), 이미지 224px 고정, 서브셋 샘플링 |

---

## 8. 최종 산출물

- [ ] **모델 checkpoint**: YOLO detector(fine-tuned), NER 모델, fusion classifier (Phase별 checkpoint 포함)
- [ ] **추론 파이프라인**: 이미지+텍스트 입력 → fake score + mismatch entity 근거 출력 (CLI/모듈)
- [ ] **실험 리포트**: ablation table(A1~A6), 모듈별 지표(mAP, NER F1), 정성 사례 분석
- [ ] **데모**: Streamlit/Gradio 웹 데모 (근거 시각화: bbox + entity 매칭 결과 표시)
- [ ] **발표자료**: 문제 정의 → 아키텍처 → ablation 개선 곡선 → 데모 시연 구성
- [ ] **데이터 산출물**: 금융 서브셋 필터링 스크립트, 자체 금융 가짜뉴스 셋(annotation 포함)
