# Epic 2: Object Detection 통합 — User Stories

> 작성자: Bob (BMAD Scrum Master) · 작성일: 2026-08-10 · 근거 문서: `docs/bmad/prd.md` (FR2, FR10, NFR1, NFR4, NFR6)
> Epic Goal: 이미지를 객체 단위 증거로 분해하는 detector를 확보하고, region feature를 cross-modal attention으로 텍스트와 결합해 baseline 대비 개선을 확인한다.

---

## Story 2.1: Detection 데이터 준비

### Status

Approved

### Story

**As a** ML 엔지니어,
**I want** PERSON/LOGO/PRODUCT/CHART/DOCUMENT/TEXT_REGION 6-class detection 학습·평가 데이터셋을 pretrained 클래스 매핑과 부분 수작업 라벨링으로 구성하고 싶다,
**so that** YOLOv8 fine-tuning과 mAP@50 평가에 사용할 수 있는 일관된 라벨 체계의 데이터를 확보한다.

### Acceptance Criteria

1. 프로젝트 공통 클래스 정의 파일(`src/vision/classes.py` 또는 config)에 6개 클래스가 고정 index로 정의된다: `0: PERSON, 1: LOGO, 2: PRODUCT, 3: CHART, 4: DOCUMENT, 5: TEXT_REGION`.
2. 라벨 소스 전략이 문서화되고 스크립트로 구현된다:
   - PERSON: COCO pretrained YOLOv8의 `person` 클래스 → PERSON으로 매핑 (pseudo-label)
   - LOGO: 공개 로고 detection 데이터셋(예: LogoDet-3K/FlickrLogos 일부) 또는 CLIP zero-shot 후보 → 수작업 검수
   - PRODUCT: COCO 사물 클래스 subset(laptop, cell phone, bottle 등) → PRODUCT로 매핑
   - CHART/DOCUMENT/TEXT_REGION: 문서·차트 공개 데이터(예: DocLayNet/PubLayNet 일부, ICDAR text detection) 매핑 + Fakeddit 금융 subset 이미지 수작업 라벨링
3. Fakeddit 금융 subset에서 샘플링한 이미지 **최소 150장**에 대해 수작업 bbox 라벨링이 완료된다 (LabelImg/CVAT/Label Studio 중 택1, YOLO txt format으로 export). 부족분은 공개 데이터셋(LogoDet-3K/FlickrLogos, DocLayNet/PubLayNet 등) 재활용을 우선한다 — 라벨링 인력 시간이 4주차에 Epic 6 수집과 충돌하므로 자체 annotation은 공개 데이터로 대체 불가능한 부분에 집중한다.
4. 최종 데이터셋이 Ultralytics YOLO format(`images/{train,val,test}`, `labels/{train,val,test}`, `dataset.yaml`)으로 생성되며, split 비율은 대략 80/10/10, seed 고정으로 재현 가능하다.
5. 데이터 검증 스크립트가 존재한다: 클래스별 instance 수 분포, 라벨 없는 이미지 수, bbox 좌표 정규화(0~1) 유효성 검사 결과를 출력한다.
6. 클래스별 최소 instance 수가 리포트되고, 심각하게 부족한 클래스(예: 50개 미만)는 augmentation 또는 추가 라벨링 필요로 명시된다.
7. **pseudo-label 앵커링 방지 gold set**: 위 150장과 별개로, pseudo-label 초안을 **보지 않고 맨손으로 라벨링한 gold set 30장**을 구축한다. 이 gold set을 정답으로 COCO pretrained pseudo-label의 **class별 recall(빠뜨린 박스 비율)**을 측정·리포트한다. pseudo-label 초안 위에 수정만 하는 방식은 누락 박스가 구조적으로 보이지 않으므로, gold set은 반드시 초안 없이 작성한다. gold set은 학습에 사용하지 않고 평가 전용으로 격리한다.

### Tasks / Subtasks

- [ ] 클래스 정의 및 config 작성 (AC: 1)
  - [ ] `src/vision/classes.py`에 6-class enum/dict 정의, `configs/detector.yaml`에 반영
- [ ] pretrained 클래스 매핑 pseudo-labeling 스크립트 작성 (AC: 2)
  - [ ] `scripts/detection/pseudo_label_coco.py`: COCO pretrained YOLOv8x로 Fakeddit 금융 subset 이미지 추론 → person/사물 클래스를 PERSON/PRODUCT로 매핑, confidence ≥ 0.5만 채택
  - [ ] 외부 공개 데이터셋(LOGO, CHART/DOCUMENT/TEXT_REGION) 다운로드·클래스 매핑 변환 스크립트 작성
- [ ] 수작업 라벨링 진행 (AC: 3)
  - [ ] 라벨링 도구 선정 및 가이드라인 문서(클래스 판정 기준, 애매 케이스 규칙) 작성 — `docs/bmad/labeling-guide.md`
  - [ ] Fakeddit 금융 subset에서 이미지 150장 이상 샘플링(seed 고정) 후 bbox 라벨링, YOLO txt export
- [ ] YOLO format 데이터셋 빌드 (AC: 4)
  - [ ] `scripts/detection/build_dataset.py`: pseudo-label + 수작업 label 병합, 중복 제거, split 생성, `dataset.yaml` 출력
- [ ] 데이터 검증 및 통계 리포트 (AC: 5, 6)
  - [ ] `scripts/detection/validate_dataset.py`: 클래스 분포/bbox 유효성/누락 검사, CSV 리포트 출력
- [ ] gold set 30장 맨손 라벨링 (AC: 7)
  - [ ] pseudo-label 미노출 상태로 라벨링(작업자에게 초안 파일 제공 금지), 별도 디렉토리 격리
  - [ ] `scripts/detection/eval_pseudo_label.py`: gold set 대비 pseudo-label class별 recall/precision 리포트
- [ ] Testing (모든 AC)
  - [ ] unit test: 클래스 매핑 함수, bbox 좌표 변환(xyxy↔YOLO normalized) 함수

### Dev Notes

- **저장 위치**: 데이터는 `data/annotations/` 하위(architecture Source Tree 준수), 코드는 `src/vision/`, 스크립트는 `scripts/detection/`(architecture의 `scripts/` 하위 세분화) — Epic 1 스캐폴딩의 monorepo 구조 준수, NFR6 모듈화.
- **pseudo-label 품질**: COCO pretrained 매핑은 recall 확보용이며 precision은 수작업 검수로 보완. confidence threshold는 config로 노출.
- **클래스 판정 기준(라벨링 가이드 핵심)**: CHART는 축/데이터 시각화가 있는 영역, DOCUMENT는 계약서/공시 등 문서 전체 페이지 형태, TEXT_REGION은 이미지 내 임의의 텍스트 블록(자막, 워터마크, 스크린샷 텍스트). DOCUMENT 내부의 텍스트는 별도 TEXT_REGION으로 중복 라벨링하지 않는다.
- **재현성**: 샘플링/분할 seed는 전역 config의 seed 사용 (NFR3).
- **리스크**: PRD Checklist에 명시된 대로 라벨링 규모·비용은 이 스토리 착수 시 확정 — mAP 결과에 따라 Story 2.2에서 증량 판단.
- **앵커링 리스크(James 제기, PO 수용)**: pseudo-label 초안을 깔고 수정하게 하면 작업자는 "잘못 그려진 박스"는 고치지만 "아예 없는 박스"는 인지하지 못한다. 이 편향은 라벨 품질 지표에 잡히지 않고 mAP를 낙관적으로 만든다. gold set 30장이 유일한 방어선이므로 규모를 줄이더라도 이 30장은 유지한다.
- **규모 축소 근거**: 300 → 150장. 4주차에 Epic 6 수집(주 30~40건)과 라벨링이 같은 사람 시간을 놓고 경합한다. mAP@50이 목표(0.5) 미달이면 Story 2.2에서 증량을 판단한다(기존 방침 유지).

### Testing

- pytest 기반 unit test: 클래스 매핑, 좌표 변환, split 결정성(동일 seed → 동일 split).
- `validate_dataset.py` 실행 결과(클래스 분포 리포트)를 스토리 완료 증빙으로 첨부.

---

## Story 2.2: YOLOv8 학습/적용

### Status

Approved

### Story

**As a** ML 엔지니어,
**I want** Story 2.1의 6-class 데이터셋으로 YOLOv8을 fine-tuning하고 mAP@50/Precision/Recall로 평가하고 싶다,
**so that** 파이프라인 하위 단계(region encoder, visual entity recognition)가 신뢰할 수 있는 bbox 검출 결과를 사용할 수 있다.

### Acceptance Criteria

1. Ultralytics YOLOv8 기반 학습 스크립트가 config(yaml) 기반으로 동작한다 — model size(n/s/m), epochs, batch size, imgsz, seed가 config로 제어된다.
2. COCO pretrained weight에서 시작하는 fine-tuning이 단일 GPU에서 완료된다 (기본: `yolov8s.pt`, imgsz=640, 약 50 epochs — GPU 메모리에 맞게 batch 조정).
3. held-out test split에서 mAP@50, mAP@50-95, 클래스별 Precision/Recall이 산출되어 `results/detection/` 하위에 저장된다 (NFR4).
4. fine-tuned 모델과 pretrained-only(COCO 매핑) baseline의 mAP@50 비교표가 기록된다.
5. 추론 wrapper 모듈 `src/vision/detector.py`가 구현된다: 이미지 경로/PIL 입력 → architecture의 Detection 스키마(`region_id, class_id, class_name, bbox [x1,y1,x2,y2], conf`) 리스트 반환. confidence threshold(기본 0.4)와 max detections(기본 R=16), NMS IoU(기본 0.5)는 config로 제어 (architecture Detector 컴포넌트 스펙 준수).
6. Fakeddit 금융 subset 전체 이미지에 대한 배치 추론 스크립트가 detection 결과를 JSON(이미지 id → detections)으로 캐싱한다 — 이후 Story 2.3/Epic 4의 입력.
7. 대표 이미지 20장에 대한 bbox 시각화 샘플이 저장되어 정성 확인이 가능하다.
8. **하드 게이트**: 본 스토리를 완료(Done) 처리하려면 **Story 6.1의 Real 샘플 수집이 착수되어 최소 30건이 `sources.jsonl`에 기록되고 `scripts/validate_eval_sources.py` 검증을 통과**해야 한다. Epic 6은 번호가 6이지만 실행은 3~7주차 병행이며, 이 게이트는 문서 주석이 아닌 완료조건으로 순서를 강제하기 위한 것이다. 검증 통과 로그를 완료 증빙으로 첨부한다.

### Tasks / Subtasks

- [ ] 학습 config 및 스크립트 작성 (AC: 1, 2)
  - [ ] `configs/detector.yaml`: model/epochs/batch/imgsz/seed/augmentation 설정
  - [ ] `scripts/detection/train_yolo.py`: Ultralytics API(`YOLO.train`)로 fine-tuning, best.pt를 `models/detection/`에 저장
- [ ] 평가 수행 (AC: 3, 4)
  - [ ] `scripts/detection/eval_yolo.py`: `YOLO.val`로 test split 평가, 클래스별 metric CSV 저장
  - [ ] pretrained-only baseline 평가 후 비교표(`results/detection/comparison.md`) 작성
- [ ] 추론 wrapper 구현 (AC: 5)
  - [ ] `src/vision/detector.py`: `Detector` 클래스 (load, predict, predict_batch), architecture Detection 스키마 준수 출력 dataclass 정의
- [ ] 배치 추론 캐싱 (AC: 6)
  - [ ] `scripts/detection/run_inference.py`: 금융 subset 전체 → `data/detection/cache/detections.json`
- [ ] 시각화 (AC: 7)
  - [ ] bbox + 클래스 라벨 overlay 이미지 20장을 `results/detection/samples/`에 저장
- [ ] Testing (AC: 5)
  - [ ] unit test: Detector 출력 스키마, threshold 필터링, 빈 검출 처리

### Dev Notes

- **모델 선택**: 단일 GPU 제약(NFR1)으로 yolov8s 기본, mAP 부족 시 yolov8m까지만 고려. DETR은 대안으로 문서화만 하고 구현하지 않는다.
- **클래스 불균형**: Story 2.1 리포트에서 부족한 클래스는 Ultralytics 기본 augmentation(mosaic, hsv, fliplr) 유지 + 필요 시 copy-paste augmentation 검토.
- **평가 기준**: 목표 mAP@50 ≥ 0.5 (전체 평균). 미달 클래스는 error analysis 메모를 남기고 Epic 5 error analysis의 입력으로 사용 — detection 오류가 downstream consistency 노이즈의 주요 원인이 될 수 있음(PRD 리스크 2).
- **캐싱 이유**: detection은 학습 중 반복 호출하지 않고 사전 캐싱하여 fusion 학습(Story 2.4) 속도를 확보한다 (NFR1 모듈별 독립 학습 전략).
- **seed**: Ultralytics `seed` 인자 + `deterministic=True` 설정으로 재현성 확보 (NFR3).

### Testing

- unit test: `Detector.predict` 출력 필드/타입, confidence threshold 동작, 검출 0건 이미지에서 빈 리스트 반환.
- 모델 검증: test split mAP@50/P/R 리포트 산출을 완료 조건으로 한다 (NFR4).

---

## Story 2.3: Region Encoder

### Status

Approved

### Story

**As a** ML 엔지니어,
**I want** 검출된 bbox crop에서 CNN/CLIP 기반 region feature를 추출하여 고정 길이 region feature 시퀀스를 구성하고 싶다,
**so that** cross-modal attention fusion(Story 2.4)이 이미지의 객체 단위 표현을 텍스트 token과 결합할 수 있다.

### Acceptance Criteria

1. `src/vision/region_encoder.py`에 `RegionEncoder` 모듈이 구현된다: 입력(원본 이미지 + detections 리스트) → 출력 `region_features [(R+1), 768]`(region crop feature + global image feature 1개, Linear projection으로 768-dim 통일 — architecture의 `V' ∈ R^{(R+1)×768}` 스펙), `region_mask [(R+1)]`, `region_meta`(class_id, bbox, confidence).
2. bbox crop 로직이 구현된다: xyxy 좌표 clamp(이미지 경계), 최소 크기 필터(예: 짧은 변 16px 미만 제외), 선택적 context padding(bbox 확장 비율 config, 기본 10%).
3. backbone은 config로 선택 가능하다: `clip` (CLIP ViT-B/32 image encoder, raw D=512, 기본값) 또는 `resnet` (ResNet-50 penultimate, raw D=2048). 어느 쪽이든 학습 가능한 Linear projection으로 공통 768-dim에 매핑한다. backbone은 frozen이 기본이다.
4. region 수는 confidence 상위 R개(기본 R=16, architecture의 이미지당 최대 16 regions 준수)로 truncate하고, R 미만이면 zero-padding + mask 처리한다. global image feature가 항상 시퀀스에 포함되므로 검출 0건 이미지도 최소 1개(global) region으로 동작한다.
5. region feature에 class embedding(6-class learnable embedding)과 bbox 위치 encoding(normalized [x1,y1,x2,y2,w,h] projection)을 더하는 옵션이 config로 제어된다 (기본 on).
6. 금융 subset 전체에 대해 region feature를 사전 추출·캐싱하는 스크립트가 있으며(`.npz` 또는 `.pt` per split), Story 2.4 학습 시 디스크 캐시에서 로드된다.
7. 처리 통계(이미지당 평균 region 수, 필터링된 bbox 수, global-only 비율)가 리포트된다.

### Tasks / Subtasks

- [ ] crop 유틸 구현 (AC: 2)
  - [ ] `src/vision/crop.py`: clamp, 최소 크기 필터, context padding, 배치 crop
- [ ] RegionEncoder 모듈 구현 (AC: 1, 3, 4, 5)
  - [ ] CLIP/ResNet backbone 로더 (HuggingFace/torchvision, frozen 옵션) + 768-dim Linear projection
  - [ ] class embedding + bbox positional encoding projection layer
  - [ ] global image feature 결합((R+1) 시퀀스), top-R truncation, padding/mask, 검출 0건 처리
- [ ] feature 캐싱 파이프라인 (AC: 6)
  - [ ] `scripts/detection/extract_regions.py`: detections.json + 이미지 → split별 feature 캐시 파일
  - [ ] 캐시 로더 `RegionFeatureDataset` (Story 2.4에서 사용)
- [ ] 통계 리포트 (AC: 7)
  - [ ] 추출 스크립트 실행 시 통계 JSON/CSV 출력
- [ ] Testing (AC: 1, 2, 4)
  - [ ] unit test: crop 경계 케이스(이미지 밖 bbox, 극소 bbox), padding/mask shape, 검출 0건(global-only) 동작, 출력 dtype/shape

### Dev Notes

- **CLIP 우선 이유**: CLIP image embedding은 Epic 4의 visual entity recognition(로고 CLIP 분류)과 backbone을 공유할 수 있어 메모리·일관성 이점 (PRD Technical Assumptions). `openai/clip-vit-base-patch32` 사용.
- **frozen backbone**: 단일 GPU 제약(NFR1) 하에서 backbone은 freeze하고 projection/embedding layer만 fusion 단계에서 학습. 캐싱이 가능해지는 전제이기도 하다 — class/bbox embedding은 학습 대상이므로 캐시에는 raw backbone feature + meta만 저장하고, embedding 결합은 학습 시 on-the-fly로 수행한다.
- **interface 계약**: Story 2.4의 fusion 모델은 `(region_features, region_mask, region_meta)`만 소비한다 — detector/encoder 교체가 가능해야 ablation이 성립 (NFR6).
- **R=16 근거**: architecture Detection 스키마의 "이미지당 최대 R=16" 상한 준수. Fakeddit 이미지의 객체 수 분포를 Story 2.2 캐시로 확인 후 필요 시 하향 조정 가능, config 노출.
- **차원 계약**: 캐시에는 raw backbone feature(512/2048) + meta만 저장하고, 768-dim projection·class/bbox embedding은 학습 대상이므로 학습 시 on-the-fly로 적용 — fusion(Story 2.4)이 소비하는 최종 시퀀스는 `V' [(R+1), 768]`.

### Testing

- pytest unit test 중심 (전처리 로직은 PRD Testing Requirements의 unit test 대상).
- 소규모 smoke test: 샘플 이미지 10장 → 캐시 생성 → 로드 round-trip 검증.

---

## Story 2.4: Cross-modal Attention Fusion

### Status

Approved

### Story

**As a** ML 엔지니어,
**I want** region feature 시퀀스와 BERT token feature를 cross-modal attention으로 결합하는 분류 모델을 학습하고 싶다,
**so that** "+Object Detection" 구성의 성능을 측정하여 late fusion baseline(Story 1.4) 대비 개선 여부를 ablation table에 기록할 수 있다.

### Acceptance Criteria

1. `src/fusion/cross_modal_attention.py`에 fusion 모델이 구현된다:
   - Text encoder: Epic 1에서 확정한 BERT 계열 backbone의 token-level hidden states `T [256, 768]` (architecture Token Encoder 스펙)
   - Region 입력: Story 2.3의 `region_features V' [(R+1), 768]` + mask
   - 양측 공통 차원 `d_model=768`에서 2-stream multi-head cross-attention (text→region 및 region→text 양방향, 기본 8 heads, 2 layers — architecture Fusion Classifier 스펙)
   - attended feature pooling(masked mean 또는 [CLS]) → concat → MLP classifier → fake probability
2. padding region/token은 attention mask로 정확히 제외된다.
3. 학습 스크립트가 config 기반으로 동작한다: lr, epochs, batch size, d_model, heads, layers, seed. text encoder freeze/unfreeze가 config로 제어된다(기본: freeze 후 상위 2개 layer만 unfreeze).
4. 동일 평가 셋(Epic 1의 test split)에서 Accuracy/Precision/Recall/F1/AUROC가 산출된다 (NFR4, FR10).
5. ablation 결과 기록: 기존 결과표(`results/ablation.csv` 또는 동등물)에 "+Object Detection" row가 추가되고, BERT only / ResNet only / BERT+Image(late fusion)와 동일 조건(동일 split, seed, 평가 코드) 비교가 명시된다.
6. attention weight 추출 API가 구현된다: 샘플별 (token × region) attention map을 반환 — Epic 5 데모/error analysis에서 근거 시각화에 사용.
7. 학습 로그(loss curve, val metric)가 Epic 1에서 선택한 로깅 방식(CSV/TensorBoard/W&B)으로 기록된다.

### Tasks / Subtasks

- [ ] 모델 구현 (AC: 1, 2, 6)
  - [ ] projection layers, bidirectional cross-attention block, masked pooling, classifier head
  - [ ] `return_attention=True` 옵션으로 attention map 반환
- [ ] Dataset/DataLoader 구성 (AC: 1)
  - [ ] `RegionFeatureDataset`(Story 2.3 캐시) + 텍스트 tokenization 결합 collate_fn (dynamic padding + mask)
- [ ] 학습 스크립트 (AC: 3, 7)
  - [ ] `scripts/fusion/train_cross_modal.py`: config 로드, seed 고정, early stopping(val F1), best checkpoint 저장
- [ ] 평가 및 ablation 기록 (AC: 4, 5)
  - [ ] Epic 1 공통 평가 루프 재사용, test metric 산출
  - [ ] ablation 결과표에 "+Object Detection" row 추가, 비교 코멘트 작성
- [ ] Testing (AC: 1, 2, 6)
  - [ ] unit test: forward pass shape, mask 적용(padding 위치 attention weight ≈ 0), attention map shape, 검출 0건(global region 1개만 존재) 샘플 처리

### Dev Notes

- **아키텍처 근거**: word-region pair 수준 cross-modal matching(PRD 핵심 목표)의 기반 구조. architecture Fusion Classifier(2-stream cross-attention, 2 layers, 8 heads, d=768)와 동일 골격이며, 본 스토리는 consistency vector 없이 cross-attention 경로만 학습("+Object Detection" 구성) — Epic 4(Story 4.3)에서 consistency feature가 이 모델에 concat 결합된다. GPU 제약 시 layer/head 축소는 config로 허용하되 최종 결과는 architecture 기본값 기준으로 보고 (NFR1).
- **Text backbone**: architecture는 KLUE-BERT(한국어 셋 기준)를 명시하나 PRD Technical Assumptions에 따라 Fakeddit(영어) 학습 시 Epic 1에서 확정한 DeBERTa 계열을 사용한다. hidden dim 768은 양쪽 모두 동일하므로 인터페이스(`T [256, 768]`)는 불변 — backbone 선택은 config로 제어.
- **학습 전략**: region backbone frozen + 캐시 feature 사용으로 fusion 모델만 학습 → 단일 GPU에서 1 epoch가 빠르게 돌도록 설계. text encoder full fine-tuning은 메모리 초과 시 즉시 freeze로 전환.
- **공정 비교**: ablation 유효성을 위해 Story 1.4와 동일한 train/val/test split, 동일 seed 목록(최소 1개, 가능하면 3 seeds 평균), 동일 평가 스크립트를 사용해야 한다 (FR10, NFR3).
- **성능 기대치**: late fusion 대비 F1 개선이 없더라도 이 스토리는 실패가 아니다 — 결과를 그대로 기록하고 원인 가설(detection 품질, region 수, attention 용량)을 남겨 Epic 5 error analysis의 입력으로 한다.
- **interface**: 이 모델의 attended feature와 attention map은 Epic 4에서 consistency feature와 재결합되므로, pooled feature를 반환하는 `encode()` 메서드를 classifier와 분리해 둔다 (NFR6).

### Testing

- unit test: 모델 forward/mask/attention (위 Tasks 참조).
- 모델 검증: held-out test split에서 5개 지표 산출 + ablation row 기록을 완료 조건으로 한다.
- smoke test: 소규모 subset(예: 500 샘플) 1 epoch overfit 확인으로 학습 루프 정상 동작 검증.

## Change Log

- 2026-08-10 (PO Sarah): Story 2.1 AC3 자체 annotation 300 → 150장 축소(공개 데이터 재활용 우선), AC7 gold set 30장 신설(pseudo-label 앵커링 방지, recall 측정). Story 2.2 AC8 하드 게이트 신설(Story 6.1 최소 30건 착수). Story 2.1·2.2 Status는 Approved 유지 — 범위 축소와 검증 강화이며 구현 착수 전이라 재승인 불필요(PO 재승인 완료).
