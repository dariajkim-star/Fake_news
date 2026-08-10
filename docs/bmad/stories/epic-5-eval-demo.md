# Epic 5: 평가 · Ablation · 데모 — User Stories

> 작성자: Bob (BMAD Scrum Master) · 작성일: 2026-08-10
> 참조: `docs/bmad/prd.md` (Epic 5, FR8~FR10, NFR3~NFR4), `docs/DEV_PLAN.md` (§4 평가 지표, §6 Ablation 실험 설계, §8 최종 산출물)

---

## Story 5.1: Ablation 실험 매트릭스 실행 및 결과 리포트

### Status
Approved

### Story
**As a** 연구자(과제 수행자),
**I want** A1~A6 전 구성(BERT only / ResNet only / BERT+ResNet late fusion / +Object Detection / +Textual entity features / +Entity consistency)을 동일 split·seed 조건에서 일괄 학습·평가하는 ablation 실험 매트릭스 실행 스크립트,
**so that** entity consistency feature가 최종 분류 F1/AUROC를 추가 개선한다는 핵심 가설을 재현 가능한 결과 표로 입증할 수 있다.

### Acceptance Criteria
1. `scripts/run_ablation.py`(또는 동등한 entry point)가 config 디렉토리(`configs/ablation/a1_bert_only.yaml` ~ `a6_full.yaml`)를 순회하며 A1~A6 전 구성을 학습+평가하거나, 기존 checkpoint가 있으면 `--eval-only`로 평가만 수행할 수 있다.
2. 모든 실험은 동일한 train/val/test split과 고정 seed를 사용하며, `--seeds 42,43,44` 옵션으로 3-seed 실행 시 지표의 mean ± std를 산출한다.
3. 각 실험 결과는 `outputs/ablation/{config_name}/{seed}/metrics.json`에 Accuracy, Precision, Recall, F1, AUROC가 저장되고, 실행에 사용된 config 사본과 git commit hash가 함께 기록된다.
4. `scripts/report_ablation.py`가 전 결과를 집계하여 (a) Markdown 표(`outputs/ablation/ablation_table.md`), (b) CSV(`ablation_table.csv`), (c) A3→A4→A5→A6 F1 개선 곡선 plot(`ablation_curve.png`)을 생성한다.
5. Ablation 결과 표에서 A6(+Entity consistency)의 F1이 A5 대비 기록되고, 개선 여부(+Δ%p)가 표에 명시된다 (개선 미달 시에도 수치가 정직하게 기록되어 Story 5.2의 분석 입력이 된다).
6. Fakeddit 금융 subset test와 자체 금융 셋(held-out) 두 평가 셋 모두에 대해 A1~A6 결과가 산출된다.

### Tasks / Subtasks
- [ ] Task 1: Ablation config 세트 정비 (AC: 1, 2)
  - [ ] `configs/ablation/`에 a1~a6 yaml 작성 — 각 config는 model 구성 flag(`use_image`, `use_text`, `use_regions`, `use_entity_features`, `use_consistency`)만 다르고 data/split/seed/하이퍼파라미터는 공통 base config 상속
  - [ ] 공통 base config에 split 파일 경로·seed·batch size·epoch 고정
- [ ] Task 2: `scripts/run_ablation.py` 구현 (AC: 1, 2, 3)
  - [ ] config glob → 순차 실행 루프, `--configs`, `--seeds`, `--eval-only`, `--resume` CLI 인자 (argparse)
  - [ ] 실험별 output 디렉토리 생성, config 사본·git hash·실행 시각 저장
  - [ ] Epic 1~4에서 구축한 공통 train/eval 루프 재사용 (모델 factory가 config flag로 구성 분기)
  - [ ] 각 실행 종료 시 `metrics.json` 저장 (sklearn 기반 Accuracy/P/R/F1/AUROC)
  - [ ] wandb run 로깅 연동 (run name = `{config_name}-s{seed}`, config·지표 기록; 오프라인 환경 대비 `--no-wandb` flag 시 CSV fallback — architecture "Experiment Tracking" 준수)
- [ ] Task 3: 두 평가 셋 평가 지원 (AC: 6)
  - [ ] `--eval-sets fakeddit_fin,custom_fin` 옵션으로 test 셋별 metrics 분리 저장 (`metrics_fakeddit_fin.json`, `metrics_custom_fin.json`)
- [ ] Task 4: `scripts/report_ablation.py` 구현 (AC: 4, 5)
  - [ ] `outputs/ablation/` 스캔 → pandas DataFrame 집계 (seed별 mean ± std)
  - [ ] Markdown/CSV 표 생성 (행: A1~A6, 열: Acc/P/R/F1/AUROC, Δ F1 vs 이전 행)
  - [ ] matplotlib으로 A3→A4→A5→A6 F1 개선 곡선 plot 저장
- [ ] Task 5: 전체 매트릭스 실행 및 결과 커밋 (AC: 2, 5, 6)
  - [ ] 3-seed × 6 config 실행 (GPU 시간 부족 시 1-seed 우선 완주 후 seed 추가)
  - [ ] 최종 `ablation_table.md`를 `docs/results/`에 복사·커밋

### Dev Notes
- **Ablation 매트릭스 정의는 DEV_PLAN §6을 그대로 따른다**: A1 BERT only, A2 ResNet only, A3 late fusion, A4 +YOLO regions+Cross Attention, A5 +Textual entity features, A6 +Entity/Event consistency score. PRD FR10의 5구성 표기(BERT only/ResNet only/BERT+Image/+OD/+Entity consistency)는 A1/A2/A3/A4/A6에 대응 — 리포트 표는 A1~A6 6행으로 작성.
- **모델 factory 분기**: Epic 1~4의 fusion 모델이 config flag 기반으로 하위 구성을 끌 수 있어야 한다(NFR6 모듈화). consistency feature off 시 해당 입력 차원을 zero-fill이 아니라 concat 자체에서 제외할 것 (파라미터 수 차이는 리포트에 명기).
- **재현성(NFR3)**: `torch.manual_seed`, `numpy`, `random`, `cudnn.deterministic=True` 일괄 설정 유틸 재사용. split은 Epic 1의 고정 split 파일(예: `data/splits/*.json`)을 참조하고 재생성 금지.
- **AUROC**: `sklearn.metrics.roc_auc_score`에 fake class probability(softmax/sigmoid 출력) 사용. threshold 기반 지표(P/R/F1)는 0.5 고정, val 셋 기반 threshold 튜닝은 하지 않는다(구성 간 공정 비교).
- **컴퓨팅 제약(NFR1)**: A4~A6는 detector/NER 출력을 사전 계산 캐시(예: `data/cache/regions/`, `data/cache/entities/`)에서 로드 — ablation 재실행 시 upstream 모듈 재추론 방지.
- **실험 추적(architecture)**: wandb를 1차 로깅 채널로 사용(run별 config/metric, ablation 비교 표). `metrics.json`은 wandb와 무관하게 항상 로컬 저장(재현 검증·report 스크립트 입력의 단일 source of truth). 오프라인 대안은 CSV logging.
- **출력 경로**: architecture Source Tree의 `experiments/`가 실험 결과 루트다 — 본 스토리의 `outputs/ablation/`은 `experiments/ablation/`으로 매핑해 구현해도 무방하며, 최종 표는 어느 쪽이든 `docs/results/`로 복사한다(report 스크립트의 스캔 루트를 config로 주입).
- **성공 기준(DEV_PLAN Phase 4)**: A6가 A5 대비 F1/AUROC +1%p 이상이면 가설 입증. 미달 시에도 결과를 그대로 기록하고 Story 5.2 error analysis를 대안 기여로 전환(PRD 리스크 3).

### Testing
- Unit: config 상속/flag 파싱 테스트 — a1~a6 config 로드 시 기대 flag 조합이 나오는지 (`tests/test_ablation_configs.py`).
- Unit: `report_ablation.py` 집계 로직 — mock metrics.json 3-seed 입력 → mean ± std, Δ F1 계산 검증.
- Smoke: `run_ablation.py --configs a1 --seeds 42 --max-steps 10` 형태의 축소 실행이 end-to-end로 metrics.json을 생성하는지 CI/로컬에서 확인.
- 결과 무결성: 동일 config·seed 2회 실행 시 지표가 일치(재현성)하는지 1회 검증.

---

## Story 5.2: Error Analysis — MISMATCH 오탐/미탐 및 오류 전파 분석

### Status
Approved

### Story
**As a** 연구자,
**I want** 최종 모델(A6)의 오분류 사례와 cross-modal matching의 MISMATCH 오탐/미탐 사례를 컴포넌트별(detection → visual entity → NER → matching)로 추적·분류하는 error analysis 스크립트와 리포트,
**so that** entity consistency의 개선(또는 미개선) 원인을 규명하고, 개선 실패 시에도 오류 전파 분석 자체를 과제 기여로 제시할 수 있다.

### Acceptance Criteria
1. `scripts/error_analysis.py`가 test 셋에 대해 A6 모델의 예측·중간 산출물(detection bbox, visual entity, textual entity, match 판정, consistency score)을 샘플 단위로 dump한 `outputs/error_analysis/predictions.jsonl`을 생성한다.
2. 오분류 샘플(FP/FN)이 자동 분류된다 — 최소 카테고리: (a) detection miss, (b) visual entity 오인식, (c) NER 누락/오류, (d) matching 오판(MATCH↔MISMATCH), (e) UNKNOWN 과다로 consistency 무효, (f) 모델 자체 오류(중간 산출물 정상).
3. 자체 금융 셋의 mismatch 유형 annotation(PERSON/LOGO(ORG)/NUMBER/EVENT) 기준으로 유형별 MISMATCH 탐지 정확도(Precision/Recall) 표가 산출된다.
4. A5 vs A6 예측이 갈린 샘플(A6가 새로 맞춘 것 / 새로 틀린 것) 목록이 추출되어 consistency feature의 기여·손해 사례를 각 5건 이상 확보한다.
5. 분석 결과가 `docs/results/error_analysis.md`로 정리된다 — 카테고리별 비율 표, 유형별 정확도 표, 대표 사례 최소 6건(이미지 썸네일 + 텍스트 + 판정 근거 포함).

### Tasks / Subtasks
- [ ] Task 1: 예측·중간 산출물 dump 구현 (AC: 1)
  - [ ] 추론 파이프라인(Story 4.4 모듈)에 `--dump-intermediate` 모드 추가 — 샘플 id, label, pred, prob, bbox 리스트, visual/textual entity 리스트, match 결과, consistency score를 jsonl 1행으로 기록
- [ ] Task 2: 오류 카테고리 자동 분류기 구현 (AC: 2)
  - [ ] rule 기반 분류: detection 출력 없음→(a), visual entity confidence<threshold 다수→(b)/(e), textual entity 0건→(c), gt mismatch 유형 존재하나 match=MATCH→(d) 등
  - [ ] 카테고리별 카운트/비율 집계 CSV 출력
- [ ] Task 3: mismatch 유형별 정확도 산출 (AC: 3)
  - [ ] 자체 금융 셋 annotation 로드 → 유형별 MISMATCH 판정 P/R 계산 (DEV_PLAN §4 "Cross-modal matching 판정 정확도")
- [ ] Task 4: A5 vs A6 diff 분석 (AC: 4)
  - [ ] 두 구성의 predictions.jsonl join → flip 샘플 추출(`a6_fixed.jsonl`, `a6_broke.jsonl`), consistency score 분포 비교
- [ ] Task 5: 리포트 작성 (AC: 5)
  - [ ] 대표 사례 선정(유형별 최소 1건, 기여/손해 사례 포함), bbox overlay 이미지 저장(`outputs/error_analysis/cases/`)
  - [ ] `docs/results/error_analysis.md` 작성 — 발표 자료(Story 5.4)에서 재사용 가능한 형태

### Dev Notes
- Story 5.1의 산출물(predictions, checkpoint)에 의존 — 5.1 완료 후 착수. A5/A6 diff는 5.1의 동일 seed checkpoint 사용.
- **오류 전파 프레임**: 파이프라인 순서(detection → visual entity → NER → matching → fusion)를 따라 "가장 upstream의 실패 지점"으로 귀속시키는 first-failure attribution 규칙 사용. 다중 원인 샘플은 primary/secondary 태그 허용.
- UNKNOWN 비율이 높은 샘플군(리스크: Visual Entity Recognition 정확도)은 별도 집계 — PRD FR12의 UNKNOWN 처리 정책이 오류 전파를 실제로 차단했는지 정량 확인.
- 이미지 썸네일 포함 리포트는 markdown 상대 경로로 `outputs/error_analysis/cases/` 이미지를 참조 (repo 용량 고려, 640px 이하 리사이즈).
- 개선 미달 시나리오: A6 F1이 A5 대비 개선되지 않았다면 (e)/(b) 카테고리 비율과 UNKNOWN 통계로 "consistency 신호가 노이즈에 희석된" 정도를 정량 제시 — 발표의 대안 기여 포인트.

### Testing
- Unit: 오류 카테고리 분류기 — 합성 jsonl 입력(각 카테고리별 최소 1건)으로 기대 카테고리 산출 검증 (`tests/test_error_analysis.py`).
- Unit: mismatch 유형별 P/R 계산 — 소형 annotation fixture로 수치 검증.
- Smoke: test 셋 10샘플 축소 실행으로 predictions.jsonl → 리포트까지 전 단계 통과 확인.

---

## Story 5.3: Gradio 데모 앱 — Fake Score + Mismatch Evidence 시각화

### Status
Approved

### Story
**As a** 발표 청중/과제 평가자,
**I want** 이미지 업로드 + 뉴스 텍스트 입력만으로 fake score와 mismatch evidence(이미지 bbox 하이라이트 + 텍스트 entity 하이라이트 + consistency 점수 표)를 한 화면에서 확인할 수 있는 Gradio 데모,
**so that** 어떤 entity가 불일치했는지 근거를 직관적으로 이해할 수 있다.

### Acceptance Criteria
1. `demo/app.py` 실행(`python demo/app.py`) 시 로컬 Gradio 앱이 뜨고, 이미지 업로드 + 텍스트 입력 + 분석 버튼의 단일 화면 흐름이 동작한다.
2. 분석 결과로 (a) fake probability 게이지/점수(`gr.Label` 또는 gauge), (b) 검출 bbox가 overlay된 이미지 — MATCH 녹색/MISMATCH 적색/UNKNOWN 회색 + 라벨 텍스트 병기, (c) 텍스트 entity 하이라이트(`gr.HighlightedText`, 동일 색상 체계), (d) Entity consistency/Event consistency/Image-Text similarity 세부 점수 표가 표시된다.
3. Entity pair 상세 테이블이 표시된다 — 각 행: visual entity(클래스, 인식 결과), textual entity, 판정(MATCH/MISMATCH/UNKNOWN), similarity score.
4. 최소 4개의 pre-loaded 예시 샘플(`gr.Examples`)이 탑재된다 — Real 1건 이상, PERSON/LOGO/NUMBER mismatch 각 1건 이상(자체 금융 셋에서 선정).
5. 단일 샘플 end-to-end 추론이 GPU 기준 수 초 이내에 완료된다 (NFR2) — 모델은 앱 시작 시 1회 로드.
6. 색상만으로 판정을 구분하지 않는다 — bbox와 entity 하이라이트에 MATCH/MISMATCH/UNKNOWN 라벨 텍스트가 병기된다 (색각 접근성).

### Tasks / Subtasks
- [ ] Task 1: 추론 파이프라인 어댑터 (AC: 2, 3, 5)
  - [ ] Story 4.4의 end-to-end 파이프라인을 `demo/inference_service.py`로 wrapping — 앱 시작 시 모델·detector·NER 1회 로드(lazy singleton), `predict(image, text) -> DemoResult` dataclass 반환 (fake_prob, bboxes+판정, text entity spans+판정, pair table, sub-scores)
- [ ] Task 2: 결과 시각화 렌더러 (AC: 2, 6)
  - [ ] bbox overlay: PIL/OpenCV로 판정별 색상(RGB: 녹 #2e7d32 / 적 #c62828 / 회 #757575) + `"PERSON: Jensen Huang [MISMATCH]"` 형식 라벨 텍스트 draw
  - [ ] 텍스트 하이라이트: entity span → `gr.HighlightedText` (category = "MATCH"/"MISMATCH"/"UNKNOWN", color_map 지정)
- [ ] Task 3: Gradio UI 조립 (AC: 1, 2, 3)
  - [ ] `gr.Blocks` 레이아웃 — 좌: 입력(Image, Textbox, 분석 Button), 우: 결과(Label/gauge, annotated Image, HighlightedText, `gr.Dataframe` 2개: sub-scores 표 + entity pair 표)
  - [ ] 에러 핸들링: detection 0건/entity 0건 시 "evidence 없음, global similarity만 사용" 안내 메시지 표시
- [ ] Task 4: 예시 샘플 탑재 (AC: 4)
  - [ ] `demo/examples/` 디렉토리에 이미지+텍스트 4건 이상 배치, `gr.Examples`로 연결
  - [ ] 각 예시의 기대 출력(정상 동작) 수동 검증 및 `demo/examples/README.md`에 기대 결과 기록
- [ ] Task 5: 성능·안정성 마무리 (AC: 5)
  - [ ] 추론 latency 측정 로그(초 단위) 추가, GPU/CPU fallback 동작 확인
  - [ ] `demo/README.md`에 실행 방법·요구 checkpoint 경로 문서화

### Dev Notes
- **Gradio 채택** (PRD는 Gradio/Streamlit 중 택1 — 본 스토리는 Gradio로 확정): `gr.Blocks` + `gr.HighlightedText`가 entity 하이라이트 요구에 가장 부합. 버전은 gradio 4.x 고정(requirements에 pin).
- **DemoResult 계약**: Story 4.4의 구조화 출력(FR8)을 그대로 소비 — 데모에서 추론 로직 재구현 금지, 파이프라인 모듈 import만 허용 (NFR6).
- **checkpoint 경로는 config/env로 주입** (`FINFACT_CKPT_DIR` 또는 `demo/config.yaml`) — 하드코딩 금지, 발표 장비 이동 대비.
- **launch 옵션**: `demo.launch(share=False, server_name="127.0.0.1")` 기본. 발표 시 `--share` CLI flag 옵션 제공.
- 게이지 표현은 `gr.Label(num_top_classes=2)`(Fake/Real 확률)로 단순화 가능 — custom HTML gauge는 시간 남을 때만.
- 예시 샘플은 Story 5.2에서 발굴한 대표 사례와 겹치게 선정하면 발표 스토리 라인이 일관됨.
- 대기 시간 UX: 분석 버튼 클릭 시 Gradio 기본 loading 표시 사용, 첫 로드 지연은 앱 시작 시 warm-up 추론 1회로 흡수.

### Testing
- Unit: 렌더러 — 합성 DemoResult 입력으로 bbox overlay 이미지 생성(색상/라벨 텍스트 포함) 및 HighlightedText tuple 리스트 형식 검증 (`tests/test_demo_render.py`).
- Unit: inference_service — mock 파이프라인 주입 시 DemoResult 필드 완전성 검증.
- 수동 시연 테스트(PRD Testing Requirements): 예시 4건 + 즉석 입력 2건(Real/Fake 각 1)으로 전 UI 요소 동작 체크리스트 수행, latency 수 초 이내 확인.
- Edge: 텍스트만/이미지 손상/entity 0건 입력 시 앱이 crash 없이 안내 메시지를 표시하는지 확인.

---

## Story 5.4: 발표 자료 및 재현 문서

### Status
Approved

### Story
**As a** 과제 수행자,
**I want** ablation 결과·error analysis·데모를 아우르는 발표 슬라이드와, 제3자가 실험을 재현할 수 있는 README/재현 가이드,
**so that** 과제 deliverable(발표 + 재현 가능한 코드)을 완성하고 평가자가 결과를 검증할 수 있다.

### Acceptance Criteria
1. 발표 슬라이드(`docs/presentation/` — pptx 또는 pdf)가 다음 구성으로 작성된다: 문제 정의(금융 도메인 특수성) → 아키텍처(detection→entity→matching→fusion) → ablation 개선 곡선(A3→A4→A6) → error analysis 핵심 사례 → 데모 시연(라이브 또는 스크린샷 backup) → 한계·향후 과제.
2. 슬라이드에 Story 5.1의 ablation table과 F1 개선 곡선 plot, Story 5.2의 대표 사례(mismatch evidence 시각화 이미지)가 포함된다.
3. 루트 `README.md`가 갱신된다 — 프로젝트 개요, 아키텍처 다이어그램, 환경 설정(`requirements.txt`/conda env, GPU 요구사항), 데이터 준비(Fakeddit 다운로드 + 금융 subset 필터링 스크립트 실행법), 학습/평가/ablation/데모 실행 커맨드.
4. 재현 가이드로 clean 환경에서 (a) 데이터 준비 스크립트, (b) `run_ablation.py --eval-only`(공개 checkpoint 기준), (c) `demo/app.py` 실행이 문서 커맨드만으로 성공함을 1회 검증한다.
5. 데모 시연 실패 대비 backup이 준비된다 — 예시 4건의 데모 스크린샷(또는 화면 녹화)과 시연 시나리오 스크립트(입력 순서, 예상 결과, 소요 시간)가 `docs/presentation/demo_backup/`에 저장된다.

### Tasks / Subtasks
- [ ] Task 1: 결과 자산 정리 (AC: 2)
  - [ ] `docs/results/`에 ablation_table.md, ablation_curve.png, error_analysis.md, 대표 사례 이미지 취합 (Story 5.1/5.2 산출물)
- [ ] Task 2: 발표 슬라이드 작성 (AC: 1, 2)
  - [ ] 스토리라인 초안: 문제(single CLIP similarity의 한계) → 접근(object-level entity consistency) → 결과(ablation 곡선) → 근거 시각화(데모) — 10~15장 내외
  - [ ] 아키텍처 다이어그램 1장 제작 (모듈 파이프라인 + ablation 구성 A1~A6 대응 표시)
  - [ ] 개선 미달 시 대안 슬라이드: error analysis 기반 원인 분석·오류 전파 정량화 강조
- [ ] Task 3: README/재현 가이드 작성 (AC: 3)
  - [ ] `requirements.txt` 버전 pin 최종화, seed·split 고정 재현 절차 명시 (NFR3)
  - [ ] 실행 커맨드 블록: 데이터 준비 → (선택) 학습 → ablation 평가 → 리포트 생성 → 데모 실행
  - [ ] checkpoint 배포 방식 결정(release asset/드라이브 링크) 및 다운로드 절차 문서화
- [ ] Task 4: 재현 검증 (AC: 4)
  - [ ] clean venv(또는 새 Colab 세션)에서 README 커맨드만으로 (a)(b)(c) 수행, 막히는 지점 문서 수정
- [ ] Task 5: 데모 backup 준비 (AC: 5)
  - [ ] 예시 4건 시연 스크린샷 캡처(또는 1~2분 화면 녹화), 시연 시나리오 스크립트 작성
  - [ ] 발표 리허설 1회 — 시연 포함 발표 시간 측정, 시나리오 조정

### Dev Notes
- Story 5.1~5.3 완료에 의존하는 Epic 5 마무리 스토리 — 결과 수치·이미지는 반드시 실제 산출물에서 가져온다(placeholder 금지).
- **발표 핵심 메시지** (DEV_PLAN §6): "A3→A4→A6 F1 단계적 개선 곡선" 1장이 승부처. 개선 미달 시 Story 5.2의 first-failure attribution 표를 대체 핵심으로 승격.
- 아키텍처 다이어그램은 PRD Service Architecture의 모듈 순서(`detection → visual_entity → text_entity → matching → fusion → demo`)를 따르고, 각 모듈 옆에 해당 ablation 구성(A2/A4/A5/A6)을 배지로 표시하면 실험 설계 설명이 한 장으로 끝난다.
- 재현 가이드는 "평가자 = GPU 1장 + 인터넷"을 가정 — 전체 재학습이 아니라 checkpoint 기반 `--eval-only` 재현을 1차 경로로 제시하고, full 재학습 커맨드는 부록으로.
- Fakeddit 원본 이미지는 재배포 불가 — README에 다운로드 스크립트 사용을 안내하고, 자체 금융 셋만 repo/release에 포함(라이선스 확인).
- 발표 시연은 로컬 Gradio 실행이 1안, `--share` 링크가 2안, backup 스크린샷/녹화가 3안 — 3단 fallback을 시나리오 스크립트에 명시.

### Testing
- 재현 검증(AC 4) 자체가 이 스토리의 핵심 테스트 — clean 환경 체크리스트(환경 생성 → 설치 → 데이터 → 평가 → 데모)를 문서화하고 결과(성공/수정 사항)를 기록.
- 문서 링크·이미지 경로 무결성 확인 (README와 슬라이드 내 상대 경로 깨짐 없음, markdown lint 수준 점검).
- 데모 backup 자산이 실제 최신 checkpoint 출력과 일치하는지 확인 (스크린샷 재캡처 기준: 최종 모델 확정 이후).
