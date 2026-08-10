# 인계 문서 (2026-08-10)

다음 작업 세션을 위한 현황 요약. **이 문서를 먼저 읽고 시작할 것.**

---

## 1. 프로젝트 한 줄 요약

**FinFact** — 뉴스 이미지 속 시각적 증거(YOLO로 검출한 인물·로고·차트)와 기사 텍스트의 주장(NER/event로 뽑은 entity)을 **entity 단위로 대조**해 금융 가짜뉴스를 탐지한다. 단순 Fake/Real 확률이 아니라 **어떤 entity가 불일치하는지 근거**를 함께 낸다.

핵심 입증 목표: ablation A1~A6에서 **A5(textual entity) 대비 A6(+entity consistency)의 F1 개선**.

---

## 2. 현재 상태 — 한 문장

**코드는 Epic 1(baseline)까지 완성되어 pytest 183개가 통과하지만, 실데이터가 없어 성능 숫자가 하나도 없다.**

| 구분 | 상태 |
|---|---|
| 기획 문서 | 완료 (project brief / PRD v1.2 / architecture v1.1 / 스토리 25개) |
| Epic 1 구현 | 코드 완료, **Done 아님** (데이터 블로킹) |
| Epic 2~6 | 전부 미착수 |
| 테스트 | 183개 통과, **네트워크 접근 0회 / 전부 합성 데이터** |
| ablation 표 | 3행 전부 `missing` |

최신 커밋: `dfe1b65` (main, 푸시 완료). 옵시디언 미러: `C:\Users\user\Desktop\ob_storage\FinFact\`

---

## 3. 지금 당장 막고 있는 것 (최우선)

**전 라인이 데이터·네트워크에서 막혀 있다.** 다음 세 가지가 풀려야 진행된다:

1. **Fakeddit 메타데이터(tsv) + 이미지 확보** → `docs/DATA.md` 절차 참조
2. **HuggingFace / torchvision 사전학습 가중치 다운로드 가능한 환경**
3. **GPU 환경**

### 데이터 받으면 가장 먼저 할 일 (순서 중요)

1. **라벨 인코딩 눈으로 확인** — Fakeddit 원본 `2_way_label`은 **1=real**이고 내부 규약은 **1=FAKE**라 반전 매핑이다. `src/data/labels.py`의 `FAKEDDIT_2WAY_TO_INTERNAL`에 고정되어 있고 항등이면 실패하는 테스트가 걸려 있지만, **실제 배포본 인코딩은 아직 아무도 못 봤다.** 여기가 틀리면 전 실험이 뒤집힌 채로 "잘 나왔다"는 결론이 나오고 아무도 못 알아챈다.
2. **금융 subset 실규모 측정** — `scripts/filter_financial.py` 실행. 이 숫자 하나가 프로젝트 방향을 바꾼다:
   - train split **5k 이상** → 현행 유지
   - **3k~5k** → 2-stage(전체 pretrain + 금융 fine-tuning)만
   - **3k 미만** → FR11을 "평가 전용"으로 축소, **"금융 특화 학습" 주장 철회**
3. **Epic 1 기준선 3종 학습** → ablation A1~A3 채우기, AUROC ≥ 0.60 판정

---

## 4. 완료된 것 (재확인 불필요)

`src/` 아래 배관은 전부 검증됐다:

- `src/utils/` — config(`_base_` 상속 + `--set` 오버라이드), seed 고정, CSV/TensorBoard 로깅
- `src/training/trainer.py` — 모델 비의존 학습 루프, best checkpoint, early stopping, AMP
- `src/evaluation/` — Accuracy/P/R/F1/AUROC (FAKE=1 기준), ablation 취합
- `src/data/` — 라벨 매핑, 전처리, seed 고정 분할, 금융 키워드 필터, `FakedditDataset`
- `src/fusion/` — `TextEncoder`(768) / `ImageEncoder`(2048) / `LateFusionClassifier`(concat 2816)
- `scripts/` — train, evaluate, predict, collect_ablation, download/preprocess/filter_financial

### 반드시 지켜야 할 계약 (architecture.md의 C1~C5)

```python
model(batch: dict) -> logits [B, 2]   # 라벨은 batch["label"]
```

이 계약만 지키면 Trainer를 한 줄도 안 고치고 모델을 갈아끼울 수 있다. 데이터셋·모델은 `src/data/registry.py`·`src/fusion/registry.py`에 등록하고 config의 `data.name`/`model.name`으로 선택한다. **이게 ablation(NFR6)의 전제이므로 깨뜨리지 말 것.**

---

## 5. 교차검증 회의에서 뒤집힌 결정 (중요 — 되돌리지 말 것)

BMAD 팀(Analyst/PM/PO/Dev) 교차검증에서 초기 계획의 오류 3건이 발견되어 수정됐다. 근거가 문서에 남아 있으니 **다시 원래대로 돌리려 하지 말 것.**

| 항목 | 원래 계획 | 수정 후 | 이유 |
|---|---|---|---|
| **핵심 가설 검증 위치** | 자체 금융 셋(n=300) | **Fakeddit 금융 subset(5k~20k, 3-seed)** | n=300에서 F1 1%p는 샘플 3건. McNemar 검출에 8~10%p 필요 |
| **유형별 지표 표** | mismatch 유형별 P/R 보고 | **산출·보고 금지**, 정성 사례 3~5건으로 대체 | 유형당 n=30 → Wilson 95% CI 폭 27%p, 유형 간 비교 불가 |
| **annotation 품질** | intra-annotator κ ≥ 0.8 | κ + **제3자 스팟체크 30건 필수** | κ는 일관성이지 정확성이 아님 (같은 실수 반복 시 κ 0.9인데 라벨은 틀림) |

**자체 금융 셋(FinFact-Eval, 300건)의 역할 재정의**: (i) FR16 matching 판정 정확도의 **유일한** 정답 소스(Fakeddit엔 entity pair annotation이 없음), (ii) 도메인 이전 정성 확인, (iii) 데모 예시. 이 역할에서는 300건이 적정하다.

---

## 6. 진짜 병목은 GPU가 아니라 사람 시간

라벨링 작업 두 건이 **같은 사람의 3~7주차**에 몰린다:

- **Story 2.1** — bbox 라벨링 **150장** (300에서 축소) + **gold set 30장**
- **Epic 6** — 금융 뉴스 수집·조작·annotation **300건**

### 이미 걸어둔 방어 장치

- **Story 1.5** (Epic 1 내 앵커) — 출처 정책 확정 + 시드 20건 + **건당 소요 시간 측정** → 300건이 일정에 맞는지 실측으로 판단
- **Story 2.2 AC8** — 완료하려면 Story 6.1 최소 30건 착수 필요 (하드 게이트)
- **Story 5.1 AC7** — 착수하려면 Story 6.1~6.5 Done 필요 (하드 게이트)
- **4주차 bbox / 5주차 수집** 번갈이 배치
- **Fallback ladder** 300 → 180 → 100 → 40, 각 단계에서 포기하는 주장 명시, **6주차 말 실측으로 자동 판정**(사후 재량 없음)

### gold set 30장은 줄이지 말 것

pseudo-label 초안 위에 수정만 하면 **"아예 없는 박스"는 구조적으로 안 보인다**(앵커링). 이 편향은 라벨 품질 지표에 안 잡히고 mAP만 낙관적으로 만든다. gold set이 유일한 방어선이다.

---

## 7. 오너 결정 대기 중인 안건

1. **네트워크·GPU 환경** 언제 열리나
2. **Epic 6 300건** 직접 감당 가능한가 (안 되면 fallback ladder 적용)
3. **제3자 스팟체크 30건** 해줄 사람 확보 가능한가 (안 되면 seed 자기 점검 + 한계 명시)
4. **checkpoint 보관 위치** — git엔 metric+config+hash만, checkpoint는 로컬 보존 의무 (재현성 실증 전까지 삭제 금지)

---

## 8. 미해결 / 남은 정리 작업

- **Story 1.4 Status = `Changes Requested`** — AC5(git hash·split 해시·환경 기록)와 AC7(실데이터 2회 학습 실증)이 확장되어 현 구현이 미충족. 데이터 확보 후 처리.
- `src/evaluation/ablation.py`의 `AblationRow`에 **`ablation_id` 필드 추가** 필요 (A1/A2/A3 명시) — drift 리뷰 지적사항, 미반영
- `requirements.txt`의 `# Demo (Epic 6)` 주석 → `Epic 5`로 정정 필요
- `requirements.txt`의 `wandb` — 코드 경로가 없으므로 optional로 내리거나 실제 구현 필요
- Epic 6 스토리 6개는 **Draft** 상태 (PO 승인 대기)
- **A6가 개선을 못 보일 경우** → Story 5.2 AC6에 "오류 전파 정량 분석을 기여로" 설계가 이미 박혀 있고, **판정은 5.1 결과 나온 날 확정**(사후 서사 조정 금지, HARKing 차단)

---

## 9. 문서 지도

| 파일 | 내용 |
|---|---|
| `README.md` | 프로젝트 소개, 파이프라인, 설치·실행 |
| `docs/DATA.md` | Fakeddit 준비 절차, 라벨 규약, manifest 스키마 |
| `docs/DEV_PLAN.md` | Phase 1~4, Phase↔Epic 매핑, 10주 스케줄, ablation A1~A6 |
| `docs/bmad/prd.md` | **v1.2** — FR1~17, NFR1~9, Epic 1~6, Revision Notes에 결정 근거 |
| `docs/bmad/architecture.md` | **v1.1** — Tech Stack, Source Tree(✅구현/⬜계획), **Contracts C1~C5** |
| `docs/ARCHITECTURE.md` | 상세 기술 설계 (구현 정본은 bmad/architecture.md) |
| `docs/bmad/stories/epic-1~6*.md` | 스토리 25개 |

**주의**: 구현하며 설계에서 벗어나면 `docs/bmad/architecture.md`를 함께 갱신할 것 (Coding Standards 원칙). 한 번 안 지켜서 drift 8건이 쌓였고 정리하는 데 별도 작업이 필요했다.

---

## 10. 다음 세션 시작 문장 제안

> `docs/HANDOFF.md`를 읽고 현황 파악해줘. 데이터는 [확보함 / 아직 없음]이야.

데이터가 있으면 → §3의 3단계(라벨 확인 → subset 측정 → 기준선 학습)부터.
데이터가 없으면 → Story 1.5(수집 게이트) 또는 Epic 2 코드 작업 중 사람 손이 필요 없는 부분부터.
