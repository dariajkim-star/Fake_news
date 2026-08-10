# Epic 6: FinFact-Eval — 금융 도메인 특화 held-out 평가셋 — User Stories

> 작성자: John (BMAD PM) · 작성일: 2026-08-10
> 참조: `docs/bmad/prd.md` (Epic 6, FR13~FR17, NFR5·NFR8·NFR9, Revision Notes v1.1), `docs/DEV_PLAN.md` (§2.3 자체 금융 셋, §4 평가 지표, §5 3~7주차), `README.md` (§3, §5)
> 소비처: Story 5.1 (AC 6 `custom_fin` 평가 셋), Story 5.2 (AC 3 mismatch 유형별 정확도), Story 5.3 (AC 4 예시 샘플)

**실행 순서 주의**: Epic 번호는 6이지만 실행은 마지막이 아니다. DEV_PLAN §5 기준 **3주차 수집 시작 → 7주차 셋 완성**으로 Epic 2~4와 병행하며, **Epic 5 착수 전에 Story 6.1~6.5가 완료**되어야 한다. Story 6.6은 Epic 4의 matching 모듈(Story 4.2)과 Visual Entity Recognition(Story 4.1) 이후에 실행 가능하다.

**규모 (커밋 목표)**: 총 300건 = Real 150 + Fake 150. Fake는 mismatch 유형 5종 × 25~35건 — **유형별 지표를 내기 위한 배분이 아니라 유형 쏠림을 막기 위한 커버리지 배분이다.** OCR 검증 subset 50~80 crop. 여유 시 400건까지 확장(stretch).

**이 셋의 역할 (FR13 — 오해 방지를 위해 스토리 착수 전 반드시 읽을 것)**
1. **FR16 matching 판정 정확도의 유일한 정답 소스.** Fakeddit에는 entity pair annotation이 없어 이 셋 없이는 FR16이 측정 불가다. 1차 존재 이유가 이것이다.
2. **도메인 이전의 정성 확인** — 보조 표, 유의성 주장 없음.
3. **데모·발표 예시 공급.**

**이 셋으로 하지 않는 것**: 핵심 가설(A5→A6) 검정. 그것은 Fakeddit 금융 subset(5k~20k, 3-seed)에서 한다(FR10). n=300에서 F1 1%p는 샘플 3건이고 McNemar 유의성에는 8~10%p 점프가 필요하다.

**보고 수준 상한 (FR16)**: 결론적 지표는 **전체 단위 1세트**(3-way accuracy + MISMATCH P/R/F1, pair 총계)까지다. **mismatch 유형별 지표 표는 만들지 않는다** — 유형당 n=30에서 recall 0.80의 Wilson 95% CI는 [0.63, 0.90](폭 27%p)이라 유형 간 차이가 통계적으로 판별되지 않는다. 유형별 특성은 **정성 사례 3~5건**으로 보고한다.

**Fallback ladder (6주차 말 실측 건수로 자동 판정, 사후 재량 없음)**

| 규모 | 성격 | 보고 가능 | 포기하는 주장 |
|---|---|---|---|
| 300 (150/150) | held-out 정량 평가 | 전체 MISMATCH P/R/F1(CI ±6~8%p), 3-way accuracy, 정성 사례 | (유형별 정량 비교는 애초에 비대상) |
| 180 (90/90) | 정량, 정밀도 하락 | 전체 지표 + CI ±9~11%p 병기 필수 | 소수점 단위 비교 |
| 100 (50/50) | 탐색적 평가 | matching 정확도 총계 1개 + 정성 사례 | 금융 도메인 분류 성능 표(역할 ii) 철회 |
| 40 | 정성 사례집 | 데모 예시, 대표 사례 서술 | FR16 정량 지표 전부 철회, 한계로 명시 |

어느 단계로 내려가든 **역할 (i)이 최우선**이다 — 건수를 줄이더라도 entity pair annotation 품질(6.3~6.4)은 축소하지 않는다.

**사람 시간 경합**: Story 2.1의 수작업 bbox와 4주차에 충돌한다. 합의된 조정은 bbox 300→150 축소 + **4주차 bbox / 5주차 수집 번갈이 배치**다(PRD Revision Notes v1.2 §5). 6.1 수집 페이스 계획은 이 배치를 전제로 짤 것.

---

## Story 6.1: 수집 소스·라이선스 정책 확정 및 Real 샘플 수집

### Status
Draft

### Story
**As a** 과제 수행자,
**I want** 자동 크롤링 없이 재사용이 허용된 공개 출처만으로 금융 뉴스(텍스트 + 대표 이미지) Real 샘플 150건을 수집하고 출처·라이선스를 샘플 단위로 기록하는 절차와 산출물,
**so that** 금융 도메인 held-out 평가셋의 합법적이고 재배포 가능한 기반을 확보할 수 있다.

### Acceptance Criteria
1. `docs/data/finfact_eval_sources.md`에 허용 출처 목록과 각 출처의 라이선스/이용 조건, 그리고 **금지 사항(자동 크롤링, 스크래핑 스크립트, robots.txt 무시, 로그인 벽 우회)** 이 명문화된다 (NFR5).
2. Real 샘플 150건이 수집되어 `data/finfact_eval/raw/`에 배치되고, 샘플별로 `id`, `source_url`, `publisher`, `license`, `collected_at`, `image_sha256`, `title`, `body_excerpt`, `image_path`가 `sources.jsonl`에 기록된다 (NFR9).
3. 각 샘플은 금융 도메인 요건을 만족한다 — 기업/인물/금액/날짜 중 **최소 2종의 entity를 포함**하고 기사 내용과 관련된 대표 이미지(인물·로고·차트·문서 중 하나 이상 포함)를 가진다.
4. 이미지 유형 분포가 기록되고 편중되지 않는다 — PERSON 포함, LOGO 포함, CHART/DOCUMENT 포함 샘플이 각각 전체의 15% 이상.
5. 라이선스 확인 불가 또는 재배포 금지 이미지는 셋에서 제외되며, 제외 사유가 `sources_rejected.jsonl`에 남는다.
6. 수집은 수동 선별 기록으로 남는다 — 자동 수집 스크립트를 작성하지 않았음이 리포지토리 상태로 확인 가능해야 한다(수집 보조 도구는 URL 목록으로부터의 단건 다운로드 유틸까지만 허용).

### Tasks / Subtasks
- [ ] Task 1: 출처·라이선스 정책 문서화 (AC: 1, 5)
  - [ ] 재사용 허용 후보 조사 — 공개 라이선스 뉴스 아카이브/데이터셋, 기관 보도자료, 공시(전자공시 등), CC 라이선스 이미지 아카이브
  - [ ] 출처별 허용 범위(본문 전문/발췌만/이미지 재배포 가능 여부)를 표로 정리, 애매하면 "발췌 + URL만" 보수적 기준 채택
- [ ] Task 2: 수집 워크시트/스키마 준비 (AC: 2)
  - [ ] `sources.jsonl` 스키마 확정 및 검증 스크립트(`scripts/validate_eval_sources.py`) — 필수 필드 누락·중복 id·해시 불일치 검출
  - [ ] 단건 다운로드 유틸(`scripts/fetch_eval_asset.py --url ... --id ...`) — 입력은 사람이 확정한 URL 1건, 배치 크롤링 금지
- [ ] Task 3: Real 150건 수집 (AC: 2, 3, 4)
  - [ ] 3주차 착수(Epic 1의 Story 1.5 게이트에서 정책 확정 + 시드 20건 선행), **4주차는 bbox 작업에 양보**, 5·6·7주차에 집중 배치 — 작업 주당 40~50건 페이스
  - [ ] 매주 누적 건수를 주간 체크포인트에 기록(실행 순서 강제 장치, PRD Revision Notes v1.2 §4)
  - [ ] 수집 중 이미지 유형 분포를 주기적으로 점검하고 부족 유형(대개 CHART/DOCUMENT)을 우선 보충
- [ ] Task 4: 제외 처리 및 1차 검증 (AC: 5, 6)
  - [ ] 라이선스 미확인/저해상도/이미지-기사 무관 샘플 제외 및 사유 기록
  - [ ] 검증 스크립트 통과 확인

### Dev Notes
- **NFR5가 이 스토리의 제약의 핵심**이다. "빠르니까"를 이유로 크롤러를 작성하면 프로젝트의 out-of-scope 선언과 정면 충돌한다. 수집 속도가 문제라면 규모를 줄일지언정 수집 방식을 바꾸지 않는다.
- 언어 정책: 주 실험은 영어(Fakeddit + DeBERTa)로 고정되어 있으므로(DEV_PLAN §7) FinFact-Eval도 **영어를 기본**으로 하고, 한국어 샘플은 데모용(Story 5.3)으로 20건 이내 별도 태그(`lang: ko`)로 보관한다. 한국어 샘플은 주 지표 집계에서 제외한다.
- 이미지 저장은 원본 해상도 유지 + 별도 640px 썸네일(리포트/데모용). repo에는 이미지 자체를 커밋하지 않고 `data/`는 gitignore 유지(README §6 규약 준수), annotation과 sources.jsonl만 버전 관리.
- 본문은 전문 대신 **title + 첫 3~5문장 발췌**로 저장하면 대부분의 라이선스 리스크가 사라지고, 모델 입력(최대 토큰 제한)에도 충분하다.

### Testing
- Unit: `validate_eval_sources.py` — 필드 누락/중복 id/해시 불일치 fixture로 검출 검증 (`tests/test_eval_sources.py`).
- 수동 점검: 무작위 10건에 대해 `source_url` 접속 → 기사·이미지 일치 및 라이선스 표기 재확인.
- 분포 점검: 이미지 유형 분포 집계 스크립트 실행으로 AC 4 충족 확인.

---

## Story 6.2: 통제된 Fake 샘플 생성 (mismatch 조작)

### Status
Draft

### Story
**As a** 과제 수행자,
**I want** 수집한 Real 샘플을 기반으로 mismatch 유형별 조작을 가해 Fake 샘플 150건을 생성하고 조작 내역을 기계 판독 가능하게 기록하는 절차,
**so that** "어떤 불일치가 심어졌는지"가 정답으로 알려진 통제된 평가셋을 확보하고 entity consistency의 기여를 유형별로 측정할 수 있다.

### Acceptance Criteria
1. Fake 샘플 150건이 생성되어 mismatch 유형 5종(`PERSON_MISMATCH`, `ORG_LOGO_MISMATCH`, `NUMBER_MISMATCH`, `DATE_MISMATCH`, `EVENT_MISMATCH`)에 **각 25~35건** 배분된다 — 목적은 유형별 지표 산출이 아니라 유형 쏠림 방지(커버리지)다.
2. 각 Fake 샘플은 `source_real_id`(파생 원본 Real 샘플)와 `manipulation`(조작 내용: 대상 필드, 원래 값, 바뀐 값, 조작 방법)을 기록한다.
3. 조작은 **entity 수준에서만** 이루어진다 — 이미지 교체/텍스트 값 치환/이미지-텍스트 재조합에 한정하며, 픽셀 단위 합성·딥페이크 생성은 하지 않는다(PRD Out of scope: forgery 자체 탐지 비대상).
4. 조작 후 샘플은 **표면적으로 자연스러워야 한다** — 텍스트만 읽거나 이미지만 봐서는 조작을 알 수 없고, 두 modality를 대조해야만 불일치가 드러나는지 생성 시 수동 확인한다.
5. Fake 파생에 사용된 Real 샘플은 Real 클래스에도 그대로 남기지 않는다 — 동일 원본에서 나온 Real/Fake 쌍이 평가셋에 동시 존재하지 않도록 원본을 분리 배정하고, 배정 결과가 검증 스크립트로 확인된다.
6. 조작 절차가 `docs/data/finfact_eval_manipulation.md`에 재현 가능한 수준으로 문서화된다(유형별 조작 레시피 + 사용 도구).

### Tasks / Subtasks
- [ ] Task 1: 유형별 조작 레시피 정의 (AC: 1, 3, 6)
  - [ ] PERSON: 기사 인물과 다른 실존 인물 사진으로 대표 이미지 교체 (같은 산업군 인물 선호 — 난이도 확보)
  - [ ] ORG_LOGO: 경쟁사 로고/행사 사진으로 교체 (예: NVIDIA 기사 + AMD 행사 이미지)
  - [ ] NUMBER: headline/본문의 금액·비율을 변조해 이미지 속 차트/문서 수치와 모순 유발
  - [ ] DATE: 기사 시점과 이미지 내 표기 날짜(문서·차트 축)를 어긋나게 조정
  - [ ] EVENT: 관계/사건 술어 변경(예: "공급 계약" → "인수 합의")으로 이미지 맥락과 모순 유발
- [ ] Task 2: 원본 배정 및 조작 수행 (AC: 1, 2, 5)
  - [ ] Real 풀에서 Fake 파생용 원본을 분리 배정(중복 사용 금지), 배정표 저장
  - [ ] 조작 수행 및 `manipulation` 레코드 작성 (필드/원값/신값/방법/작업일)
- [ ] Task 3: 자연스러움 검수 (AC: 4)
  - [ ] 각 샘플에 대해 "텍스트 단독 판별 가능?" / "이미지 단독 판별 가능?" 2문항 체크 — 하나라도 Yes면 재작업
- [ ] Task 4: 검증 스크립트 (AC: 5)
  - [ ] `scripts/validate_eval_set.py`에 Real/Fake 원본 중복 검사 및 유형별 개수 검사 추가

### Dev Notes
- **왜 통제 생성인가**: 실제 유통된 금융 가짜뉴스를 150건 수집하는 것은 라이선스·수집 난도상 8~10주 과제에서 비현실적이다. 대신 조작을 우리가 심으면 mismatch 유형 정답이 자동으로 확보되어 FR16(matching 판정 정확도)이 측정 가능해진다 — 이 셋의 목적은 "야생 분포 재현"이 아니라 **"entity 대조 능력의 통제된 측정"** 임을 리포트에 명시할 것.
- **한계 명시 의무**: 합성 Fake로 측정한 성능은 실제 분포에서의 성능 상한/하한 어느 쪽도 보장하지 않는다. Story 5.4 발표 자료의 "한계" 슬라이드에 반드시 포함한다.
- AC 4는 이 셋의 난도를 결정한다. 조작이 조잡하면(예: 해상도·톤이 확연히 다른 이미지 붙이기) 모델이 entity 대조 없이도 맞혀버려 핵심 가설 검증이 무의미해진다.
- 인물/로고 교체 시 사용하는 대체 이미지도 Story 6.1과 동일한 라이선스 기준을 통과해야 한다.
- 유형 배분은 균등(각 30건)을 기본으로 하되, 수집된 Real의 이미지 유형 분포에 따라 25~35 범위에서 조정한다(CHART/DOCUMENT가 적으면 NUMBER/DATE가 하한 쪽).

### Testing
- Unit: 유형별 개수·`source_real_id` 참조 무결성·Real/Fake 원본 중복 검사 (`tests/test_eval_set_integrity.py`).
- 수동 검수: 전 샘플 AC 4 체크리스트 통과 기록(스프레드시트 또는 jsonl 필드 `single_modality_detectable: false`).
- 재현성: 문서화된 레시피만 보고 제3자가 임의 1건을 동일 절차로 재생성 가능한지 1회 확인.

---

## Story 6.3: mismatch taxonomy 정의 및 annotation guideline

### Status
Draft

### Story
**As a** annotator(과제 수행자 본인),
**I want** mismatch 유형 6라벨의 정의·경계 사례·판정 규칙과 entity pair 기록 스키마를 담은 annotation guideline,
**so that** annotation이 일관되게 수행되고 라벨 품질(κ)이 측정 가능해진다.

### Acceptance Criteria
1. `docs/data/finfact_eval_taxonomy.md`에 6라벨(`PERSON_MISMATCH`, `ORG_LOGO_MISMATCH`, `NUMBER_MISMATCH`, `DATE_MISMATCH`, `EVENT_MISMATCH`, `NONE`)의 정의, 포함/제외 기준, 각 유형 예시 2건 이상, 경계 사례 판정 규칙이 기술된다 (FR14).
2. primary 유형 1개(필수) + secondary 유형 0개 이상(선택)의 부여 규칙과 우선순위(다중 불일치 시 어떤 것을 primary로 할지)가 명시된다.
3. 근거 entity pair 기록 스키마가 확정된다 — `evidence: [{visual_class, visual_value, textual_type, textual_value, verdict}]`, `verdict ∈ {MATCH, MISMATCH, UNKNOWN}`. Real 샘플도 MATCH pair를 최소 1개 기록한다(MATCH 판정의 정답 확보).
4. annotation 산출 포맷이 `annotations.jsonl` 1샘플 1행 JSON으로 확정되고 JSON Schema(`schemas/finfact_eval.schema.json`)로 형식 검증이 가능하다.
5. 파일럿 annotation 20건(Real 10 / Fake 10) 수행 후, 발견된 모호 사례를 반영해 guideline이 최소 1회 개정되며 개정 내역이 문서에 남는다.
6. UNKNOWN 사용 기준이 정의된다 — 사람이 봐도 판정 불가(이미지 내 해당 entity 부재/판독 불가)한 경우에만 사용하며, 모델의 UNKNOWN(FR12)과는 다른 의미임이 명시된다.
7. **guideline은 본 라벨링(Story 6.4) 시작 전에 확정되고 버전이 고정(freeze)된다** — 파일럿 기반 개정을 마친 시점의 문서를 `v1.0`으로 태그하고, 이후 변경은 반드시 버전을 올리고 **영향받는 기존 라벨의 재검토 범위를 함께 기록**한다. 라벨링 도중 규칙이 바뀌면 앞뒤 라벨의 기준이 달라져 κ도 정확성도 의미를 잃는다.
8. **seed 케이스 15건이 guideline에 포함된다** — 판정이 명확한 정답 예제(라벨 + 판정 근거 서술)를 유형별로 배치한다. 이는 (a) 판정 기준의 구체적 준거이자 (b) Story 6.4에서 제3자 스팟체크를 확보하지 못했을 때의 자기 점검 대체 수단이다. seed 케이스는 평가 지표 집계에서 제외하지 않되, 자기 점검에 사용된 사실을 기록한다.

### Tasks / Subtasks
- [ ] Task 1: taxonomy 문서 초안 (AC: 1, 2, 6)
  - [ ] 라벨별 정의·예시·반례 작성, EM-FEND/CFFN의 inconsistency 유형 구분을 참고로 인용
  - [ ] primary 선정 규칙: 조작 의도(Story 6.2의 `manipulation` 대상 필드)를 primary로 하되, annotation은 blind 수행이므로 조작 기록을 보지 않고 판정 → 사후 대조
- [ ] Task 2: 스키마 확정 (AC: 3, 4)
  - [ ] `annotations.jsonl` 필드 정의 — `id`, `label(0=REAL/1=FAKE, src/data/labels.py 규약 준수)`, `primary_type`, `secondary_types[]`, `evidence[]`, `ocr_targets[]`, `lang`, `annotator`, `annotated_at`
  - [ ] JSON Schema 작성 + `scripts/validate_annotations.py`
- [ ] Task 3: 파일럿 20건 및 guideline 개정 (AC: 5)
  - [ ] 파일럿 수행 → 판단이 흔들린 항목 목록화 → 규칙 추가
- [ ] Task 4: seed 케이스 작성 및 freeze (AC: 7, 8)
  - [ ] 유형별 정답 예제 15건 선정 + 판정 근거 서술 작성
  - [ ] guideline `v1.0` 태그 및 freeze 선언 — 이 시점 이후에만 Story 6.4 본 라벨링 착수 가능(선행 조건)

### Dev Notes
- 라벨 정수 규약은 repo 전역 규약(0=REAL, 1=FAKE, README §7)을 그대로 따른다 — Fakeddit 원본 규약과 반대이므로 혼동 주의.
- `ocr_targets[]`는 Story 6.5의 OCR 검증 subset 입력이다 — `{bbox, ground_truth_text}` 형태로 CHART/DOCUMENT/TEXT_REGION crop의 정답 문자열을 기록한다(FR17).
- **blind 원칙**: 조작을 만든 사람과 annotator가 동일 인물이므로 순수 blind는 불가능하다. 완화책으로 (a) 조작 후 최소 3일 경과 후 annotation, (b) `manipulation` 필드를 가린 상태로 작업, (c) Real/Fake를 섞어 무작위 순서로 제시한다. 이 한계는 리포트에 명시한다.
- evidence pair는 **모델 출력 형식과 1:1 대응**시킨다(Story 4.2의 matching 출력) — 대응이 어긋나면 FR16 측정 시 매칭 로직을 새로 짜야 한다. Story 4.2 착수 전 스키마 합의 필요.
- **guideline을 먼저 고정해야 하는 이유(Dev 지적)**: κ는 일관성 지표라 판정 기준이 글로 고정되지 않은 상태에서 재라벨링을 하면 "같은 오해를 두 번 반복"해도 높게 나온다. 기준을 문서로 박아두는 것이 정확성 방어의 1차선이고, 제3자 스팟체크(Story 6.4)가 2차선이다. 순서를 바꾸면 둘 다 무력해진다.
- **유형별 지표를 만들지 않기로 했지만 유형 라벨은 계속 기록한다** — 정성 사례 선정(Story 5.2)과 커버리지 점검에 필요하기 때문이다. 기록과 보고를 혼동하지 말 것.

### Testing
- Unit: JSON Schema 검증 — 유효/무효 fixture 각 3건 (`tests/test_annotation_schema.py`).
- 파일럿 결과 리뷰: 20건 중 규칙 미비로 판정 보류된 비율이 10% 이하가 될 때까지 guideline 개정.

---

## Story 6.4: annotation 수행 및 품질 관리 (Cohen's κ)

### Status
Draft

### Story
**As a** 과제 수행자,
**I want** FinFact-Eval 300건 전수 annotation과, 재라벨링 기반 일치도(κ) **및 제3자 스팟체크**를 함께 거는 품질 관리 절차,
**so that** 평가셋 라벨의 일관성뿐 아니라 **정확성**까지 근거를 갖추고 이후 모든 지표의 토대로 삼을 수 있다.

### Acceptance Criteria
0. **선행 조건**: Story 6.3의 guideline이 `v1.0`으로 freeze된 이후에만 본 라벨링을 시작한다. 라벨링 중 guideline 변경이 발생하면 버전을 올리고 영향 범위를 재라벨링한 뒤 그 사실을 기록한다 (AC 6에 포함).
1. 300건 전수에 대해 `annotations.jsonl`이 완성되고 Story 6.3의 스키마 검증을 통과한다.
2. 전체의 20%(60건) 표본에 대해 **최소 7일 간격의 blind 재라벨링**이 수행되고, 1차/2차 라벨로 intra-annotator Cohen's κ가 산출된다 — primary 유형(6-way)과 이진 라벨(Real/Fake) 각각에 대해.
3. κ ≥ 0.8을 품질 기준으로 하며, 미달 시 guideline 개정 → 해당 유형 전수 재검토 → 재측정 사이클을 수행하고 그 이력이 기록된다. **단 κ 통과만으로는 품질 게이트를 통과하지 않는다** — AC 3-1을 함께 만족해야 한다.
   - **3-1 (정확성 게이트, 필수)**: 제3자 검토자 1인이 무작위 **30건**을 guideline v1.0만 보고 독립 annotation하는 **스팟체크**를 수행한다. 원 라벨과의 불일치 건은 전수 검토되어 (a) 원 라벨 오류, (b) guideline 미비, (c) 검토자 오해 중 하나로 귀속된다. (a)/(b)로 판정된 건이 30건 중 **4건(13%)을 초과하면** guideline을 개정하고 해당 유형 전수를 재라벨링한 뒤 스팟체크를 재수행한다.
   - **3-2 (제3자 확보 불가 시 대체 경로)**: 검토자를 구하지 못한 경우, Story 6.3의 **seed 케이스 15건**을 라벨링 중간·종료 시점에 각 1회 자기 점검(정답을 가린 상태)하여 seed 정확도를 산출·보고한다. 이는 스팟체크의 **완전한 대체가 아니며**, 그 사실과 "정확성 검증이 self-report에 의존한다"는 한계를 AC 6 리포트와 발표 자료 한계 슬라이드에 명시한다.
4. 1차/2차가 불일치한 항목은 adjudication되어 최종 라벨과 **판정 사유**가 기록된다 (`adjudication_note`).
5. 조작 기록(Story 6.2 `manipulation`)과 annotation의 primary 유형 일치율이 산출된다 — 불일치 건은 "조작 의도와 실제 관측 가능한 불일치가 어긋난 사례"로 분리 검토한다.
6. 품질 관리 결과가 `docs/data/finfact_eval_quality.md`로 정리된다 — κ 수치(6-way/2-way), 스팟체크 결과와 귀속 분류, guideline 버전 이력, 불일치 사례 유형, adjudication 요약, 그리고 **"κ는 일관성 지표이며 정확성을 담보하지 않는다"는 한계 명시**.
7. 품질 게이트 통과 조건이 리포트 첫머리에 요약된다 — `κ(2-way) ≥ 0.8` **AND** `κ(6-way) ≥ 0.8` **AND** `스팟체크 오류 귀속 ≤ 4/30`(또는 3-2 대체 경로 수행 및 한계 명시). 하나라도 미달인 상태로 Story 6.5 패키징에 진입하지 않는다.

### Tasks / Subtasks
- [ ] Task 1: 전수 annotation (AC: 1)
  - [ ] 무작위 순서 제시, 세션당 30~40건으로 분할해 피로 편향 완화
- [ ] Task 2: 재라벨링 및 κ 산출 (AC: 2, 3)
  - [ ] 20% 표본 무작위 추출(seed 고정), 7일 이상 경과 후 2차 라벨링
  - [ ] `scripts/compute_annotation_agreement.py` — sklearn `cohen_kappa_score`, 6-way/2-way 각각 + confusion matrix 출력
- [ ] Task 3: adjudication (AC: 4)
  - [ ] 불일치 항목 목록화 → 최종 라벨 확정 → 사유 기록
- [ ] Task 4: 제3자 스팟체크 30건 (AC: 3-1, 3-2)
  - [ ] 검토자 섭외(동료/조교) → guideline v1.0 + 무작위 30건 전달(원 라벨 비공개) → 독립 annotation 회수
  - [ ] 불일치 건 귀속 분류((a) 원 라벨 오류 / (b) guideline 미비 / (c) 검토자 오해) 및 기준 초과 시 재라벨링 사이클
  - [ ] 섭외 실패 시 3-2 seed 자기 점검 경로로 전환하고 한계 기록
- [ ] Task 5: 조작 의도 대조 (AC: 5)
  - [ ] `manipulation.target_field` ↔ `primary_type` 매핑 표로 일치율 계산, 불일치 사례 정성 분석
- [ ] Task 6: 품질 리포트 작성 (AC: 6, 7)

### Dev Notes
- **DEV_PLAN §2.3의 "2인 교차 검증"은 단일 인원 과제에서 성립하지 않는다.** 본 스토리는 이를 intra-annotator 재라벨링 κ + adjudication + **제3자 스팟체크 30건**의 조합으로 대체한다. DEV_PLAN 문구는 후속 조치로 정정 대상(PRD Revision Notes v1.1·v1.2 참조).
- **κ는 일관성(consistency)이지 정확성(accuracy)이 아니다** — 이것이 v1.2에서 스팟체크를 추가한 이유다. 같은 사람이 같은 오해를 두 번 반복하면 κ는 0.9가 나오면서 라벨은 전부 틀려 있을 수 있고, 이 실패 모드는 재라벨링을 아무리 반복해도 드러나지 않는다. 방어선은 3중이다: (1) 라벨링 **전** guideline freeze(Story 6.3 AC 7), (2) 재라벨링 κ(일관성), (3) 제3자 스팟체크(정확성). 셋 중 (3)이 빠지면 정확성에 대한 외부 증거가 0이 되므로, 대체 경로(3-2)를 쓰더라도 그 사실을 반드시 드러낸다.
- 스팟체크 30건은 정밀 추정용이 아니라 **체계적 오류 탐지용**이다 — 오류율 13%(4/30)를 넘으면 "개별 실수"가 아니라 "기준이 잘못 잡혔다"고 보는 것이 이 임계값의 취지다. 30건에서 오류율의 CI는 넓으므로 스팟체크 오류율 자체를 품질 수치로 인용하지 말 것.
- κ 해석: 0.8 이상 "almost perfect", 0.6~0.8 "substantial". 0.6 미만이면 라벨을 근거로 한 FR16 지표 자체가 흔들리므로 반드시 재작업한다.
- 조작 의도와 관측 유형의 불일치(AC 5)는 버그가 아니라 **발견**이다 — 예: NUMBER를 조작했는데 이미지에 해당 수치가 없어 사람도 판정 불가(UNKNOWN)한 경우, 그 샘플은 "모델이 못 맞히는 게 정상"이며 지표 해석 시 분리해야 한다.
- κ 계산 시 UNKNOWN을 별도 클래스로 포함할 것(제외하면 일치도가 인위적으로 올라간다).

### Testing
- Unit: κ 계산 스크립트 — 알려진 정답이 있는 소형 fixture(완전일치 κ=1.0, 무작위 κ≈0) 검증 (`tests/test_annotation_agreement.py`).
- 절차 검증: 재라벨링 간격이 7일 이상임을 타임스탬프로 확인.
- 절차 검증: 본 라벨링 시작 시각이 guideline v1.0 freeze 시각 이후임을 확인(AC 0).
- 스팟체크 산출물 검증: 30건 귀속 분류 표가 완전하고((a)/(b)/(c) 미분류 0건) 기준 초과 시 재라벨링 이력이 남았는지 확인.

---

## Story 6.5: 데이터셋 패키징 및 held-out 무결성 검증

### Status
Draft

### Story
**As a** 실험 수행자/평가자,
**I want** FinFact-Eval을 고정된 스키마·고정된 id 목록으로 패키징하고 학습 데이터와의 오염이 없음을 자동 검증하는 장치,
**so that** 이 셋의 평가 수치를 held-out 결과로 신뢰할 수 있고 제3자가 재사용할 수 있다.

### Acceptance Criteria
1. `data/finfact_eval/` 구조가 확정된다 — `annotations.jsonl`, `sources.jsonl`, `manifest.json`(버전, 생성일, 샘플 수, 유형별 분포, 스키마 버전), `ocr_subset.jsonl`, `images/`(gitignore), `thumbnails/`.
2. 평가셋 id 목록이 `data/finfact_eval/eval_ids.txt`로 고정되며, 이후 변경 시 `manifest.json`의 버전이 올라가고 변경 이력이 기록된다.
3. `scripts/check_holdout_integrity.py`가 (a) FinFact-Eval id와 Fakeddit train/val split의 교집합 0, (b) 이미지 sha256 기준 중복 0, (c) 텍스트 근사 중복(정규화 후 완전 일치 또는 높은 n-gram 중첩) 0을 검증하고, 위반 시 non-zero exit한다 (NFR8).
4. 이 검증이 CI/테스트에 편입되어 학습 파이프라인이 실수로 FinFact-Eval을 학습에 사용하는 경로가 차단된다 — 학습 데이터 로더가 `eval_ids.txt`에 속한 샘플을 만나면 예외를 던진다.
5. 재배포 패키지가 구성된다 (NFR9) — 라이선스 확인된 자산만 포함하고, 나머지는 `sources.jsonl`의 URL + sha256으로 제3자가 스스로 재구성할 수 있도록 `scripts/rebuild_finfact_eval.py`(입력: sources.jsonl, 단건 다운로드 반복)를 제공한다.
6. `docs/data/finfact_eval_datasheet.md`(간이 datasheet)가 작성된다 — 목적, 구성, 수집 방법, 조작 방법, 라벨링 절차·품질, 알려진 편향과 한계, 사용 시 주의(합성 Fake이므로 야생 분포와 다름).

### Tasks / Subtasks
- [ ] Task 1: 디렉토리·manifest 스키마 확정 및 생성 스크립트 (AC: 1, 2)
- [ ] Task 2: 무결성 검증 스크립트 구현 (AC: 3)
  - [ ] id 교집합, 이미지 해시 중복, 텍스트 정규화 후 중복/n-gram 중첩(임계값 config화)
- [ ] Task 3: 학습 경로 차단 가드 (AC: 4)
  - [ ] `src/data/`의 데이터셋 클래스에 `forbidden_ids` 체크 추가(Epic 1 코드 변경은 최소 침습 — 로더 초기화 시 1회 검사)
  - [ ] pytest에 무결성 검증 케이스 추가
- [ ] Task 4: 재배포 패키지 및 rebuild 스크립트 (AC: 5)
- [ ] Task 5: datasheet 작성 (AC: 6)

### Dev Notes
- Epic 1의 데이터 로더에 손을 대는 유일한 지점이 AC 4다. 기존 동작을 바꾸지 않고 **검사만 추가**한다(`forbidden_ids`가 비어 있으면 no-op).
- 텍스트 근사 중복 검사가 필요한 이유: FinFact-Eval Real 샘플의 원문 기사가 Fakeddit의 어떤 게시물과 동일 사건을 다룰 수 있다. 완전 배제는 불가능하므로 "동일 문서 수준 중복"만 차단하고, 사건 수준 중복 가능성은 datasheet의 한계에 기재한다.
- 버전 관리: 셋이 확정된 뒤 샘플을 추가하면 이전 실험 결과와 비교 불가해진다. **Story 5.1 ablation 실행 시점의 manifest 버전을 결과에 함께 기록**하고, 그 이후 추가는 v2로 분리한다.
- `rebuild_finfact_eval.py`도 크롤러가 아니다 — 입력이 사람이 확정해 커밋한 URL 목록이며 새 URL을 스스로 발견하지 않는다는 점을 스크립트 docstring에 명시한다(NFR5 오해 방지).

### Testing
- Unit: 무결성 검증 스크립트 — 오염된 fixture(중복 id 1건, 동일 해시 1건) 입력 시 non-zero exit 검증 (`tests/test_holdout_integrity.py`).
- Unit: 로더 가드 — forbidden id 포함 배치 로드 시 예외 발생 검증.
- Smoke: `rebuild_finfact_eval.py`를 3건 샘플로 실행해 해시 일치 재구성 확인.

---

## Story 6.6: FinFact-Eval 평가 하네스 — 금융 특화 성능·matching 판정 정확도·OCR CER

### Status
Draft

### Story
**As a** 연구자,
**I want** FinFact-Eval을 입력으로 (a) A1~A6 구성별 금융 도메인 분류 성능(보조 표), (b) cross-modal matching 판정 정확도(전체 단위 + CI), (c) OCR CER을 산출하는 평가 하네스,
**so that** "entity 대조 능력"을 실측 수치로 뒷받침하고 도메인 이전을 정성적으로 확인할 수 있다.

### Acceptance Criteria
1. `scripts/eval_finfact.py`가 checkpoint + FinFact-Eval을 입력으로 Accuracy/Precision/Recall/F1/AUROC를 산출하고 `metrics_custom_fin.json`으로 저장한다 — Story 5.1의 `--eval-sets custom_fin` 경로와 동일한 출력 계약을 따른다. **이 표는 보조 표로 표시되며, 구성 간 차이에 대한 유의성 주장을 하지 않는다**(FR10 — 핵심 가설 검증은 Fakeddit 금융 subset 담당). 리포트 해당 절 머리에 그 사실이 한 줄로 명시된다.
2. **Matching 판정 정확도(FR16)** 가 **전체 단위 1세트**로 산출된다 — annotation `evidence[]`를 정답으로 하여 (a) MATCH/MISMATCH/UNKNOWN 3-way accuracy, (b) MISMATCH 클래스 Precision/Recall/F1(pair 총계), (c) 3-way confusion matrix, (d) **(a)(b)의 Wilson 95% CI 병기**(필수 — CI 없는 수치는 리포트에 싣지 않는다).
   - **2-1**: **mismatch 유형별 지표 표는 산출하지 않는다.** 유형 라벨은 계속 기록하고 per-pair 원자료(`pair_results.jsonl`)에 유형을 포함하되, 유형별 P/R/F1을 집계한 표는 리포트·발표 자료 어디에도 싣지 않는다. 유형별 특성은 Story 5.2의 **정성 사례 3~5건**으로 전달한다.
   - **2-2**: 리포트에 이 결정의 근거가 각주로 남는다 — "유형당 n≈30에서 recall 0.80의 Wilson 95% CI는 [0.63, 0.90](폭 27%p)이므로 유형 간 차이가 통계적으로 판별되지 않는다."
3. 정답 evidence pair와 모델 출력 pair의 정렬 규칙이 명시적으로 구현된다 — (visual_class, textual_type) 기준 매칭, 정답에 있으나 모델이 내놓지 않은 pair는 `missed`, 반대는 `spurious`로 별도 집계한다.
4. **OCR CER(FR17)** 이 `ocr_subset.jsonl`(crop + 정답 문자열)에 대해 산출된다 — 문자 단위 Levenshtein 기반 CER, 전체 및 클래스별(CHART/DOCUMENT/TEXT_REGION).
5. OCR subset이 50 crop 미달이면 스크립트가 CER을 산출하지 않고 명시적 경고와 함께 "측정 불가 — 수치 추출 정확도(정성 검증)로 대체"를 리포트에 기록한다 (FR17 탈출 규칙).
6. 결과가 `docs/results/finfact_eval_report.md`로 정리되어 Story 5.1/5.2/5.4가 그대로 인용할 수 있다.
7. 리포트 머리에 **표본 규모와 주장 범위**가 명시된다 — 전체 샘플 수, 평가된 pair 수, 적용된 fallback ladder 단계, 그리고 "이 셋으로 하는 주장 / 하지 않는 주장"(FR13 역할 3종 + 가설 검증 비대상) 요약.

### Tasks / Subtasks
- [ ] Task 1: 분류 성능 평가 경로 (AC: 1)
  - [ ] Story 5.1의 `--eval-sets` 인터페이스에 `custom_fin` 데이터 어댑터 등록 (중복 구현 금지 — 공통 evaluation 모듈 재사용, NFR6)
- [ ] Task 2: matching 판정 정확도 산출 (AC: 2, 3)
  - [ ] 정답/예측 pair 정렬기 구현 + missed/spurious 집계
  - [ ] 3-way accuracy, MISMATCH P/R/F1(총계), Wilson 95% CI, confusion matrix 출력
  - [ ] per-pair 원자료 `pair_results.jsonl` 저장(유형 라벨 포함) — 정성 사례 선정(Story 5.2) 입력용. **유형별 집계 표 생성 코드는 작성하지 않는다**(AC 2-1)
- [ ] Task 3: OCR CER 산출 (AC: 4, 5)
  - [ ] crop 추출 → OCR 추론 → 정규화(공백/통화기호/천단위 구분자 규칙) → CER 계산
  - [ ] subset 규모 게이트 및 대체 지표 경로
- [ ] Task 4: 리포트 생성 (AC: 1, 6, 7)
  - [ ] Markdown 표 자동 생성(표본 수·CI 열 포함), 머리말에 주장 범위 요약 + AC 2-2 각주 삽입, `docs/results/`에 저장

### Dev Notes
- **의존성**: Story 4.1(Visual Entity Recognition), 4.2(Matching), 6.5(패키징) 완료 후 착수 가능. Story 5.1보다 먼저 완료되어야 5.1 AC 6이 성립한다.
- **CER 정규화 규칙이 결과를 좌우한다**: `$1,234.5` vs `1234.5`를 다르게 볼지 미리 정하고 문서화할 것. 본 프로젝트의 관심은 "수치가 맞게 읽혔는가"이므로 **통화기호·천단위 구분자·공백은 정규화하여 제거**하고 CER을 계산하며, raw CER도 함께 병기한다.
- **UNKNOWN 처리 주의**: 모델의 UNKNOWN(FR12, 신뢰도 미달)과 annotation의 UNKNOWN(사람도 판정 불가)은 의미가 다르다. 3-way accuracy에서는 동일 클래스로 취급하되, "정답 MISMATCH인데 모델 UNKNOWN"은 별도 셀로 보고해야 FR12의 오류 전파 차단 정책이 recall을 얼마나 희생했는지 보인다(Story 5.2 AC와 연결).
- **유형별 표를 만들지 않는 이유(v1.2 결정, 반복 강조)**: 유형당 n≈30에서 recall 0.80의 Wilson 95% CI는 [0.63, 0.90]으로 폭이 27%p다. PERSON 0.85 vs NUMBER 0.70이 나와도 두 CI가 완전히 겹치므로 "어느 유형에서 약한가"는 판별 불가다. 구현 중 "표가 있으면 보기 좋으니까" 되살리고 싶어지겠지만, 그 표는 독자에게 존재하지 않는 결론을 읽게 만든다. 유형별 통찰은 정성 사례로만 전달한다.
- 지표는 소수점 이하 1자리까지만 보고하고 항상 CI를 병기한다.
- 이 스토리의 산출물이 곧 **FR16의 유일한 실측 근거**다(Fakeddit엔 entity pair 정답이 없다) — Story 5.4 발표에서 Fakeddit 금융 subset의 A1~A6 곡선(가설 검증)과 FinFact-Eval의 matching 정확도(대조 능력 직접 측정)를 **역할이 다른 두 증거**로 나란히 제시하는 것이 올바른 프레이밍이다. 둘을 같은 축에 놓고 비교하지 말 것.

### Testing
- Unit: pair 정렬기 — 정답/예측 fixture(정확 매칭, missed, spurious 각 1건 이상)로 집계 검증 (`tests/test_matching_eval.py`).
- Unit: CER 계산 — 알려진 문자열 쌍(정답/예측)으로 수치 검증, 정규화 on/off 차이 확인.
- Unit: subset 규모 게이트 — 49건 fixture 입력 시 CER 미산출 + 경고 경로 진입 검증.
- Smoke: 10샘플 축소 실행으로 리포트까지 end-to-end 생성 확인.
