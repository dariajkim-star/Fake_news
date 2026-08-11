# 문제정의 분석 스크립트

README §1.1의 숫자를 만든 스크립트다. **숫자를 인용하려면 그 숫자가 어떻게 나왔는지 재현 가능해야 한다**는
원칙에 따라 리포에 둔다. 모델 학습과는 무관하며, 파이프라인 실행에 필요하지 않다.

입력은 `fake_news_painpoint_crawlers/output/*.csv` (크롤링 raw, git 제외 — 재현하려면 크롤러 재실행).

| 스크립트 | 무엇을 만드는가 | README 참조 |
|---|---|---|
| `incident_funnel.py` | 기사 → 사건 퍼널 (2,995 → 2,806 → 778 → 189) | §1.1.2 |
| `painpoint_tags.py` | 채널별·축별 키워드 태그 분포, 피해액 추출 시도 | §1.1 |
| `dfdc_family_sizing.py` | DFDC part별 family 규모 (영상 받기 전 part 선택 근거) | §5.1 |

```bash
python scripts/research/incident_funnel.py
```

```bash
python scripts/research/dfdc_family_sizing.py 10
```

## 주의 — 이 스크립트들이 만든 숫자를 그대로 믿지 말 것

두 가지는 **의도적으로 폐기**했고, 그 이유가 README에 남아 있다.

- **피해액 자동 합산** (`painpoint_tags.py`): 본문에서 금액을 정규식으로 뽑으면 연기금 규모(1,400조원)·
  시장안정 프로그램(100조원) 같은 무관한 숫자가 섞인다. 실행해보면 나오지만 **쓰지 않는다.**
  피해액은 제목에 명시된 개별 사건만 인용한다.
- **축 간 동시 출현률**: 수집분 대부분이 제목+요약(약 150자)뿐이라 두 축이 한 스니펫에 같이 등장할
  확률이 구조적으로 과소추정된다. 채널 특성의 산물이지 현상의 성질이 아니다.

`incident_funnel.py`의 클러스터링도 완전하지 않다 — 제목 유사도 기반이라 같은 사건이 다른 금액으로
보도되면 분리된다. 최종 15건은 이 출력을 **후보로 삼아 수동 병합·검증**한 결과이며,
정본은 [`data/research/coded_cases.csv`](../../data/research/coded_cases.csv)다.
