# 데이터 준비 가이드 (Fakeddit)

> Story 1.2 산출물. 라이선스상 원본 데이터는 repo에 포함되지 않으며 `.gitignore`의 `data/`로 제외된다.

## 1. 디렉토리 규약

```
data/fakeddit/
├── raw/          # 공식 배포 tsv (multimodal_train / _validate / _test_public)
├── images/       # <sample_id>.jpg
└── processed/    # 파이프라인 산출물
    ├── manifest.csv          # 클린 manifest (전체)
    ├── train.csv / val.csv / test.csv   # 고정 분할 — 모든 실험이 이 파일을 참조
    ├── split_report.json     # 분할별 샘플 수·라벨 분포
    ├── preprocess_report.json
    └── financial/            # 금융 subset (FR11)
        ├── train.csv / val.csv / test.csv
        └── financial_report.json
```

## 2. 라벨 규약 (중요)

| 출처 | 값 | 의미 |
|---|---|---|
| 내부 표준 `NewsSample.label` | `0` | REAL |
| 내부 표준 `NewsSample.label` | `1` | FAKE |
| Fakeddit 원본 `2_way_label` | `1` | true (real) |
| Fakeddit 원본 `2_way_label` | `0` | fake |

즉 매핑은 항등이 아니라 **반전**이다 (`2_way_label 1 -> 0`, `0 -> 1`).
단일 진실 공급원은 `src/data/labels.py`의 `FAKEDDIT_2WAY_TO_INTERNAL`이며,
`tests/test_labels.py`가 이 매핑을 고정한다. 라벨 반전은 전 실험을 무효화하므로
직접 `int(row["2_way_label"])`를 쓰지 말고 반드시 `map_fakeddit_2way_label()`을 거칠 것.

> 다운로드한 배포본의 인코딩이 위와 다르면(배포 버전 차이) **상수와 테스트를 함께**
> 수정하고 본 문서를 갱신한다.

## 3. 준비 절차

```bash
# 0) 디렉토리 생성 + 배치 상태 점검 (네트워크 접근 없음)
python scripts/download_fakeddit.py --root data/fakeddit --check

# 1) 공식 배포처에서 multimodal tsv를 내려받아 data/fakeddit/raw/ 에 배치
#    https://github.com/entitize/Fakeddit  (arXiv:1911.03854, 비상업적 연구용)

# 2) 실험에 쓸 subset 이미지만 내려받기 (전량은 비현실적)
python scripts/download_fakeddit.py --root data/fakeddit \
    --metadata data/fakeddit/raw/multimodal_train.tsv --limit 5000 --download-images

# 3) 클린 manifest + 고정 분할 생성 (공식 분할 사용)
python scripts/preprocess_fakeddit.py --root data/fakeddit

#    또는 단일 tsv에서 seed 고정 stratified split
python scripts/preprocess_fakeddit.py --root data/fakeddit \
    --metadata data/fakeddit/raw/multimodal_train.tsv --split-mode stratified --seed 42

# 4) 금융 키워드 subset 생성 (FR11)
python scripts/filter_financial.py --config configs/fakeddit.yaml
```

전처리는 (a) 텍스트 결측, (b) 이미지 파일 부재, (c) PIL로 열리지 않는 손상 이미지를
제거하고 제거 건수를 `preprocess_report.json`에 남긴다. 이미지 없이 텍스트만으로
실험하려면 `--allow-missing-image`를 준다.

## 4. manifest 스키마

`sample_id, image_path, title, body, text, label, source` — architecture.md의
`NewsSample`과 호환된다. Fakeddit은 `clean_title`만 제공하므로 `body`는 빈 문자열,
`text`는 학습 입력용 합성 필드, `source`는 `"fakeddit"`이다.
`image_path`는 데이터 루트 기준 상대경로이며 config `data.image_root`로 복원된다.

## 5. 학습에서의 사용

```yaml
data:
  name: fakeddit          # src/data/registry.py 레지스트리 키
  modality: both          # text | image | both
  tokenizer: {name: bert-base-uncased, max_length: 128}
  subsample: 20000        # 단일 GPU 제약(NFR1)
```

```bash
python scripts/train.py --config configs/fakeddit.yaml
```

`data.tokenizer.name`을 `null`로 두면 tokenizer 없이 raw text 문자열이 배치의
`text` 키로 전달된다 (네트워크 없는 환경/테스트 경로).
