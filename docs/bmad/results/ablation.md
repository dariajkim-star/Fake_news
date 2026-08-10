# Epic 1 Ablation Table (Phase 1 baseline)

> 자동 생성: `python scripts/collect_ablation.py` · 평가 split: **test**
> 라벨 규약 0=REAL, 1=FAKE · Precision/Recall/F1은 FAKE(양성=1) 기준

| Model | exp_name | Accuracy | Precision | Recall | F1 | Auroc | Status |
|---|---|---|---|---|---|---|---|
| BERT only | `bert_only` | — | — | — | — | — | missing |
| ResNet only | `resnet_only` | — | — | — | — | — | missing |
| BERT+Image (late fusion) | `late_fusion` | — | — | — | — | — | missing |

## F1 비교 (AC4)

아직 비교할 수 없다 — late fusion 또는 단일 모달 baseline의 test 지표가 비어 있다. 실데이터 학습 후 `scripts/collect_ablation.py`를 다시 실행할 것.

## 비고

실데이터·실가중치 미확보 상태 — 세 행 모두 test 지표가 비어 있다. Fakeddit manifest와 bert-base-uncased/ImageNet 가중치 확보 후 configs/{bert_only,resnet_only,late_fusion}.yaml을 학습하고 scripts/evaluate.py --split test로 지표를 채운 뒤 이 스크립트를 재실행할 것.
