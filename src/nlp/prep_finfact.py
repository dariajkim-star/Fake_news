"""Fin-Fact 전처리: 층화 split + claim-guided evidence 문장 선택 (README §5.2·§8.2).

결정 사항 반영:
  - 데이터셋이 split을 제공하지 않으므로 70/15/15 층화 분할을 직접 만든다
    (seed 42, 라벨 stratify, 분할 해시 기록 — Vision과 동일 규약).
  - evidence는 head truncation하지 않는다. 문장 단위로 쪼개 claim과의 TF-IDF
    유사도 상위 문장부터 256 토큰 예산까지 담고, 선택된 문장은 원문 순서를 유지한다.
    이는 외부 검색이 아니라 "제공된 evidence 내부에서의 문장 선택"이다 (§2.1과 충돌 없음).

사용:
    python -m src.nlp.prep_finfact
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

LABELS = {"false": 0, "true": 1, "neutral": 2}
EVIDENCE_BUDGET = 256          # N2 max_len과 동일 (special token 여유 포함 안 함)
SEED = 42

_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT.split(text or "") if s.strip()]


def select_evidence(claim: str, evidence: str, tokenizer, budget: int) -> tuple[str, dict]:
    """claim과 유사한 evidence 문장을 예산까지 담는다. 원문 순서 유지."""
    sents = split_sentences(evidence)
    if not sents:
        return "", {"sents_total": 0, "sents_kept": 0, "tokens": 0}

    from sklearn.feature_extraction.text import TfidfVectorizer
    try:
        vec = TfidfVectorizer(lowercase=True, stop_words="english").fit([claim] + sents)
        import numpy as np
        sims = (vec.transform([claim]) @ vec.transform(sents).T).toarray()[0]
        order = np.argsort(-sims)
    except ValueError:                      # claim이 stop word뿐인 극단 케이스
        order = range(len(sents))

    lens = [len(tokenizer(s, add_special_tokens=False)["input_ids"]) for s in sents]
    picked: set[int] = set()
    used = 0
    for i in order:
        if used + lens[i] > budget:
            continue
        picked.add(int(i))
        used += lens[i]
    kept = [sents[i] for i in sorted(picked)]          # 원문 순서 복원
    return " ".join(kept), {"sents_total": len(sents), "sents_kept": len(kept), "tokens": used}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("data/model/processed/text"))
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer
    import pandas as pd
    from sklearn.model_selection import train_test_split

    ds = load_dataset("amanrangapur/Fin-Fact")["train"]
    df = pd.DataFrame({"claim": ds["claim"], "evidence": ds["evidence"], "label_str": ds["label"]})
    df["label"] = df["label_str"].map(LABELS)
    n0 = len(df)
    df = df.dropna(subset=["claim", "label"]).reset_index(drop=True)
    print(f"로드 {n0:,} → 유효 {len(df):,}")

    # 70/15/15 라벨 층화
    tr, rest = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=SEED)
    va, te = train_test_split(rest, test_size=0.50, stratify=rest["label"], random_state=SEED)

    tok = AutoTokenizer.from_pretrained("microsoft/deberta-v3-small")
    stats = {"sents_total": 0, "sents_kept": 0, "tokens": 0}
    def apply(row):
        sel, st = select_evidence(row["claim"], str(row["evidence"] or ""), tok, EVIDENCE_BUDGET)
        for k in stats:
            stats[k] += st[k]
        return sel

    args.out.mkdir(parents=True, exist_ok=True)
    split_hash = {}
    for name, part in (("train", tr), ("val", va), ("test", te)):
        part = part.copy()
        part["evidence_selected"] = part.apply(apply, axis=1)
        part[["claim", "evidence_selected", "label", "label_str"]].to_csv(
            args.out / f"{name}.csv", index=False, encoding="utf-8-sig")
        split_hash[name] = hashlib.sha256(
            "".join(sorted(part["claim"].astype(str))).encode()).hexdigest()[:16]
        dist = part["label_str"].value_counts().to_dict()
        print(f"  {name:>5}: {len(part):,}  {dist}")

    keep_rate = stats["sents_kept"] / max(stats["sents_total"], 1)
    report = {
        "seed": SEED, "evidence_budget_tokens": EVIDENCE_BUDGET,
        "rows": {"train": len(tr), "val": len(va), "test": len(te)},
        "sentence_keep_rate": round(keep_rate, 4),
        "avg_selected_tokens": round(stats["tokens"] / (len(tr) + len(va) + len(te)), 1),
        "split_sha256": split_hash,
        "method": "claim-guided TF-IDF sentence selection within provided evidence "
                  "(not retrieval); selected sentences keep original order",
    }
    (args.out / "prep_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"문장 보존율 {keep_rate:.1%}, 평균 선택 토큰 {report['avg_selected_tokens']}")


if __name__ == "__main__":
    main()
