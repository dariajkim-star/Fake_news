"""금융 키워드 subset 필터링 (Story 1.2 AC4 / FR11).

주의(prd.md#Checklist 리스크): 금융 subset의 **규모 부족**이 주요 리스크이므로
필터링 시 항상 통계 리포트를 출력해 조기에 규모를 확인한다.

매칭 규칙: 대소문자 무시 + 단어 경계(`\\b`) — "restocking"은 "stock"에
매칭되지 않는다. 다중 단어 키워드("interest rate")는 공백을 유연하게 처리한다.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pandas as pd

from src.data.preprocess import manifest_report

#: config에서 덮어쓰지 않았을 때 쓰는 기본 금융 키워드 (Story 1.2 AC4 예시 포함)
DEFAULT_FINANCIAL_KEYWORDS: tuple[str, ...] = (
    "stock",
    "stocks",
    "share price",
    "invest",
    "investor",
    "investment",
    "ceo",
    "cfo",
    "earnings",
    "revenue",
    "profit",
    "ipo",
    "crypto",
    "cryptocurrency",
    "bitcoin",
    "merger",
    "acquisition",
    "bank",
    "banking",
    "nasdaq",
    "dow jones",
    "s&p",
    "hedge fund",
    "dividend",
    "bankruptcy",
    "wall street",
    "interest rate",
    "inflation",
    "market cap",
    "sec filing",
)


def compile_keyword_pattern(keywords: Sequence[str]) -> re.Pattern[str]:
    """키워드 리스트를 대소문자 무시 + 단어 경계 정규식으로 컴파일한다."""
    if not keywords:
        raise ValueError("금융 키워드 리스트가 비어 있습니다")
    alternatives = []
    for keyword in keywords:
        normalized = " ".join(str(keyword).strip().split())
        if not normalized:
            continue
        # 다중 단어는 공백 1개 이상으로 유연하게 매칭
        escaped = r"\s+".join(re.escape(token) for token in normalized.split(" "))
        # 키워드가 기호로 시작/끝나면 \b가 성립하지 않으므로 조건부로 붙인다 (예: "s&p")
        prefix = r"\b" if normalized[0].isalnum() else ""
        suffix = r"\b" if normalized[-1].isalnum() else ""
        alternatives.append(f"{prefix}{escaped}{suffix}")
    if not alternatives:
        raise ValueError("유효한 금융 키워드가 없습니다")
    return re.compile("|".join(alternatives), flags=re.IGNORECASE)


def matched_keywords(text: str, pattern: re.Pattern[str]) -> list[str]:
    """텍스트에서 매칭된 키워드(소문자 정규화) 목록 — 리포트/디버깅용."""
    return sorted({" ".join(match.group(0).lower().split()) for match in pattern.finditer(text or "")})


def is_financial(text: str, pattern: re.Pattern[str]) -> bool:
    """텍스트가 금융 키워드를 하나라도 포함하는지."""
    return pattern.search(text or "") is not None


def filter_financial(
    manifest: pd.DataFrame,
    keywords: Sequence[str] = DEFAULT_FINANCIAL_KEYWORDS,
    text_columns: Sequence[str] = ("title", "body"),
) -> pd.DataFrame:
    """금융 키워드에 매칭되는 행만 남긴 subset manifest를 반환한다.

    반환 프레임에는 진단용 ``matched_keywords`` 컬럼이 추가된다.
    """
    pattern = compile_keyword_pattern(keywords)
    columns = [column for column in text_columns if column in manifest.columns] or ["text"]

    def _joined(row: Mapping[str, object]) -> str:
        return " ".join(str(row.get(column) or "") for column in columns)

    records = manifest.to_dict(orient="records")
    hits = [matched_keywords(_joined(row), pattern) for row in records]
    mask = [bool(hit) for hit in hits]

    subset = manifest[pd.Series(mask, index=manifest.index)].copy()
    subset["matched_keywords"] = [
        ";".join(hit) for hit, keep in zip(hits, mask) if keep
    ]
    return subset.reset_index(drop=True)


def keyword_counts(subset: pd.DataFrame) -> dict[str, int]:
    """subset에서 키워드별 히트 수 (어떤 키워드가 subset을 지배하는지 확인)."""
    counts: dict[str, int] = {}
    for value in subset.get("matched_keywords", pd.Series(dtype=str)):
        for keyword in str(value).split(";"):
            if keyword:
                counts[keyword] = counts.get(keyword, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def financial_report(
    subset: pd.DataFrame,
    full: pd.DataFrame | None = None,
    name: str = "financial_subset",
) -> dict[str, object]:
    """subset 규모·라벨 분포·키워드 히트 리포트 (규모 부족 리스크 조기 감지)."""
    report = manifest_report(subset, name=name)
    if full is not None and len(full):
        report["coverage_ratio"] = round(len(subset) / len(full), 4)
        report["n_full"] = int(len(full))
    report["keyword_counts"] = keyword_counts(subset)
    return report


def write_financial_report(report: Mapping[str, object], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(dict(report), fp, ensure_ascii=False, indent=2)
    return path


def load_keywords(source: Iterable[str] | None) -> list[str]:
    """config에서 온 키워드(없으면 기본값)를 리스트로 정규화한다."""
    if source is None:
        return list(DEFAULT_FINANCIAL_KEYWORDS)
    keywords = [str(item) for item in source]
    return keywords or list(DEFAULT_FINANCIAL_KEYWORDS)
