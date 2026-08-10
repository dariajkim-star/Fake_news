"""금융 키워드 필터 테스트 — Story 1.2 AC4, AC7."""

from __future__ import annotations

import pytest

from src.data.financial import (
    DEFAULT_FINANCIAL_KEYWORDS,
    compile_keyword_pattern,
    filter_financial,
    financial_report,
    is_financial,
    keyword_counts,
    load_keywords,
    matched_keywords,
)


@pytest.fixture
def pattern():
    return compile_keyword_pattern(["stock", "ceo", "interest rate", "s&p"])


@pytest.mark.parametrize(
    "text",
    [
        "tesla stock soars",
        "STOCK market crash",  # 대소문자 무시
        "the ceo resigned",
        "fed raises the interest  rate today",  # 다중 공백 허용
    ],
)
def test_matches_financial_text(pattern, text):
    assert is_financial(text, pattern) is True


@pytest.mark.parametrize(
    "text",
    [
        "restocking the shelves",  # 부분 문자열 — 매칭되면 안 됨
        "stockholm is cold",
        "the cerebral cortex",
        "interesting rates of decay",
        "",
    ],
)
def test_rejects_substring_only_matches(pattern, text):
    assert is_financial(text, pattern) is False


def test_symbol_keyword_matches(pattern):
    assert is_financial("the s&p 500 rallied", pattern) is True


def test_matched_keywords_are_normalized(pattern):
    assert matched_keywords("CEO buys STOCK", pattern) == ["ceo", "stock"]


def test_empty_keyword_list_raises():
    with pytest.raises(ValueError):
        compile_keyword_pattern([])
    with pytest.raises(ValueError):
        compile_keyword_pattern(["  "])


def test_filter_financial_selects_expected_rows(synthetic_manifest):
    subset = filter_financial(synthetic_manifest, DEFAULT_FINANCIAL_KEYWORDS)
    selected = set(subset["sample_id"])
    # 금융 문구를 담은 합성 샘플 (conftest.SYNTHETIC_TITLES 참조)
    assert {"s000", "s002", "s004", "s006", "s008"} <= selected
    # 'restocking'(s003)은 부분 문자열이므로 제외되어야 한다
    assert "s003" not in selected
    assert "s001" not in selected


def test_subset_preserves_manifest_columns(synthetic_manifest):
    subset = filter_financial(synthetic_manifest, ["stock", "ceo"])
    for column in synthetic_manifest.columns:
        assert column in subset.columns
    assert "matched_keywords" in subset.columns


def test_financial_report_has_coverage_and_counts(synthetic_manifest):
    subset = filter_financial(synthetic_manifest, DEFAULT_FINANCIAL_KEYWORDS)
    report = financial_report(subset, synthetic_manifest, name="all")
    assert report["n_samples"] == len(subset)
    assert report["n_full"] == len(synthetic_manifest)
    assert 0.0 < report["coverage_ratio"] <= 1.0
    assert sum(report["keyword_counts"].values()) >= len(subset)


def test_keyword_counts_sorted_desc(synthetic_manifest):
    subset = filter_financial(synthetic_manifest, DEFAULT_FINANCIAL_KEYWORDS)
    values = list(keyword_counts(subset).values())
    assert values == sorted(values, reverse=True)


def test_load_keywords_falls_back_to_default():
    assert load_keywords(None) == list(DEFAULT_FINANCIAL_KEYWORDS)
    assert load_keywords([]) == list(DEFAULT_FINANCIAL_KEYWORDS)
    assert load_keywords(["stock"]) == ["stock"]
