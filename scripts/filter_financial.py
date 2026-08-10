"""금융 키워드 subset manifest 생성 (Story 1.2 AC4 / FR11).

    python scripts/filter_financial.py --config configs/fakeddit.yaml
    python scripts/filter_financial.py --config configs/fakeddit.yaml --split test

키워드는 config `data.financial_keywords`에서 읽고, 없으면
`src/data/financial.py`의 `DEFAULT_FINANCIAL_KEYWORDS`를 쓴다.

산출물: `<manifest_dir>/financial/<split>.csv` + `financial_report.json`
subset 규모 부족이 프로젝트 주요 리스크이므로 통계를 항상 출력한다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.fakeddit import resolve_split_manifest_path  # noqa: E402
from src.data.financial import (  # noqa: E402
    filter_financial,
    financial_report,
    load_keywords,
    write_financial_report,
)
from src.data.preprocess import format_report, read_manifest, write_manifest  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.logging import setup_console_logging  # noqa: E402

LOGGER = setup_console_logging()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="금융 키워드 subset 필터링")
    parser.add_argument("--config", required=True, help="실험 config yaml")
    parser.add_argument(
        "--set", dest="overrides", nargs="*", default=[], metavar="KEY=VALUE", help="config 오버라이드"
    )
    parser.add_argument(
        "--split",
        nargs="*",
        default=["train", "val", "test"],
        help="필터링할 분할 (기본: train val test)",
    )
    parser.add_argument("--out-dir", default=None, help="subset 출력 디렉토리")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, object]:
    args = parse_args(argv)
    cfg = load_config(args.config, overrides=args.overrides)
    keywords = load_keywords(cfg.get("data.financial_keywords", None))
    LOGGER.info("금융 키워드 %d개 사용", len(keywords))

    first_path = resolve_split_manifest_path(cfg, args.split[0])
    out_dir = Path(args.out_dir) if args.out_dir else first_path.parent / "financial"
    out_dir.mkdir(parents=True, exist_ok=True)

    reports: list[dict[str, object]] = []
    for split in args.split:
        manifest = read_manifest(resolve_split_manifest_path(cfg, split))
        subset = filter_financial(manifest, keywords)
        write_manifest(subset, out_dir / f"{split}.csv")
        # matched_keywords는 진단용이라 표준 manifest 스키마에서 빠지므로 별도 저장
        subset.to_csv(out_dir / f"{split}_matched.csv", index=False, encoding="utf-8")
        report = financial_report(subset, manifest, name=split)
        reports.append(report)
        LOGGER.info("[%s] subset %d/%d건", split, len(subset), len(manifest))

    print(format_report(reports))
    total = sum(int(report["n_samples"]) for report in reports)  # type: ignore[arg-type]
    if total < 1000:
        LOGGER.warning(
            "금융 subset이 %d건으로 작습니다 - 키워드 확장 또는 subset 단독 학습 축소를 검토하세요 "
            "(prd.md#Checklist 규모 부족 리스크).",
            total,
        )

    summary = {"keywords": keywords, "splits": reports, "total": total}
    write_financial_report(summary, out_dir / "financial_report.json")
    return summary


if __name__ == "__main__":
    main()
