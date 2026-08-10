"""Ablation table 취합 엔트리포인트 (Story 1.4 AC4, AC5 / FR10).

    python scripts/collect_ablation.py
    python scripts/collect_ablation.py --split test --output-dir outputs

`outputs/<exp_name>/metrics_test.json`을 읽어 `outputs/ablation_table.csv`와
`docs/bmad/results/ablation.md`를 생성한다. 아직 학습하지 않은 실험은
status=missing으로 표시되므로 "무엇이 비어 있는지"가 표에서 바로 보인다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.ablation import EPIC1_ROWS, AblationRow, write_ablation  # noqa: E402
from src.utils.logging import setup_console_logging  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Epic 1 ablation table 취합")
    parser.add_argument("--output-dir", default="outputs", help="실험 출력 루트 (기본: outputs)")
    parser.add_argument("--split", default="test", help="비교할 split (기본: test)")
    parser.add_argument("--csv", default=None, help="csv 출력 경로 (기본: <output-dir>/ablation_table.csv)")
    parser.add_argument(
        "--markdown", default=None, help="markdown 출력 경로 (기본: docs/bmad/results/ablation.md)"
    )
    parser.add_argument("--notes", default=None, help="표 아래에 덧붙일 분석 코멘트")
    parser.add_argument(
        "--row",
        dest="rows",
        nargs="*",
        default=[],
        metavar="LABEL=EXP_NAME",
        help="행 추가/교체 (기본 3행 외 Epic 2+ 실험을 넣을 때)",
    )
    return parser.parse_args(argv)


def _resolve_rows(raw: list[str]) -> tuple[AblationRow, ...]:
    if not raw:
        return EPIC1_ROWS
    rows: list[AblationRow] = []
    for item in raw:
        if "=" not in item:
            raise SystemExit(f"--row는 LABEL=EXP_NAME 형식이어야 합니다: {item!r}")
        label, exp_name = item.split("=", 1)
        rows.append(AblationRow(label.strip(), exp_name.strip()))
    return tuple(rows)


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    logger = setup_console_logging()

    result = write_ablation(
        output_dir=args.output_dir,
        rows=_resolve_rows(args.rows),
        split=args.split,
        csv_path=args.csv,
        markdown_path=args.markdown,
        notes=args.notes,
    )
    logger.info("ablation table 기록 | %s | %s", result["csv"], result["markdown"])
    missing = result["table"].loc[result["table"]["status"] == "missing", "exp_name"].tolist()
    if missing:
        logger.warning("아직 test 지표가 없는 실험: %s", ", ".join(missing))
    print(result["table"].to_string(index=False))
    return result


if __name__ == "__main__":
    main()
