"""Ablation table 취합 (Story 1.4 AC4, AC5 / FR10).

Epic 1이 만드는 3개 행(BERT only / ResNet only / BERT+Image late fusion)의
지표를 **동일 test set 기준**으로 한 표에 모은다. Epic 2~5에서 행이 추가될 때는
`AblationRow`를 덧붙이기만 하면 되도록 데이터 주도로 작성했다.

입력: `outputs/<exp_name>/metrics_<split>.json` (없으면 `metrics.json`)
출력: `outputs/ablation_table.csv` + `docs/bmad/results/ablation.md`
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from src.evaluation.metrics import METRIC_KEYS


@dataclass(frozen=True)
class AblationRow:
    """ablation table의 한 행 = (표시 이름, 실험 이름)."""

    model: str
    exp_name: str


#: Epic 1 ablation (A1~A3) — Epic 2 이후 행은 이 리스트에 추가한다.
EPIC1_ROWS: tuple[AblationRow, ...] = (
    AblationRow("BERT only", "bert_only"),
    AblationRow("ResNet only", "resnet_only"),
    AblationRow("BERT+Image (late fusion)", "late_fusion"),
)

COLUMNS: tuple[str, ...] = ("model", "exp_name", "split", *METRIC_KEYS, "status")


def load_metrics(output_dir: str | Path, exp_name: str, split: str = "test") -> dict[str, Any] | None:
    """`outputs/<exp_name>/metrics_<split>.json` -> dict. 없으면 None.

    지표 키는 Trainer가 `<split>_f1`처럼 prefix를 붙여 저장하므로 여기서 벗겨낸다.
    """
    exp_dir = Path(output_dir) / str(exp_name)
    for filename in (f"metrics_{split}.json", "metrics.json"):
        path = exp_dir / filename
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8") as fp:
            payload = json.load(fp)
        result: dict[str, Any] = {}
        for key in METRIC_KEYS:
            if f"{split}_{key}" in payload:
                result[key] = payload[f"{split}_{key}"]
            elif key in payload:
                result[key] = payload[key]
        result["source"] = str(path)
        result["split"] = str(payload.get("split", split))
        return result
    return None


def collect_ablation(
    output_dir: str | Path = "outputs",
    rows: Sequence[AblationRow] = EPIC1_ROWS,
    split: str = "test",
) -> pd.DataFrame:
    """실험별 metrics.json을 읽어 ablation DataFrame을 만든다.

    아직 학습하지 않은 실험은 지표 NaN + ``status="missing"``으로 남겨
    "무엇이 아직 비어 있는지"가 표에서 바로 보이게 한다.
    """
    records: list[dict[str, Any]] = []
    for row in rows:
        metrics = load_metrics(output_dir, row.exp_name, split=split)
        record: dict[str, Any] = {
            "model": row.model,
            "exp_name": row.exp_name,
            "split": split,
            "status": "ok" if metrics else "missing",
        }
        for key in METRIC_KEYS:
            record[key] = float(metrics[key]) if metrics and key in metrics else float("nan")
        records.append(record)
    return pd.DataFrame(records, columns=list(COLUMNS))


def f1_comparison(table: pd.DataFrame, fusion_exp: str = "late_fusion") -> dict[str, Any]:
    """late fusion F1을 단일 모달 baseline 각각과 비교한다 (AC4).

    Returns:
        ``{"fusion_f1": .., "baselines": {exp: f1}, "deltas": {exp: Δ},
        "best_single_modal": exp|None, "improved": bool|None}``
    """
    frame = table.set_index("exp_name")
    fusion_f1 = float(frame.loc[fusion_exp, "f1"]) if fusion_exp in frame.index else float("nan")
    baselines = {
        exp: float(frame.loc[exp, "f1"])
        for exp in frame.index
        if exp != fusion_exp
    }
    available = {exp: value for exp, value in baselines.items() if value == value}  # NaN 제외
    best = max(available, key=available.get) if available else None
    improved: bool | None
    if fusion_f1 != fusion_f1 or best is None:
        improved = None
    else:
        improved = fusion_f1 > available[best]
    return {
        "fusion_f1": fusion_f1,
        "baselines": baselines,
        "deltas": {exp: fusion_f1 - value for exp, value in available.items()},
        "best_single_modal": best,
        "improved": improved,
    }


def _fmt(value: float) -> str:
    return "—" if value != value else f"{value:.4f}"


def render_markdown(table: pd.DataFrame, split: str = "test", notes: str | None = None) -> str:
    """ablation.md 본문을 만든다 (AC5)."""
    lines = [
        "# Epic 1 Ablation Table (Phase 1 baseline)",
        "",
        f"> 자동 생성: `python scripts/collect_ablation.py` · 평가 split: **{split}**",
        "> 라벨 규약 0=REAL, 1=FAKE · Precision/Recall/F1은 FAKE(양성=1) 기준",
        "",
        "| Model | exp_name | " + " | ".join(k.capitalize() for k in METRIC_KEYS) + " | Status |",
        "|---|---|" + "---|" * (len(METRIC_KEYS) + 1),
    ]
    for _, row in table.iterrows():
        cells = " | ".join(_fmt(float(row[key])) for key in METRIC_KEYS)
        lines.append(f"| {row['model']} | `{row['exp_name']}` | {cells} | {row['status']} |")

    comparison = f1_comparison(table)
    lines += ["", "## F1 비교 (AC4)", ""]
    if comparison["improved"] is None:
        lines.append(
            "아직 비교할 수 없다 — late fusion 또는 단일 모달 baseline의 test 지표가 비어 있다. "
            "실데이터 학습 후 `scripts/collect_ablation.py`를 다시 실행할 것."
        )
    else:
        best = comparison["best_single_modal"]
        deltas = ", ".join(f"{exp} 대비 {value:+.4f}" for exp, value in comparison["deltas"].items())
        verdict = "개선됨" if comparison["improved"] else "개선되지 않음"
        lines.append(
            f"late fusion F1 = {comparison['fusion_f1']:.4f} ({deltas}). "
            f"최고 단일 모달은 `{best}`이며, late fusion은 이에 대해 **{verdict}**."
        )
    if notes:
        lines += ["", "## 비고", "", notes]
    return "\n".join(lines) + "\n"


def write_ablation(
    output_dir: str | Path = "outputs",
    rows: Sequence[AblationRow] = EPIC1_ROWS,
    split: str = "test",
    csv_path: str | Path | None = None,
    markdown_path: str | Path | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    """ablation table을 csv + markdown으로 기록한다 (AC5)."""
    table = collect_ablation(output_dir, rows=rows, split=split)
    csv_path = Path(csv_path) if csv_path else Path(output_dir) / "ablation_table.csv"
    markdown_path = (
        Path(markdown_path) if markdown_path else Path("docs/bmad/results/ablation.md")
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(csv_path, index=False)
    markdown_path.write_text(render_markdown(table, split=split, notes=notes), encoding="utf-8")
    return {
        "table": table,
        "csv": csv_path,
        "markdown": markdown_path,
        "comparison": f1_comparison(table),
    }
