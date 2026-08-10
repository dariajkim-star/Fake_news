"""실험 로거 — config의 `logging.backend`로 CSV / TensorBoard를 선택한다 (AC4)."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Mapping, Protocol

LOGGER = logging.getLogger("finfact")


def setup_console_logging(level: int = logging.INFO) -> logging.Logger:
    """콘솔 로깅을 한 번만 설정하고 프로젝트 공용 logger를 돌려준다."""
    if not LOGGER.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", "%H:%M:%S"))
        LOGGER.addHandler(handler)
    LOGGER.setLevel(level)
    return LOGGER


class ExperimentLogger(Protocol):
    """epoch 단위 스칼라 기록 인터페이스."""

    def log(self, step: int, metrics: Mapping[str, Any]) -> None: ...

    def close(self) -> None: ...


class CSVLogger:
    """metric을 `<out_dir>/history.csv` 한 파일에 누적한다.

    epoch마다 키가 늘어날 수 있으므로, 새 키가 등장하면 기존 행을 유지한 채
    헤더를 다시 쓴다(행 수가 epoch 수준이라 재작성 비용은 무시할 만하다).
    """

    def __init__(self, out_dir: str | Path, filename: str = "history.csv") -> None:
        self.path = Path(out_dir) / filename
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._rows: list[dict[str, Any]] = []
        self._fields: list[str] = ["step"]

    def log(self, step: int, metrics: Mapping[str, Any]) -> None:
        row: dict[str, Any] = {"step": step, **dict(metrics)}
        self._rows.append(row)
        for key in row:
            if key not in self._fields:
                self._fields.append(key)
        self._flush()

    def _flush(self) -> None:
        with self.path.open("w", encoding="utf-8", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=self._fields)
            writer.writeheader()
            writer.writerows(self._rows)

    def close(self) -> None:
        self._flush()


class TensorBoardLogger:
    """TensorBoard SummaryWriter 래퍼 (tensorboard 미설치 시 import 시점에 실패)."""

    def __init__(self, out_dir: str | Path) -> None:
        from torch.utils.tensorboard import SummaryWriter

        self._writer = SummaryWriter(log_dir=str(Path(out_dir) / "tb"))

    def log(self, step: int, metrics: Mapping[str, Any]) -> None:
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self._writer.add_scalar(key, value, step)

    def close(self) -> None:
        self._writer.flush()
        self._writer.close()


def build_logger(out_dir: str | Path, backend: str = "csv") -> ExperimentLogger:
    """config 값으로 로거를 만든다. backend: `csv` | `tensorboard`."""
    backend = (backend or "csv").lower()
    if backend == "csv":
        return CSVLogger(out_dir)
    if backend in {"tensorboard", "tb"}:
        return TensorBoardLogger(out_dir)
    raise ValueError(f"지원하지 않는 logging backend: {backend!r} (csv | tensorboard)")
