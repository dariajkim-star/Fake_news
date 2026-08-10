"""YAML 기반 실험 config 시스템.

실험 하나 = config 파일 하나 원칙(NFR3). 모든 엔트리포인트는 `--config`로
설정을 받고, 실행 시 사용한 config 사본을 `outputs/<exp_name>/config.yaml`에
남겨 결과와 설정을 항상 함께 추적할 수 있게 한다.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterator, Mapping

import yaml

# config 최상위에 반드시 존재해야 하는 키 (누락 시 즉시 실패)
REQUIRED_KEYS: tuple[str, ...] = ("exp_name", "seed")


class ConfigError(ValueError):
    """config 파일이 유효하지 않을 때 발생."""


class Config(Mapping):
    """중첩 dict를 감싼 읽기 전용 config 객체.

    `cfg.train.lr` 같은 속성 접근과 `cfg["train"]["lr"]` 매핑 접근을 모두 지원하고,
    `cfg.get("train.lr", 1e-4)`처럼 점 경로 조회도 가능하다.
    """

    def __init__(self, data: Mapping[str, Any] | None = None) -> None:
        self._data: dict[str, Any] = {}
        for key, value in dict(data or {}).items():
            self._data[key] = Config(value) if isinstance(value, Mapping) else value

    # -- Mapping 프로토콜 -------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __getattr__(self, key: str) -> Any:
        try:
            return self._data[key]
        except KeyError as exc:  # pragma: no cover - 속성 접근 실패 경로
            raise AttributeError(f"config에 '{key}' 키가 없습니다") from exc

    def __repr__(self) -> str:  # pragma: no cover - 디버깅용
        return f"Config({self.to_dict()!r})"

    # -- 편의 메서드 ------------------------------------------------------
    def get(self, path: str, default: Any = None) -> Any:
        """점(.)으로 구분된 경로로 값을 조회한다. 없으면 default."""
        node: Any = self
        for part in path.split("."):
            if isinstance(node, Config) and part in node:
                node = node[part]
            else:
                return default
        return node

    def to_dict(self) -> dict[str, Any]:
        return {
            key: value.to_dict() if isinstance(value, Config) else copy.deepcopy(value)
            for key, value in self._data.items()
        }

    def save(self, path: str | Path) -> Path:
        """config 사본을 yaml로 저장한다 (실험 재현용)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fp:
            yaml.safe_dump(self.to_dict(), fp, allow_unicode=True, sort_keys=False)
        return path


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """override를 base 위에 재귀적으로 덮어쓴 새 dict를 반환한다."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _parse_scalar(raw: str) -> Any:
    """CLI 오버라이드 문자열을 yaml 규칙으로 파싱한다 ("3" -> 3, "true" -> True)."""
    return yaml.safe_load(raw)


def _apply_overrides(data: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """`--set train.lr=1e-4` 형태의 오버라이드를 적용한다."""
    for item in overrides:
        if "=" not in item:
            raise ConfigError(f"오버라이드는 key=value 형식이어야 합니다: {item!r}")
        path, raw = item.split("=", 1)
        node = data
        parts = path.strip().split(".")
        for part in parts[:-1]:
            child = node.get(part)
            if not isinstance(child, dict):
                child = {}
                node[part] = child
            node = child
        node[parts[-1]] = _parse_scalar(raw.strip())
    return data


def load_config(
    path: str | Path,
    overrides: list[str] | None = None,
    required_keys: tuple[str, ...] = REQUIRED_KEYS,
) -> Config:
    """yaml config를 읽어 검증된 `Config`를 반환한다.

    `_base_` 키로 다른 config를 상속할 수 있으며(상대 경로 허용), 이후
    `overrides`가 마지막에 적용된다.
    """
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config 파일을 찾을 수 없습니다: {path}")

    with path.open("r", encoding="utf-8") as fp:
        loaded = yaml.safe_load(fp)
    if loaded is None:
        loaded = {}
    if not isinstance(loaded, dict):
        raise ConfigError(f"config 최상위는 mapping이어야 합니다: {path}")

    base_ref = loaded.pop("_base_", None)
    data: dict[str, Any] = {}
    if base_ref is not None:
        base_path = Path(base_ref)
        if not base_path.is_absolute():
            base_path = (path.parent / base_path).resolve()
        data = load_config(base_path, required_keys=()).to_dict()
    data = _deep_merge(data, loaded)
    data = _apply_overrides(data, overrides or [])

    missing = [key for key in required_keys if key not in data]
    if missing:
        raise ConfigError(f"config에 필수 키가 없습니다: {', '.join(missing)} ({path})")

    return Config(data)
