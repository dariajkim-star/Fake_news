"""config 로딩 / 검증 / 오버라이드 테스트 (Story 1.1 AC2, AC7)."""

from __future__ import annotations

import pytest
import yaml

from src.utils.config import Config, ConfigError, load_config

REPO_CONFIG = "configs/base.yaml"


def _write(tmp_path, name: str, payload: dict) -> str:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(payload, allow_unicode=True), encoding="utf-8")
    return str(path)


def test_repo_base_config_loads():
    cfg = load_config(REPO_CONFIG)
    assert cfg.exp_name == "base"
    assert cfg.seed == 42
    assert cfg.get("train.monitor") == "f1"


def test_nested_access_and_dot_path(tmp_path):
    path = _write(tmp_path, "c.yaml", {"exp_name": "e", "seed": 1, "train": {"lr": 0.5}})
    cfg = load_config(path)
    assert isinstance(cfg.train, Config)
    assert cfg.train.lr == 0.5
    assert cfg["train"]["lr"] == 0.5
    assert cfg.get("train.lr") == 0.5
    assert cfg.get("train.missing", "fallback") == "fallback"


def test_missing_required_key_raises(tmp_path):
    path = _write(tmp_path, "bad.yaml", {"seed": 1})  # exp_name 누락
    with pytest.raises(ConfigError, match="exp_name"):
        load_config(path)


def test_missing_file_raises(tmp_path):
    with pytest.raises(ConfigError, match="찾을 수 없습니다"):
        load_config(tmp_path / "nope.yaml")


def test_cli_overrides_are_typed(tmp_path):
    path = _write(tmp_path, "c.yaml", {"exp_name": "e", "seed": 1, "train": {"epochs": 10}})
    cfg = load_config(path, overrides=["train.epochs=1", "train.amp=true", "exp_name=smoke"])
    assert cfg.train.epochs == 1  # 문자열이 아니라 int로 파싱
    assert cfg.train.amp is True
    assert cfg.exp_name == "smoke"


def test_malformed_override_raises(tmp_path):
    path = _write(tmp_path, "c.yaml", {"exp_name": "e", "seed": 1})
    with pytest.raises(ConfigError, match="key=value"):
        load_config(path, overrides=["train.epochs"])


def test_base_inheritance_deep_merges(tmp_path):
    _write(tmp_path, "parent.yaml", {"exp_name": "p", "seed": 1, "train": {"lr": 0.1, "epochs": 9}})
    child = _write(tmp_path, "child.yaml", {"_base_": "parent.yaml", "exp_name": "c", "train": {"lr": 0.2}})
    cfg = load_config(child)
    assert cfg.exp_name == "c"
    assert cfg.train.lr == 0.2
    assert cfg.train.epochs == 9  # 부모 값 유지


def test_save_roundtrip(tmp_path):
    path = _write(tmp_path, "c.yaml", {"exp_name": "e", "seed": 1, "train": {"lr": 0.5}})
    cfg = load_config(path)
    saved = cfg.save(tmp_path / "out" / "config.yaml")
    assert load_config(saved).to_dict() == cfg.to_dict()
