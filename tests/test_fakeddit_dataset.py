"""FakedditDataset / collate / 레지스트리 통합 테스트 — Story 1.2 AC5, AC6, AC7."""

from __future__ import annotations

import pytest
import torch

from src.data.fakeddit import (
    FakedditDataError,
    FakedditDataset,
    build_fakeddit_dataset,
    collate_fn,
    make_collate_fn,
    resolve_split_manifest_path,
    subsample_manifest,
)
from src.data.preprocess import write_manifest
from src.data.registry import build_dataloaders, build_dataset
from src.utils.config import Config


def _dataset(manifest, tokenizer=None, **kwargs) -> FakedditDataset:
    return FakedditDataset(manifest=manifest, tokenizer=tokenizer, image_size=32, **kwargs)


def test_both_modality_returns_image_text_and_label(synthetic_manifest, fake_tokenizer):
    dataset = _dataset(synthetic_manifest, fake_tokenizer, modality="both")
    item = dataset[0]
    assert item["image"].shape == (3, 32, 32)
    assert item["image"].dtype == torch.float32
    assert item["label"].dtype == torch.long
    assert int(item["label"]) in (0, 1)
    assert item["input_ids"].ndim == 1
    assert item["attention_mask"].shape == item["input_ids"].shape


def test_text_only_modality_has_no_image(synthetic_manifest, fake_tokenizer):
    item = _dataset(synthetic_manifest, fake_tokenizer, modality="text")[0]
    assert "image" not in item
    assert isinstance(item["text"], str) and item["text"]


def test_image_only_modality_has_no_text(synthetic_manifest, fake_tokenizer):
    item = _dataset(synthetic_manifest, fake_tokenizer, modality="image")[0]
    assert "image" in item
    assert "input_ids" not in item and "text" not in item


def test_without_tokenizer_raw_text_is_passed_through(synthetic_manifest):
    """네트워크 없는 환경에서도 Dataset이 동작해야 한다."""
    item = _dataset(synthetic_manifest, tokenizer=None, modality="text")[0]
    assert "input_ids" not in item
    assert isinstance(item["text"], str)


def test_invalid_modality_raises(synthetic_manifest):
    with pytest.raises(FakedditDataError):
        _dataset(synthetic_manifest, modality="audio")


def test_image_modality_requires_image_path(synthetic_manifest):
    broken = synthetic_manifest.copy()
    broken.loc[0, "image_path"] = ""
    with pytest.raises(FakedditDataError):
        _dataset(broken, modality="image")


def test_max_length_truncation(synthetic_manifest, fake_tokenizer):
    dataset = _dataset(synthetic_manifest, fake_tokenizer, modality="text", max_length=4)
    assert dataset[0]["input_ids"].shape[0] <= 4


def test_dataset_can_load_from_csv_path(synthetic_manifest, tmp_path, fake_tokenizer):
    path = write_manifest(synthetic_manifest, tmp_path / "train.csv")
    dataset = FakedditDataset(
        manifest=path, modality="text", tokenizer=fake_tokenizer, image_size=32
    )
    assert len(dataset) == len(synthetic_manifest)


# -- collate ---------------------------------------------------------------


def test_collate_pads_variable_length_sequences(synthetic_manifest, fake_tokenizer):
    dataset = _dataset(synthetic_manifest, fake_tokenizer, modality="both")
    batch = collate_fn([dataset[i] for i in range(4)])
    assert batch["label"].shape == (4,)
    assert batch["image"].shape == (4, 3, 32, 32)
    assert batch["input_ids"].shape[0] == 4
    assert batch["input_ids"].shape == batch["attention_mask"].shape
    # padding 위치의 attention_mask는 0
    lengths = [int(dataset[i]["input_ids"].shape[0]) for i in range(4)]
    for row, length in enumerate(lengths):
        assert batch["attention_mask"][row, :length].sum() == length
        assert batch["attention_mask"][row, length:].sum() == 0
    assert isinstance(batch["sample_id"], list)


def test_collate_uses_tokenizer_pad_id(synthetic_manifest, fake_tokenizer):
    dataset = _dataset(synthetic_manifest, fake_tokenizer, modality="text")
    batch = make_collate_fn(fake_tokenizer)([dataset[i] for i in range(3)])
    shortest = min(int(dataset[i]["input_ids"].shape[0]) for i in range(3))
    if batch["input_ids"].shape[1] > shortest:
        assert int(batch["input_ids"][:, -1].min()) >= 0


def test_collate_empty_batch():
    assert collate_fn([]) == {}


# -- subsampling (NFR1) -----------------------------------------------------


def test_subsample_limits_size_and_is_deterministic(synthetic_manifest):
    first = subsample_manifest(synthetic_manifest, 6, seed=42)
    second = subsample_manifest(synthetic_manifest, 6, seed=42)
    assert len(first) == 6
    assert first["sample_id"].tolist() == second["sample_id"].tolist()


def test_subsample_keeps_label_balance(synthetic_manifest):
    subset = subsample_manifest(synthetic_manifest, 6, seed=42)
    assert set(subset["label"]) == {0, 1}


def test_subsample_noop_when_none_or_too_large(synthetic_manifest):
    assert len(subsample_manifest(synthetic_manifest, None)) == len(synthetic_manifest)
    assert len(subsample_manifest(synthetic_manifest, 10_000)) == len(synthetic_manifest)


def test_dataset_subsample_option(synthetic_manifest, fake_tokenizer):
    dataset = _dataset(synthetic_manifest, fake_tokenizer, modality="text", subsample=4)
    assert len(dataset) == 4


# -- config / registry 통합 -------------------------------------------------


def _cfg(tmp_path, **data_overrides) -> Config:
    data = {
        "name": "fakeddit",
        "manifest_dir": str(tmp_path),
        "modality": "text",
        "batch_size": 4,
        "num_workers": 0,
        "image_size": 32,
        "tokenizer": {"name": None, "max_length": 16},
    }
    data.update(data_overrides)
    return Config({"exp_name": "t", "seed": 42, "data": data})


def test_resolve_split_manifest_path_prefers_explicit(tmp_path):
    cfg = Config({"data": {"splits": {"train": "a/b.csv"}, "manifest_dir": str(tmp_path)}})
    assert str(resolve_split_manifest_path(cfg, "train")).endswith("b.csv")
    assert resolve_split_manifest_path(cfg, "val").name == "val.csv"


def test_resolve_split_manifest_path_defaults_to_root():
    cfg = Config({"data": {"root": "data/fakeddit"}})
    assert resolve_split_manifest_path(cfg, "test").as_posix().endswith(
        "data/fakeddit/processed/test.csv"
    )


def test_registry_builds_fakeddit_from_config(synthetic_manifest, tmp_path):
    write_manifest(synthetic_manifest, tmp_path / "train.csv")
    dataset = build_dataset(_cfg(tmp_path), "train")
    assert isinstance(dataset, FakedditDataset)
    assert len(dataset) == len(synthetic_manifest)


def test_build_fakeddit_dataset_injects_tokenizer(synthetic_manifest, tmp_path, fake_tokenizer):
    write_manifest(synthetic_manifest, tmp_path / "val.csv")
    dataset = build_fakeddit_dataset(_cfg(tmp_path), "val", tokenizer=fake_tokenizer)
    assert dataset[0]["input_ids"].ndim == 1


def test_dataloader_yields_trainer_contract_batch(synthetic_manifest, tmp_path):
    """Trainer 계약: batch는 dict이고 라벨 키는 `label`."""
    for split in ("train", "val"):
        write_manifest(synthetic_manifest, tmp_path / f"{split}.csv")
    loaders = build_dataloaders(_cfg(tmp_path, modality="both"), splits=("train", "val"))
    batch = next(iter(loaders["train"]))
    assert isinstance(batch, dict)
    assert batch["label"].shape[0] <= 4
    assert batch["image"].shape[1:] == (3, 32, 32)


def test_dummy_dataset_registration_still_works():
    """Story 1.1의 dummy 등록을 깨지 않았는지 회귀 확인."""
    cfg = Config({"exp_name": "t", "seed": 42, "data": {"name": "dummy", "batch_size": 4}})
    dataset = build_dataset(cfg, "train")
    assert "label" in dataset[0]


def test_unknown_dataset_name_raises():
    cfg = Config({"exp_name": "t", "seed": 42, "data": {"name": "nope"}})
    with pytest.raises(KeyError):
        build_dataset(cfg, "train")
