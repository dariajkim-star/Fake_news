"""Fakeddit PyTorch Dataset + collate + 레지스트리 등록 (Story 1.2 AC5, AC6).

Trainer 계약(Story 1.1) 유지:
- ``__getitem__``/collate 결과는 **dict**이며 정답 라벨 키는 ``label``.
- 모델은 ``model(batch) -> logits [B, 2]``만 만족하면 된다.

모달 모드(config ``data.modality``):
- ``text``  : ``input_ids``/``attention_mask`` (또는 tokenizer 미주입 시 ``text``)
- ``image`` : ``image`` [3, H, W]
- ``both``  : 둘 다 (Story 1.4 late fusion)

tokenizer는 config로 주입하되(HuggingFace), **네트워크 없이도 동작해야 하므로**
``data.tokenizer.name: null``이면 tokenizer 없이 raw text를 그대로 전달한다.
테스트에서는 가짜 tokenizer를 생성자 인자로 주입한다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.preprocess import read_manifest
from src.data.registry import register_dataset
from src.utils.config import Config

MODALITIES: tuple[str, ...] = ("text", "image", "both")

#: ImageNet 통계 — torchvision pretrained ResNet(Story 1.3)과 정합
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)


class FakedditDataError(ValueError):
    """Dataset 설정/데이터가 유효하지 않을 때."""


def build_image_transform(
    image_size: int = 224,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
    train: bool = False,
) -> Callable[[Any], torch.Tensor]:
    """resize/normalize transform. train=True면 가벼운 augmentation을 추가한다."""
    from torchvision import transforms

    steps: list[Any] = [transforms.Resize((int(image_size), int(image_size)))]
    if train:
        steps.append(transforms.RandomHorizontalFlip(p=0.5))
    steps += [transforms.ToTensor(), transforms.Normalize(mean=list(mean), std=list(std))]
    return transforms.Compose(steps)


def build_tokenizer(cfg: Config) -> Any | None:
    """config ``data.tokenizer.name``으로 HuggingFace tokenizer를 만든다.

    이름이 없으면 ``None``을 반환해 tokenizer 없는 경로로 동작한다
    (오프라인/CI 환경에서 네트워크 의존을 제거하기 위함).
    """
    name = cfg.get("data.tokenizer.name", None)
    if not name:
        return None
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        str(name),
        local_files_only=bool(cfg.get("data.tokenizer.local_files_only", False)),
    )


def subsample_manifest(
    manifest: pd.DataFrame,
    n: int | None,
    seed: int = 42,
    stratified: bool = True,
) -> pd.DataFrame:
    """단일 GPU 제약(NFR1) 대응 — seed 고정 subsampling. n이 None/과대면 원본 반환."""
    if n is None or int(n) <= 0 or int(n) >= len(manifest):
        return manifest.reset_index(drop=True)
    n = int(n)
    ordered = manifest.sort_values("sample_id", kind="mergesort").reset_index(drop=True)
    if not stratified:
        rng = np.random.default_rng(seed)
        picked = ordered.iloc[np.sort(rng.choice(len(ordered), size=n, replace=False))]
        return picked.reset_index(drop=True)

    frames: list[pd.DataFrame] = []
    labels = sorted(ordered["label"].unique())
    remaining = n
    for index, label in enumerate(labels):
        group = ordered[ordered["label"] == label]
        quota = (
            remaining
            if index == len(labels) - 1
            else min(len(group), int(round(n * len(group) / len(ordered))))
        )
        quota = max(0, min(quota, len(group), remaining))
        rng = np.random.default_rng(int(seed) + int(label))
        picked = group.iloc[np.sort(rng.permutation(len(group))[:quota])]
        frames.append(picked)
        remaining -= quota
    combined = pd.concat(frames, ignore_index=True) if frames else ordered.iloc[0:0]
    return combined.sort_values("sample_id", kind="mergesort").reset_index(drop=True)


class FakedditDataset(Dataset):
    """클린 manifest 기반 Fakeddit Dataset.

    Args:
        manifest: manifest DataFrame 또는 csv 경로.
        modality: ``text`` | ``image`` | ``both``.
        image_root: manifest의 상대 image_path에 붙일 루트.
        transform: 이미지 transform (미지정 시 image 모드에서 기본 transform 생성).
        tokenizer: HuggingFace tokenizer 호환 객체. None이면 raw text를 전달.
        max_length / truncation: tokenizer 인자.
        subsample: 사용할 최대 샘플 수 (NFR1).
    """

    def __init__(
        self,
        manifest: pd.DataFrame | str | Path,
        modality: str = "both",
        image_root: str | Path | None = None,
        transform: Callable[[Any], torch.Tensor] | None = None,
        tokenizer: Any | None = None,
        max_length: int = 128,
        truncation: bool = True,
        subsample: int | None = None,
        seed: int = 42,
        image_size: int = 224,
        train: bool = False,
    ) -> None:
        modality = str(modality).lower()
        if modality not in MODALITIES:
            raise FakedditDataError(f"modality는 {MODALITIES} 중 하나여야 합니다 (받은 값: {modality!r})")

        frame = read_manifest(manifest) if isinstance(manifest, (str, Path)) else manifest.copy()
        self.manifest = subsample_manifest(frame, subsample, seed=seed)
        self.modality = modality
        self.image_root = Path(image_root) if image_root else None
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.truncation = bool(truncation)

        self.needs_image = modality in ("image", "both")
        self.needs_text = modality in ("text", "both")
        # DataLoader가 자동으로 집어갈 수 있도록 dataset이 자기 collate를 들고 있는다.
        self.collate_fn = make_collate_fn(tokenizer)

        self.transform = transform
        if self.needs_image and self.transform is None:
            self.transform = build_image_transform(image_size=image_size, train=train)
        if self.needs_image:
            missing = int((self.manifest["image_path"].astype(str).str.len() == 0).sum())
            if missing:
                raise FakedditDataError(
                    f"modality={modality}인데 image_path가 빈 샘플이 {missing}건 있습니다. "
                    "preprocess를 require_image=True로 다시 실행하세요."
                )

    # -- 내부 헬퍼 --------------------------------------------------------
    def _resolve_image_path(self, relative: str) -> Path:
        path = Path(relative)
        if self.image_root and not path.is_absolute():
            path = self.image_root / path
        return path

    def _load_image(self, relative: str) -> torch.Tensor:
        from PIL import Image

        path = self._resolve_image_path(relative)
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            assert self.transform is not None
            return self.transform(rgb)

    def _encode_text(self, text: str) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=self.truncation,
            padding=False,
            return_tensors=None,
        )
        item: dict[str, torch.Tensor] = {}
        for key in ("input_ids", "attention_mask", "token_type_ids"):
            if key in encoded:
                item[key] = torch.as_tensor(encoded[key], dtype=torch.long)
        if "attention_mask" not in item and "input_ids" in item:
            item["attention_mask"] = torch.ones_like(item["input_ids"])
        return item

    # -- Dataset 프로토콜 --------------------------------------------------
    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.manifest.iloc[int(idx)]
        item: dict[str, Any] = {
            "sample_id": str(row["sample_id"]),
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
        }
        if self.needs_text:
            text = str(row["text"] or row["title"] or "")
            item["text"] = text
            if self.tokenizer is not None:
                item.update(self._encode_text(text))
        if self.needs_image:
            item["image"] = self._load_image(str(row["image_path"]))
        return item


def collate_fn(batch: Sequence[Mapping[str, Any]], pad_token_id: int = 0) -> dict[str, Any]:
    """가변 길이 토큰 시퀀스를 padding하며 배치를 조립한다.

    ``label``은 항상 LongTensor [B], 문자열 필드는 list로 유지된다.
    """
    if not batch:
        return {}
    collated: dict[str, Any] = {}
    keys = list(batch[0].keys())

    pad_values = {"input_ids": pad_token_id, "attention_mask": 0, "token_type_ids": 0}
    for key in keys:
        values = [item[key] for item in batch]
        if key in pad_values:
            max_len = max(int(value.shape[0]) for value in values)
            padded = torch.full((len(values), max_len), pad_values[key], dtype=torch.long)
            for row, value in enumerate(values):
                padded[row, : value.shape[0]] = value
            collated[key] = padded
        elif isinstance(values[0], torch.Tensor):
            collated[key] = torch.stack(values)
        else:
            collated[key] = list(values)
    return collated


def make_collate_fn(tokenizer: Any | None = None) -> Callable[[Sequence[Mapping[str, Any]]], dict[str, Any]]:
    """tokenizer의 pad_token_id를 반영한 collate 함수를 만든다."""
    pad_token_id = int(getattr(tokenizer, "pad_token_id", 0) or 0)
    return lambda batch: collate_fn(batch, pad_token_id=pad_token_id)


def resolve_split_manifest_path(cfg: Config, split: str) -> Path:
    """config에서 split manifest 경로를 결정한다.

    우선순위: ``data.splits.<split>`` > ``<data.manifest_dir>/<split>.csv``
    """
    explicit = cfg.get(f"data.splits.{split}", None)
    if explicit:
        return Path(str(explicit))
    manifest_dir = cfg.get("data.manifest_dir", None)
    if not manifest_dir:
        root = cfg.get("data.root", "data/fakeddit")
        manifest_dir = Path(str(root)) / "processed"
    return Path(str(manifest_dir)) / f"{split}.csv"


def build_fakeddit_dataset(cfg: Config, split: str, tokenizer: Any | None = None) -> FakedditDataset:
    """config + split -> FakedditDataset (레지스트리와 스크립트가 공유)."""
    subsample = cfg.get(f"data.subsample_per_split.{split}", None)
    if subsample is None:
        subsample = cfg.get("data.subsample", None) if split == "train" else None

    return FakedditDataset(
        manifest=resolve_split_manifest_path(cfg, split),
        modality=str(cfg.get("data.modality", "both")),
        image_root=cfg.get("data.image_root", None),
        tokenizer=tokenizer if tokenizer is not None else build_tokenizer(cfg),
        max_length=int(cfg.get("data.tokenizer.max_length", 128)),
        truncation=bool(cfg.get("data.tokenizer.truncation", True)),
        subsample=subsample,
        seed=int(cfg.get("seed", 42)),
        image_size=int(cfg.get("data.image_size", 224)),
        train=(split == "train"),
    )


@register_dataset("fakeddit")
def _build_fakeddit(cfg: Config, split: str) -> Dataset:
    return build_fakeddit_dataset(cfg, split)
