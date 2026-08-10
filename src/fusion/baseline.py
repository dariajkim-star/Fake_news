"""Phase 1 baseline 분류기 — BERT only / ResNet only (Story 1.3) + late fusion (Story 1.4).

ablation table(FR10)의 세 행(BERT only / ResNet only / BERT+Image)을 만드는 모델들이다.
Story 1.4의 late fusion은 Story 1.3에서 정의한 인코더(`src/fusion/encoders.py`)를
**그대로 재사용**하고, 분류 head만 concat feature용으로 바꿔 끼운다.

Trainer 계약(Story 1.1):
- ``forward(batch: dict) -> logits [B, 2]``
- 라벨 키는 ``label`` (모델은 라벨을 보지 않는다)
- ``@register_model("text_only" | "image_only" | "late_fusion")``로 config ``model.name`` 선택
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from src.fusion.encoders import (
    ClassificationHead,
    ImageEncoder,
    TextEncoder,
    _BaseEncoder,
    set_requires_grad,
)
from src.fusion.registry import register_model
from src.utils.config import Config


def _section(cfg: Config, path: str) -> dict[str, Any]:
    """config 하위 블록을 평범한 dict로 꺼낸다 (없으면 빈 dict)."""
    node = cfg.get(path, None)
    if node is None:
        return {}
    if isinstance(node, Config):
        return node.to_dict()
    return dict(node)


class SingleModalClassifier(nn.Module):
    """인코더 1개 + classification head — BERT only / ResNet only 공통 구현.

    Args:
        encoder: `forward(batch) -> [B, D]`와 `output_dim`을 만족하는 인코더.
        num_classes: 출력 클래스 수 (기본 2 — Fake/Real).
        head_hidden_dims: head MLP 은닉 차원. None이면 단일 Linear.
        head_dropout: head dropout 확률.
    """

    def __init__(
        self,
        encoder: _BaseEncoder | nn.Module,
        num_classes: int = 2,
        head_hidden_dims: Any = None,
        head_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        feature_dim = int(getattr(encoder, "output_dim"))
        self.encoder = encoder
        self.head = ClassificationHead(
            input_dim=feature_dim,
            num_classes=int(num_classes),
            hidden_dims=head_hidden_dims,
            dropout=float(head_dropout),
        )
        self.feature_dim = feature_dim

    # -- 공개 인터페이스 (AC5) --------------------------------------------
    def encode(self, batch: Mapping[str, Any]) -> torch.Tensor:
        """penultimate feature vector [B, feature_dim]를 반환한다.

        Story 1.4의 late fusion / Epic 2 fusion이 이 메서드로 feature를 얻는다.
        """
        return self.encoder(batch)

    def forward(self, batch: Mapping[str, Any]) -> torch.Tensor:
        return self.head(self.encode(batch))

    @torch.no_grad()
    def predict_proba(self, batch: Mapping[str, Any]) -> torch.Tensor:
        """fake(=1) 확률 [B]를 반환한다 (내부 라벨 규약 0=REAL, 1=FAKE)."""
        was_training = self.training
        self.eval()
        probs = torch.softmax(self.forward(batch).float(), dim=-1)[:, 1]
        if was_training:
            self.train()
        return probs

    # -- 학습 편의 --------------------------------------------------------
    def param_groups(self, encoder_lr: float, head_lr: float | None = None) -> list[dict[str, Any]]:
        """인코더/head lr을 다르게 주기 위한 optimizer param group.

        frozen 인코더의 파라미터는 제외한다 (requires_grad=False).
        """
        head_lr = encoder_lr if head_lr is None else head_lr
        encoder_params = [p for p in self.encoder.parameters() if p.requires_grad]
        groups: list[dict[str, Any]] = []
        if encoder_params:
            groups.append({"params": encoder_params, "lr": float(encoder_lr)})
        groups.append({"params": list(self.head.parameters()), "lr": float(head_lr)})
        return groups

    def save_encoder(self, path: str | Path) -> Path:
        """인코더 가중치만 저장한다 (Story 1.4가 재사용)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"encoder_state": self.encoder.state_dict()}, path)
        return path

    def load_encoder(self, path: str | Path, strict: bool = True) -> "SingleModalClassifier":
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        state = payload.get("encoder_state", payload)
        self.encoder.load_state_dict(state, strict=strict)
        return self


# ---------------------------------------------------------------------------
# config -> 인코더 빌더
# ---------------------------------------------------------------------------
def build_text_encoder(cfg: Config) -> TextEncoder:
    """config ``model.text.*`` -> TextEncoder."""
    section = _section(cfg, "model.text")
    return TextEncoder(
        model_name=str(section.get("model_name", "bert-base-uncased")),
        pretrained=bool(section.get("pretrained", True)),
        pooling=str(section.get("pooling", "cls")),
        freeze=bool(section.get("freeze", False)),
        proj_dim=section.get("proj_dim", None),
        dropout=float(section.get("dropout", 0.0)),
        local_files_only=bool(section.get("local_files_only", False)),
        config_overrides=section.get("encoder_config", None),
    )


def build_image_encoder(cfg: Config) -> ImageEncoder:
    """config ``model.image.*`` -> ImageEncoder."""
    section = _section(cfg, "model.image")
    return ImageEncoder(
        arch=str(section.get("arch", "resnet50")),
        pretrained=bool(section.get("pretrained", True)),
        freeze=bool(section.get("freeze", False)),
        proj_dim=section.get("proj_dim", None),
        dropout=float(section.get("dropout", 0.0)),
    )


def _build_classifier(cfg: Config, encoder: nn.Module) -> SingleModalClassifier:
    head = _section(cfg, "model.head")
    return SingleModalClassifier(
        encoder=encoder,
        num_classes=int(cfg.get("model.num_classes", 2)),
        head_hidden_dims=head.get("hidden_dims", None),
        head_dropout=float(head.get("dropout", 0.1)),
    )


@register_model("text_only")
def build_text_only_model(cfg: Config) -> SingleModalClassifier:
    """BERT only baseline (AC1)."""
    return _build_classifier(cfg, build_text_encoder(cfg))


@register_model("image_only")
def build_image_only_model(cfg: Config) -> SingleModalClassifier:
    """ResNet only baseline (AC2)."""
    return _build_classifier(cfg, build_image_encoder(cfg))


# ---------------------------------------------------------------------------
# Story 1.4 — Late fusion (ResNet + BERT)
# ---------------------------------------------------------------------------
#: fusion head 기본 은닉 차원 (concat 2816 -> 512 -> 2)
DEFAULT_FUSION_HIDDEN_DIMS: tuple[int, ...] = (512,)


class LateFusionClassifier(nn.Module):
    """텍스트/이미지 feature를 concat해 분류하는 late fusion baseline (AC1).

    구조: ``[text_feature (768) ; image_feature (2048)]`` -> MLP head -> logits [B, 2].
    인코더는 Story 1.3의 ``TextEncoder``/``ImageEncoder``를 **그대로** 받는다
    (새로 만들지 않는다). 인코더는 `forward(batch) -> [B, D]`와 `output_dim`만
    만족하면 되므로, Epic 2의 region feature 인코더로 교체해도 head는 그대로 쓴다(NFR6).

    Args:
        text_encoder: `forward(batch) -> [B, D_t]`, `output_dim` 인코더.
        image_encoder: `forward(batch) -> [B, D_i]`, `output_dim` 인코더.
        num_classes: 출력 클래스 수 (기본 2 — Fake/Real).
        head_hidden_dims: fusion head MLP 은닉 차원. None이면 단일 Linear.
        head_dropout: head dropout 확률.
        freeze_encoders: True면 양쪽 인코더 backbone 고정 (AC2의 학습 모드 (a)).
    """

    #: ablation 비교(A1~A3)에서 이 모델이 차지하는 행 이름
    ablation_row: str = "BERT+Image (late fusion)"

    def __init__(
        self,
        text_encoder: _BaseEncoder | nn.Module,
        image_encoder: _BaseEncoder | nn.Module,
        num_classes: int = 2,
        head_hidden_dims: Any = DEFAULT_FUSION_HIDDEN_DIMS,
        head_dropout: float = 0.1,
        freeze_encoders: bool = False,
    ) -> None:
        super().__init__()
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.text_dim = int(getattr(text_encoder, "output_dim"))
        self.image_dim = int(getattr(image_encoder, "output_dim"))
        self.feature_dim = self.text_dim + self.image_dim  # 768 + 2048 = 2816
        self.head = ClassificationHead(
            input_dim=self.feature_dim,
            num_classes=int(num_classes),
            hidden_dims=head_hidden_dims,
            dropout=float(head_dropout),
        )
        if freeze_encoders:
            self.freeze_encoders()

    # -- feature 인터페이스 -------------------------------------------------
    def encode_modalities(self, batch: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        """모달별 feature를 따로 돌려준다 (feature 캐싱/분석용)."""
        return {
            "text": self.text_encoder(batch),
            "image": self.image_encoder(batch),
        }

    def encode(self, batch: Mapping[str, Any]) -> torch.Tensor:
        """concat feature [B, text_dim + image_dim]을 반환한다."""
        features = self.encode_modalities(batch)
        return torch.cat([features["text"], features["image"]], dim=-1)

    def forward(self, batch: Mapping[str, Any]) -> torch.Tensor:
        return self.head(self.encode(batch))

    @torch.no_grad()
    def predict_proba(self, batch: Mapping[str, Any]) -> torch.Tensor:
        """fake(=1) 확률 [B] (내부 라벨 규약 0=REAL, 1=FAKE)."""
        was_training = self.training
        self.eval()
        probs = torch.softmax(self.forward(batch).float(), dim=-1)[:, 1]
        if was_training:
            self.train()
        return probs

    def predict(self, image_path: Any = None, text: str | None = None, **kwargs) -> dict[str, Any]:
        """단일 샘플 추론 (AC6). 자세한 인자는 :func:`predict_single` 참조."""
        return predict_single(self, image_path=image_path, text=text, **kwargs)

    # -- 학습 모드 (AC2) ----------------------------------------------------
    def freeze_encoders(self) -> "LateFusionClassifier":
        """(a) 양쪽 인코더 frozen + fusion head만 학습."""
        for encoder in (self.text_encoder, self.image_encoder):
            if hasattr(encoder, "freeze"):
                encoder.freeze()
            else:  # 인코더 계약만 만족하는 임의 모듈
                set_requires_grad(encoder, False)
        return self

    def unfreeze_encoders(self) -> "LateFusionClassifier":
        """(b) end-to-end fine-tuning."""
        for encoder in (self.text_encoder, self.image_encoder):
            if hasattr(encoder, "unfreeze"):
                encoder.unfreeze()
            else:
                set_requires_grad(encoder, True)
        return self

    @property
    def encoders_frozen(self) -> bool:
        return not any(
            p.requires_grad
            for encoder in (self.text_encoder, self.image_encoder)
            for p in encoder.parameters()
        )

    def param_groups(self, encoder_lr: float, head_lr: float | None = None) -> list[dict[str, Any]]:
        """인코더/head lr 분리 (frozen 파라미터는 optimizer에서 제외)."""
        head_lr = encoder_lr if head_lr is None else head_lr
        encoder_params = [
            p
            for encoder in (self.text_encoder, self.image_encoder)
            for p in encoder.parameters()
            if p.requires_grad
        ]
        groups: list[dict[str, Any]] = []
        if encoder_params:
            groups.append({"params": encoder_params, "lr": float(encoder_lr)})
        groups.append({"params": list(self.head.parameters()), "lr": float(head_lr)})
        return groups

    # -- Story 1.3 인코더 checkpoint 재사용 ---------------------------------
    def save_encoders(self, text_path: str | Path | None = None, image_path: str | Path | None = None):
        """인코더 가중치를 `SingleModalClassifier.save_encoder`와 같은 형식으로 저장한다."""
        saved: dict[str, Path] = {}
        for key, path, encoder in (
            ("text", text_path, self.text_encoder),
            ("image", image_path, self.image_encoder),
        ):
            if path is None:
                continue
            target = Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"encoder_state": encoder.state_dict()}, target)
            saved[key] = target
        return saved

    def load_encoders(
        self,
        text_path: str | Path | None = None,
        image_path: str | Path | None = None,
        strict: bool = True,
    ) -> "LateFusionClassifier":
        """Story 1.3에서 학습한 인코더 checkpoint를 재사용한다 (Task: 인코더 재사용).

        ``SingleModalClassifier.save_encoder()``가 만든 파일(`{"encoder_state": ...}`)과
        raw state_dict 둘 다 받는다.
        """
        for path, encoder in ((text_path, self.text_encoder), (image_path, self.image_encoder)):
            if path is None:
                continue
            payload = torch.load(Path(path), map_location="cpu", weights_only=False)
            state = payload.get("encoder_state", payload) if isinstance(payload, Mapping) else payload
            encoder.load_state_dict(state, strict=strict)
        return self


# ---------------------------------------------------------------------------
# 단일 샘플 추론 (AC6)
# ---------------------------------------------------------------------------
def build_single_sample_batch(
    image_path: Any = None,
    text: str | None = None,
    tokenizer: Any | None = None,
    transform: Any | None = None,
    image_size: int = 224,
    max_length: int = 128,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """(이미지 경로, 텍스트) -> 배치 크기 1인 batch dict.

    학습 시 사용하는 `FakedditDataset`과 동일한 transform/tokenizer 규약을 따르므로
    학습-추론 전처리가 어긋나지 않는다. 텍스트/이미지 중 하나만 줘도 되며,
    tokenizer가 없으면 raw text만 담는다(오프라인 경로).
    """
    device = torch.device(device)
    batch: dict[str, Any] = {}

    if text is not None:
        batch["text"] = [str(text)]
        if tokenizer is not None:
            encoded = tokenizer(
                str(text), max_length=int(max_length), truncation=True, padding=False,
                return_tensors=None,
            )
            for key in ("input_ids", "attention_mask", "token_type_ids"):
                if key in encoded:
                    batch[key] = torch.as_tensor(encoded[key], dtype=torch.long).unsqueeze(0).to(device)
            if "attention_mask" not in batch and "input_ids" in batch:
                batch["attention_mask"] = torch.ones_like(batch["input_ids"])

    if image_path is not None:
        if isinstance(image_path, torch.Tensor):
            tensor = image_path if image_path.dim() == 4 else image_path.unsqueeze(0)
            batch["image"] = tensor.to(device)
        else:
            from PIL import Image

            from src.data.fakeddit import build_image_transform

            if transform is None:
                transform = build_image_transform(image_size=int(image_size), train=False)
            with Image.open(Path(str(image_path))) as image:
                batch["image"] = transform(image.convert("RGB")).unsqueeze(0).to(device)

    if not batch:
        raise ValueError("predict에는 image_path 또는 text 중 최소 하나가 필요합니다")
    return batch


@torch.no_grad()
def predict_single(
    model: nn.Module,
    image_path: Any = None,
    text: str | None = None,
    tokenizer: Any | None = None,
    transform: Any | None = None,
    image_size: int = 224,
    max_length: int = 128,
    device: torch.device | str = "cpu",
    threshold: float = 0.5,
) -> dict[str, Any]:
    """단일 (이미지, 텍스트) 샘플의 fake 확률을 반환한다 (AC6, NFR2 사전 검증).

    Returns:
        ``{"fake_prob": float(0~1), "label": 0|1, "label_name": "REAL"|"FAKE",
        "elapsed_sec": float}`` — `elapsed_sec`는 전처리+forward 전체 시간이다.
    """
    started = time.perf_counter()
    device = torch.device(device)
    model = model.to(device)
    was_training = model.training
    model.eval()

    batch = build_single_sample_batch(
        image_path=image_path,
        text=text,
        tokenizer=tokenizer,
        transform=transform,
        image_size=image_size,
        max_length=max_length,
        device=device,
    )
    logits = model(batch)
    fake_prob = float(torch.softmax(logits.float(), dim=-1)[0, 1])
    if was_training:
        model.train()

    label = int(fake_prob >= float(threshold))
    return {
        "fake_prob": fake_prob,
        "label": label,
        "label_name": "FAKE" if label else "REAL",
        "elapsed_sec": time.perf_counter() - started,
    }


# ---------------------------------------------------------------------------
# config -> late fusion 모델
# ---------------------------------------------------------------------------
def _head_hidden_dims(head: Mapping[str, Any]) -> Any:
    """fusion head 은닉 차원. 키가 없으면 기본 (512,), 명시적 null이면 단일 Linear."""
    if "hidden_dims" not in head:
        return list(DEFAULT_FUSION_HIDDEN_DIMS)
    value = head["hidden_dims"]
    if value is None:
        return None
    if isinstance(value, int):
        return int(value)
    return [int(dim) for dim in value]


@register_model("late_fusion")
def build_late_fusion_model(cfg: Config) -> LateFusionClassifier:
    """Late fusion baseline (AC1, AC2) — config ``model.name: late_fusion``.

    관련 config 키:
        ``model.text.*`` / ``model.image.*``  — Story 1.3과 완전히 동일한 인코더 설정
        ``model.freeze_encoders``             — (a) frozen / (b) end-to-end 분기
        ``model.head.hidden_dims|dropout``    — fusion MLP head
        ``model.text.checkpoint`` / ``model.image.checkpoint`` — Story 1.3 인코더 재사용
    """
    text_encoder = build_text_encoder(cfg)
    image_encoder = build_image_encoder(cfg)
    head = _section(cfg, "model.head")

    model = LateFusionClassifier(
        text_encoder=text_encoder,
        image_encoder=image_encoder,
        num_classes=int(cfg.get("model.num_classes", 2)),
        head_hidden_dims=_head_hidden_dims(head),
        head_dropout=float(head.get("dropout", 0.1)),
    )

    text_ckpt = cfg.get("model.text.checkpoint", None)
    image_ckpt = cfg.get("model.image.checkpoint", None)
    if text_ckpt or image_ckpt:
        model.load_encoders(text_path=text_ckpt, image_path=image_ckpt)

    # 개별 인코더의 `freeze` 플래그보다 상위 스위치를 뒤에 적용한다.
    freeze = cfg.get("model.freeze_encoders", None)
    if freeze is not None:
        model.freeze_encoders() if bool(freeze) else model.unfreeze_encoders()
    return model
