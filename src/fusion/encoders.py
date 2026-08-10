"""재사용 가능한 단일 모달 인코더 (Story 1.3 AC1, AC2, AC5).

설계 원칙 — **인코더는 분류기와 분리한다**:
Story 1.4의 late fusion과 Epic 2 이후의 cross-modal fusion이 여기 있는
``TextEncoder`` / ``ImageEncoder``를 **그대로 재사용**한다. 따라서 두 인코더는
분류 head를 갖지 않고, `forward(batch) -> features [B, D]`만 책임진다.

공통 인터페이스:
- ``forward(batch: dict) -> Tensor [B, output_dim]``  (pooled feature)
- ``output_dim: int``  (프로퍼티 — fusion head가 입력 dim을 질의한다)
- ``freeze()`` / ``unfreeze()``  (frozen backbone 학습 모드, NFR1)

오프라인 제약:
사전학습 가중치 다운로드가 불가능한 환경에서도 전 경로가 동작해야 하므로
``pretrained=False``면 **랜덤 초기화**로 인코더를 구성한다(네트워크 접근 없음).
테스트는 이 경로 또는 ``encoder=`` 인자로 주입한 가짜 인코더를 사용한다.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
from torch import nn

#: 프로젝트 공통 hidden dim — BERT-base [CLS] 차원이자 이후 Epic의 fusion 기준 dim.
DEFAULT_HIDDEN_DIM: int = 768

#: torchvision ResNet 변형별 pooled feature dim (참고용 — 실제로는 fc.in_features에서 읽는다)
RESNET_FEATURE_DIMS: dict[str, int] = {
    "resnet18": 512,
    "resnet34": 512,
    "resnet50": 2048,
    "resnet101": 2048,
    "resnet152": 2048,
}

TEXT_POOLINGS: tuple[str, ...] = ("cls", "mean")


class EncoderError(ValueError):
    """인코더 설정이 유효하지 않을 때."""


def set_requires_grad(module: nn.Module, flag: bool) -> nn.Module:
    for param in module.parameters():
        param.requires_grad = flag
    return module


class _BaseEncoder(nn.Module):
    """freeze/unfreeze와 output_dim 계약을 공유하는 인코더 베이스."""

    def _init_base(self, output_dim: int, freeze: bool = False) -> None:
        """서브클래스가 `nn.Module.__init__`과 하위 모듈 등록을 마친 뒤 호출한다.

        (여기서 ``nn.Module.__init__``을 다시 부르면 이미 등록된 submodule이
        초기화되므로 별도 메서드로 분리했다.)
        """
        self._output_dim = int(output_dim)
        self._frozen = False
        if freeze:
            self.freeze()

    @property
    def output_dim(self) -> int:
        return self._output_dim

    @property
    def frozen(self) -> bool:
        return self._frozen

    def freeze(self) -> "_BaseEncoder":
        """backbone 파라미터를 고정한다 (projection head는 별도 처리)."""
        set_requires_grad(self.backbone_module(), False)
        self._frozen = True
        return self

    def unfreeze(self) -> "_BaseEncoder":
        set_requires_grad(self.backbone_module(), True)
        self._frozen = False
        return self

    def backbone_module(self) -> nn.Module:  # pragma: no cover - 서브클래스가 구현
        raise NotImplementedError

    def train(self, mode: bool = True):  # noqa: D102 - nn.Module override
        super().train(mode)
        # frozen backbone은 BatchNorm/Dropout 통계도 갱신하지 않도록 eval로 유지한다.
        if self._frozen:
            self.backbone_module().eval()
        return self


# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------
def build_hf_text_backbone(
    model_name: str = "bert-base-uncased",
    pretrained: bool = True,
    local_files_only: bool = False,
    config_overrides: Mapping[str, Any] | None = None,
) -> nn.Module:
    """HuggingFace BERT 계열 backbone을 만든다.

    ``pretrained=True``면 ``AutoModel.from_pretrained``(네트워크/캐시 필요),
    ``False``면 ``BertConfig``로 **랜덤 초기화**한다(오프라인 안전 경로).
    """
    overrides = dict(config_overrides or {})
    if pretrained:
        from transformers import AutoModel

        return AutoModel.from_pretrained(str(model_name), local_files_only=bool(local_files_only))

    from transformers import BertConfig, BertModel

    defaults: dict[str, Any] = {
        "vocab_size": 128,
        "hidden_size": DEFAULT_HIDDEN_DIM,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "intermediate_size": 128,
        "max_position_embeddings": 512,
    }
    defaults.update(overrides)
    return BertModel(BertConfig(**defaults))


class TextEncoder(_BaseEncoder):
    """BERT 계열 텍스트 인코더 — batch dict -> pooled feature [B, hidden].

    Args:
        model_name: HuggingFace 모델 id (`pretrained=True`일 때만 사용).
        pretrained: 사전학습 가중치 로딩 여부. False면 랜덤 초기화(오프라인).
        encoder: 이미 만들어진 backbone을 주입 (테스트/재사용 경로). 주어지면
            model_name/pretrained는 무시된다.
        pooling: ``cls``([CLS] 토큰) 또는 ``mean``(attention_mask 가중 평균).
        freeze: backbone 고정 여부.
        proj_dim: 지정 시 pooled feature를 이 차원으로 선형 사영한다
            (인코더 간 dim 정렬용). None이면 backbone hidden dim 그대로.
        dropout: pooled feature에 적용할 dropout 확률.
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        pretrained: bool = True,
        encoder: nn.Module | None = None,
        pooling: str = "cls",
        freeze: bool = False,
        proj_dim: int | None = None,
        dropout: float = 0.0,
        local_files_only: bool = False,
        config_overrides: Mapping[str, Any] | None = None,
    ) -> None:
        pooling = str(pooling).lower()
        if pooling not in TEXT_POOLINGS:
            raise EncoderError(f"pooling은 {TEXT_POOLINGS} 중 하나여야 합니다 (받은 값: {pooling!r})")

        backbone = encoder if encoder is not None else build_hf_text_backbone(
            model_name=model_name,
            pretrained=pretrained,
            local_files_only=local_files_only,
            config_overrides=config_overrides,
        )
        hidden = _infer_text_hidden_dim(backbone)
        out_dim = int(proj_dim) if proj_dim else hidden

        nn.Module.__init__(self)
        self.encoder = backbone
        self.pooling = pooling
        self.dropout = nn.Dropout(float(dropout))
        self.projection: nn.Module = nn.Linear(hidden, out_dim) if proj_dim else nn.Identity()
        self.hidden_dim = hidden
        self._init_base(output_dim=out_dim, freeze=freeze)

    def backbone_module(self) -> nn.Module:
        return self.encoder

    def forward(self, batch: Mapping[str, Any]) -> torch.Tensor:
        if "input_ids" not in batch:
            raise EncoderError(
                "TextEncoder는 배치에 'input_ids'가 필요합니다. "
                "config `data.modality`를 text/both로 두고 tokenizer를 주입하세요."
            )
        kwargs: dict[str, Any] = {"input_ids": batch["input_ids"]}
        for key in ("attention_mask", "token_type_ids"):
            if key in batch and batch[key] is not None:
                kwargs[key] = batch[key]

        outputs = self.encoder(**kwargs)
        hidden_states = getattr(outputs, "last_hidden_state", None)
        if hidden_states is None:  # dict 또는 tuple을 반환하는 backbone 호환
            hidden_states = outputs["last_hidden_state"] if isinstance(outputs, Mapping) else outputs[0]

        if self.pooling == "cls":
            pooled = hidden_states[:, 0]
        else:
            mask = kwargs.get("attention_mask")
            if mask is None:
                pooled = hidden_states.mean(dim=1)
            else:
                weights = mask.unsqueeze(-1).to(hidden_states.dtype)
                pooled = (hidden_states * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1e-9)
        return self.projection(self.dropout(pooled))


def _infer_text_hidden_dim(backbone: nn.Module) -> int:
    config = getattr(backbone, "config", None)
    hidden = getattr(config, "hidden_size", None) if config is not None else None
    if hidden is None:
        hidden = getattr(backbone, "hidden_size", None)
    if hidden is None:
        raise EncoderError(
            "text backbone에서 hidden_size를 찾을 수 없습니다 — "
            "`config.hidden_size` 또는 `hidden_size` 속성을 제공하세요."
        )
    return int(hidden)


# ---------------------------------------------------------------------------
# Image
# ---------------------------------------------------------------------------
def build_torchvision_resnet(arch: str = "resnet50", pretrained: bool = True) -> nn.Module:
    """torchvision ResNet backbone (fc 제거 전 원본)을 만든다.

    ``pretrained=False``면 weights=None — 다운로드가 발생하지 않는다.
    """
    from torchvision import models

    if not hasattr(models, str(arch)):
        raise EncoderError(f"torchvision에 없는 ResNet 변형: {arch!r} (예: {sorted(RESNET_FEATURE_DIMS)})")
    factory = getattr(models, str(arch))
    weights = "DEFAULT" if pretrained else None
    return factory(weights=weights)


class ImageEncoder(_BaseEncoder):
    """torchvision ResNet 이미지 인코더 — batch dict -> pooled feature [B, D].

    ResNet의 ``fc``를 ``Identity``로 교체해 global average pooled feature
    (resnet50 기준 2048-dim)를 그대로 노출한다. ``proj_dim``을 주면 그 차원으로
    선형 사영한다 (예: 텍스트와 동일한 768-dim 정렬).
    """

    def __init__(
        self,
        arch: str = "resnet50",
        pretrained: bool = True,
        encoder: nn.Module | None = None,
        freeze: bool = False,
        proj_dim: int | None = None,
        dropout: float = 0.0,
        feature_dim: int | None = None,
    ) -> None:
        backbone = encoder if encoder is not None else build_torchvision_resnet(arch, pretrained)
        pooled_dim = int(feature_dim) if feature_dim else _infer_image_feature_dim(backbone, arch)
        # torchvision ResNet이면 분류 head를 제거해 feature extractor로 만든다.
        if hasattr(backbone, "fc") and isinstance(getattr(backbone, "fc"), nn.Linear):
            backbone.fc = nn.Identity()
        out_dim = int(proj_dim) if proj_dim else pooled_dim

        nn.Module.__init__(self)
        self.encoder = backbone
        self.dropout = nn.Dropout(float(dropout))
        self.projection: nn.Module = nn.Linear(pooled_dim, out_dim) if proj_dim else nn.Identity()
        self.feature_dim = pooled_dim
        self._init_base(output_dim=out_dim, freeze=freeze)

    def backbone_module(self) -> nn.Module:
        return self.encoder

    def forward(self, batch: Mapping[str, Any]) -> torch.Tensor:
        if "image" not in batch:
            raise EncoderError(
                "ImageEncoder는 배치에 'image'가 필요합니다. config `data.modality`를 image/both로 두세요."
            )
        pooled = self.encoder(batch["image"])
        if pooled.dim() > 2:  # [B, C, 1, 1] 형태를 반환하는 backbone 호환
            pooled = torch.flatten(pooled, 1)
        return self.projection(self.dropout(pooled))


def _infer_image_feature_dim(backbone: nn.Module, arch: str) -> int:
    fc = getattr(backbone, "fc", None)
    if isinstance(fc, nn.Linear):
        return int(fc.in_features)
    if getattr(backbone, "output_dim", None):
        return int(backbone.output_dim)
    if arch in RESNET_FEATURE_DIMS:
        return RESNET_FEATURE_DIMS[arch]
    raise EncoderError(
        f"image backbone의 feature dim을 추론할 수 없습니다 (arch={arch!r}) — "
        "`feature_dim=` 인자로 명시하세요."
    )


# ---------------------------------------------------------------------------
# Classification head
# ---------------------------------------------------------------------------
class ClassificationHead(nn.Module):
    """feature -> logits [B, num_classes] MLP head.

    ``hidden_dims``가 비어 있으면 단일 Linear (BERT 기본 분류 head와 동일).
    Story 1.4의 fusion head도 이 클래스를 그대로 쓴다 (입력 dim만 다름).
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 2,
        hidden_dims: Sequence[int] | int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            dims: list[int] = []
        elif isinstance(hidden_dims, int):
            dims = [int(hidden_dims)]
        else:
            dims = [int(dim) for dim in hidden_dims]

        layers: list[nn.Module] = []
        current = int(input_dim)
        for dim in dims:
            layers += [nn.Dropout(float(dropout)), nn.Linear(current, dim), nn.ReLU()]
            current = dim
        layers += [nn.Dropout(float(dropout)), nn.Linear(current, int(num_classes))]
        self.net = nn.Sequential(*layers)
        self.input_dim = int(input_dim)
        self.num_classes = int(num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)
