"""Story 1.3 — 단일 모달 baseline(BERT only / ResNet only) 테스트.

**네트워크 없이 전부 통과해야 한다**: 사전학습 가중치는 다운로드하지 않고
`pretrained=False`(랜덤 초기화) 또는 주입된 가짜 인코더로 검증한다.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.fakeddit import FakedditDataset
from src.fusion.baseline import SingleModalClassifier, build_image_encoder, build_text_encoder
from src.fusion.encoders import (
    RESNET_FEATURE_DIMS,
    ClassificationHead,
    EncoderError,
    ImageEncoder,
    TextEncoder,
)
from src.fusion.registry import build_model
from src.training import Trainer, TrainerConfig
from src.utils.config import Config

TINY_BERT = {
    "vocab_size": 128,
    "hidden_size": 32,
    "num_hidden_layers": 1,
    "num_attention_heads": 2,
    "intermediate_size": 32,
}


# ---------------------------------------------------------------------------
# 픽스처
# ---------------------------------------------------------------------------
@pytest.fixture
def tiny_text_encoder() -> TextEncoder:
    return TextEncoder(pretrained=False, config_overrides=TINY_BERT)


@pytest.fixture
def text_batch() -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.tensor([[1, 5, 9, 1, 0], [1, 7, 1, 0, 0]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]], dtype=torch.long),
        "label": torch.tensor([0, 1], dtype=torch.long),
    }


@pytest.fixture
def image_batch() -> dict[str, torch.Tensor]:
    return {
        "image": torch.randn(2, 3, 32, 32),
        "label": torch.tensor([1, 0], dtype=torch.long),
    }


class TinyTextBackbone(nn.Module):
    """HF backbone 계약(`config.hidden_size`, `last_hidden_state`)만 흉내낸 가짜 인코더."""

    class _Cfg:
        hidden_size = 16

    def __init__(self) -> None:
        super().__init__()
        self.config = self._Cfg()
        self.embed = nn.Embedding(128, 16)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        return {"last_hidden_state": self.embed(input_ids)}


class TinyImageBackbone(nn.Module):
    """pooled feature [B, 8]을 내는 가짜 이미지 backbone."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.conv(images).mean(dim=(2, 3))


# ---------------------------------------------------------------------------
# TextEncoder (AC1, AC5)
# ---------------------------------------------------------------------------
def test_text_encoder_default_output_dim_is_768():
    encoder = TextEncoder(pretrained=False, config_overrides={"num_hidden_layers": 1})
    assert encoder.output_dim == 768  # BERT [CLS] 768-dim (AC5)


def test_text_encoder_forward_shape(tiny_text_encoder, text_batch):
    features = tiny_text_encoder(text_batch)
    assert features.shape == (2, tiny_text_encoder.output_dim)
    assert features.dtype == torch.float32


def test_text_encoder_projection_changes_output_dim(text_batch):
    encoder = TextEncoder(pretrained=False, config_overrides=TINY_BERT, proj_dim=768)
    assert encoder.output_dim == 768
    assert encoder(text_batch).shape == (2, 768)


def test_text_encoder_mean_pooling_ignores_padding():
    encoder = TextEncoder(
        pretrained=False, config_overrides=TINY_BERT, pooling="mean", encoder=TinyTextBackbone()
    )
    short = {
        "input_ids": torch.tensor([[1, 5, 9]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
    }
    padded = {
        "input_ids": torch.tensor([[1, 5, 9, 0, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1, 0, 0]]),
    }
    torch.testing.assert_close(encoder(short), encoder(padded))


def test_text_encoder_rejects_unknown_pooling():
    with pytest.raises(EncoderError, match="pooling"):
        TextEncoder(pretrained=False, config_overrides=TINY_BERT, pooling="max")


def test_text_encoder_requires_input_ids(tiny_text_encoder, image_batch):
    with pytest.raises(EncoderError, match="input_ids"):
        tiny_text_encoder(image_batch)


def test_text_encoder_accepts_injected_backbone(text_batch):
    encoder = TextEncoder(encoder=TinyTextBackbone())
    assert encoder.output_dim == 16
    assert encoder(text_batch).shape == (2, 16)


def test_text_encoder_without_hidden_size_raises():
    with pytest.raises(EncoderError, match="hidden_size"):
        TextEncoder(encoder=nn.Linear(4, 4))


# ---------------------------------------------------------------------------
# ImageEncoder (AC2, AC5)
# ---------------------------------------------------------------------------
def test_image_encoder_resnet18_pooled_dim_and_shape(image_batch):
    encoder = ImageEncoder(arch="resnet18", pretrained=False)
    assert encoder.output_dim == RESNET_FEATURE_DIMS["resnet18"] == 512
    assert encoder(image_batch).shape == (2, 512)


def test_image_encoder_resnet50_pooled_dim_is_2048():
    encoder = ImageEncoder(arch="resnet50", pretrained=False)
    assert encoder.output_dim == 2048  # AC5 — ResNet pooled 2048-dim


def test_image_encoder_projection_to_768(image_batch):
    encoder = ImageEncoder(arch="resnet18", pretrained=False, proj_dim=768)
    assert encoder.output_dim == 768
    assert encoder(image_batch).shape == (2, 768)


def test_image_encoder_requires_image_key(text_batch):
    encoder = ImageEncoder(encoder=TinyImageBackbone(), feature_dim=8)
    with pytest.raises(EncoderError, match="image"):
        encoder(text_batch)


def test_image_encoder_rejects_unknown_arch():
    with pytest.raises(EncoderError, match="ResNet"):
        ImageEncoder(arch="not_a_resnet", pretrained=False)


def test_image_encoder_accepts_injected_backbone(image_batch):
    encoder = ImageEncoder(encoder=TinyImageBackbone(), feature_dim=8)
    assert encoder.output_dim == 8
    assert encoder(image_batch).shape == (2, 8)


# ---------------------------------------------------------------------------
# freeze / unfreeze (AC6)
# ---------------------------------------------------------------------------
def test_freeze_disables_backbone_grad_and_keeps_eval_mode(text_batch):
    encoder = TextEncoder(pretrained=False, config_overrides=TINY_BERT, freeze=True)
    assert encoder.frozen
    assert all(not p.requires_grad for p in encoder.encoder.parameters())

    encoder.train()
    assert not encoder.encoder.training  # frozen backbone은 BN/Dropout도 고정

    encoder.unfreeze()
    assert all(p.requires_grad for p in encoder.encoder.parameters())
    encoder.train()
    assert encoder.encoder.training


def test_frozen_encoder_receives_no_gradient(text_batch):
    encoder = TextEncoder(pretrained=False, config_overrides=TINY_BERT, freeze=True)
    model = SingleModalClassifier(encoder)
    model(text_batch).sum().backward()

    assert all(p.grad is None for p in model.encoder.encoder.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.head.parameters())


# ---------------------------------------------------------------------------
# ClassificationHead
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("hidden_dims", [None, 16, [32, 16]])
def test_classification_head_output_shape(hidden_dims):
    head = ClassificationHead(input_dim=24, hidden_dims=hidden_dims)
    assert head(torch.randn(5, 24)).shape == (5, 2)


# ---------------------------------------------------------------------------
# SingleModalClassifier — Trainer 계약 (AC3, AC5)
# ---------------------------------------------------------------------------
def test_text_classifier_forward_returns_logits_b2(tiny_text_encoder, text_batch):
    model = SingleModalClassifier(tiny_text_encoder)
    logits = model(text_batch)
    assert logits.shape == (2, 2)


def test_image_classifier_forward_returns_logits_b2(image_batch):
    model = SingleModalClassifier(ImageEncoder(encoder=TinyImageBackbone(), feature_dim=8))
    assert model(image_batch).shape == (2, 2)


def test_encode_returns_penultimate_features(tiny_text_encoder, text_batch):
    model = SingleModalClassifier(tiny_text_encoder)
    features = model.encode(text_batch)
    assert features.shape == (2, model.feature_dim) == (2, tiny_text_encoder.output_dim)


def test_predict_proba_in_unit_range_and_restores_mode(tiny_text_encoder, text_batch):
    model = SingleModalClassifier(tiny_text_encoder)
    model.train()
    probs = model.predict_proba(text_batch)
    assert probs.shape == (2,)
    assert bool(((probs >= 0) & (probs <= 1)).all())
    assert model.training  # 원래 모드 복원


def test_param_groups_split_encoder_and_head_lr(tiny_text_encoder):
    model = SingleModalClassifier(tiny_text_encoder)
    groups = model.param_groups(encoder_lr=2e-5, head_lr=1e-4)
    assert [g["lr"] for g in groups] == [2e-5, 1e-4]


def test_param_groups_skip_frozen_encoder():
    encoder = TextEncoder(pretrained=False, config_overrides=TINY_BERT, freeze=True)
    groups = SingleModalClassifier(encoder).param_groups(encoder_lr=2e-5, head_lr=1e-3)
    assert len(groups) == 1 and groups[0]["lr"] == 1e-3


def test_save_and_load_encoder_roundtrip(tmp_path, text_batch):
    source = SingleModalClassifier(TextEncoder(pretrained=False, config_overrides=TINY_BERT))
    path = source.save_encoder(tmp_path / "encoder.pt")

    target = SingleModalClassifier(TextEncoder(pretrained=False, config_overrides=TINY_BERT))
    target.load_encoder(path)
    source.eval(), target.eval()  # dropout 비활성화 후 비교
    with torch.no_grad():
        torch.testing.assert_close(source.encode(text_batch), target.encode(text_batch))


# ---------------------------------------------------------------------------
# 레지스트리 / config 주입 (AC1, AC2)
# ---------------------------------------------------------------------------
def _text_cfg(**model_extra) -> Config:
    model = {
        "name": "text_only",
        "text": {"pretrained": False, "encoder_config": TINY_BERT},
        "head": {"dropout": 0.0},
    }
    model.update(model_extra)
    return Config({"exp_name": "t", "seed": 42, "model": model})


def test_build_model_text_only_from_config(text_batch):
    model = build_model(_text_cfg())
    assert isinstance(model, SingleModalClassifier)
    assert model.feature_dim == TINY_BERT["hidden_size"]
    assert model(text_batch).shape == (2, 2)


def test_build_model_image_only_from_config(image_batch):
    cfg = Config(
        {
            "exp_name": "i",
            "seed": 42,
            "model": {"name": "image_only", "image": {"arch": "resnet18", "pretrained": False}},
        }
    )
    model = build_model(cfg)
    assert model.feature_dim == 512
    assert model(image_batch).shape == (2, 2)


def test_build_model_honours_freeze_flag():
    cfg = _text_cfg(text={"pretrained": False, "encoder_config": TINY_BERT, "freeze": True})
    model = build_model(cfg)
    assert model.encoder.frozen


def test_build_model_head_hidden_dims_from_config(text_batch):
    cfg = _text_cfg(head={"hidden_dims": [16], "dropout": 0.0})
    assert build_model(cfg)(text_batch).shape == (2, 2)


def test_existing_dummy_model_still_registered(image_batch):
    cfg = Config({"exp_name": "d", "seed": 42, "model": {"name": "dummy", "hidden_dim": 8}})
    assert build_model(cfg)(image_batch).shape == (2, 2)


def test_unknown_model_name_raises():
    with pytest.raises(KeyError, match="nope"):
        build_model(Config({"exp_name": "x", "seed": 42, "model": {"name": "nope"}}))


def test_shipped_baseline_configs_select_expected_models():
    from src.utils.config import load_config

    bert_cfg = load_config("configs/bert_only.yaml")
    resnet_cfg = load_config("configs/resnet_only.yaml")
    assert (bert_cfg.get("model.name"), bert_cfg.get("data.modality")) == ("text_only", "text")
    assert (resnet_cfg.get("model.name"), resnet_cfg.get("data.modality")) == ("image_only", "image")
    assert resnet_cfg.get("model.image.arch") == "resnet50"
    assert bert_cfg.get("model.text.model_name") == "bert-base-uncased"


# ---------------------------------------------------------------------------
# 공통 Trainer + Story 1.2 Dataset 연결 (AC3)
# ---------------------------------------------------------------------------
def _loader(dataset: FakedditDataset) -> DataLoader:
    return DataLoader(dataset, batch_size=4, collate_fn=dataset.collate_fn)


def test_text_baseline_trains_with_common_trainer(tmp_path, synthetic_manifest, fake_tokenizer):
    dataset = FakedditDataset(synthetic_manifest, modality="text", tokenizer=fake_tokenizer)
    loader = _loader(dataset)
    trainer = Trainer(
        model=build_model(_text_cfg()),
        cfg=TrainerConfig(epochs=1, lr=1e-3, log_every=0),
        out_dir=tmp_path,
    )
    summary = trainer.fit(loader, loader)
    assert summary["best_epoch"] == 1
    assert (tmp_path / "best.pt").is_file()


def test_image_baseline_trains_with_common_trainer(
    tmp_path, synthetic_manifest, synthetic_image_dir
):
    assert synthetic_image_dir.is_dir()
    dataset = FakedditDataset(synthetic_manifest, modality="image", image_size=32)
    loader = _loader(dataset)
    cfg = Config(
        {
            "exp_name": "i",
            "seed": 42,
            "model": {"name": "image_only", "image": {"arch": "resnet18", "pretrained": False}},
        }
    )
    trainer = Trainer(
        model=build_model(cfg), cfg=TrainerConfig(epochs=1, lr=1e-3, log_every=0), out_dir=tmp_path
    )
    result = trainer.evaluate(loader, prefix="test")
    assert set(result) >= {"test_accuracy", "test_f1", "test_auroc"}

    summary = trainer.fit(loader, loader)
    assert summary["best_epoch"] == 1


def test_same_seed_gives_identical_logits(text_batch):
    """AC7 재현성 — 동일 seed로 만든 모델은 동일 출력을 낸다."""
    from src.utils.seed import set_seed

    outputs = []
    for _ in range(2):
        set_seed(42)
        model = build_model(_text_cfg()).eval()
        with torch.no_grad():
            outputs.append(model(text_batch))
    torch.testing.assert_close(outputs[0], outputs[1])


def test_config_builders_are_reusable_for_late_fusion():
    """Story 1.4가 인코더 빌더를 그대로 재사용할 수 있는지 (NFR6)."""
    cfg = Config(
        {
            "exp_name": "lf",
            "seed": 42,
            "model": {
                "text": {"pretrained": False, "encoder_config": TINY_BERT},
                "image": {"arch": "resnet18", "pretrained": False},
            },
        }
    )
    text_encoder, image_encoder = build_text_encoder(cfg).eval(), build_image_encoder(cfg).eval()
    assert text_encoder.output_dim + image_encoder.output_dim == TINY_BERT["hidden_size"] + 512

    batch = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.ones(1, 3, dtype=torch.long),
        "image": torch.randn(1, 3, 32, 32),
    }
    concat = torch.cat([text_encoder(batch), image_encoder(batch)], dim=-1)
    assert concat.shape == (1, TINY_BERT["hidden_size"] + 512)
