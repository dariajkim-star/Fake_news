"""Story 1.4 — Late fusion baseline (BERT + ResNet) 테스트.

**네트워크 없이 전부 통과해야 한다**: 사전학습 가중치는 다운로드하지 않고
`pretrained=False`(랜덤 초기화) 또는 주입된 가짜 인코더로만 검증한다.
"""

from __future__ import annotations

import json

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.fakeddit import FakedditDataset
from src.evaluation.ablation import (
    EPIC1_ROWS,
    AblationRow,
    collect_ablation,
    f1_comparison,
    load_metrics,
    render_markdown,
    write_ablation,
)
from src.fusion.baseline import (
    LateFusionClassifier,
    SingleModalClassifier,
    build_image_encoder,
    build_single_sample_batch,
    build_text_encoder,
    predict_single,
)
from src.fusion.encoders import ImageEncoder, TextEncoder
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


class TinyImageBackbone(nn.Module):
    """pooled feature [B, 8]을 내는 가짜 이미지 backbone (네트워크/가중치 불필요)."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.conv(images).mean(dim=(2, 3))


# ---------------------------------------------------------------------------
# 픽스처
# ---------------------------------------------------------------------------
@pytest.fixture
def tiny_text_encoder() -> TextEncoder:
    return TextEncoder(pretrained=False, config_overrides=TINY_BERT)


@pytest.fixture
def tiny_image_encoder() -> ImageEncoder:
    return ImageEncoder(encoder=TinyImageBackbone(), feature_dim=8)


@pytest.fixture
def fusion_model(tiny_text_encoder, tiny_image_encoder) -> LateFusionClassifier:
    return LateFusionClassifier(tiny_text_encoder, tiny_image_encoder, head_dropout=0.0)


@pytest.fixture
def multimodal_batch() -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.tensor([[1, 5, 9, 1, 0], [1, 7, 1, 0, 0]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]], dtype=torch.long),
        "image": torch.randn(2, 3, 32, 32),
        "label": torch.tensor([0, 1], dtype=torch.long),
    }


def _fusion_cfg(**model_extra) -> Config:
    model = {
        "name": "late_fusion",
        "text": {"pretrained": False, "encoder_config": TINY_BERT},
        "image": {"arch": "resnet18", "pretrained": False},
        "head": {"hidden_dims": [16], "dropout": 0.0},
    }
    model.update(model_extra)
    return Config({"exp_name": "lf", "seed": 42, "model": model})


# ---------------------------------------------------------------------------
# AC1 — concat 차원과 output shape
# ---------------------------------------------------------------------------
def test_feature_dim_is_sum_of_encoder_dims(fusion_model):
    assert fusion_model.text_dim == TINY_BERT["hidden_size"]
    assert fusion_model.image_dim == 8
    assert fusion_model.feature_dim == TINY_BERT["hidden_size"] + 8


def test_native_dims_concat_to_2816():
    """AC1 — BERT [CLS] 768 + ResNet-50 pooled 2048 = 2816 (가중치 다운로드 없음)."""
    text = TextEncoder(pretrained=False, config_overrides={"num_hidden_layers": 1})
    image = ImageEncoder(arch="resnet50", pretrained=False)
    model = LateFusionClassifier(text, image)
    assert (text.output_dim, image.output_dim) == (768, 2048)
    assert model.feature_dim == 2816
    assert model.head.input_dim == 2816


def test_encode_returns_concat_features(fusion_model, multimodal_batch):
    features = fusion_model.encode(multimodal_batch)
    assert features.shape == (2, fusion_model.feature_dim)


def test_encode_modalities_matches_concat_order(fusion_model, multimodal_batch):
    fusion_model.eval()
    with torch.no_grad():
        parts = fusion_model.encode_modalities(multimodal_batch)
        concat = fusion_model.encode(multimodal_batch)
    assert parts["text"].shape == (2, fusion_model.text_dim)
    assert parts["image"].shape == (2, fusion_model.image_dim)
    torch.testing.assert_close(concat, torch.cat([parts["text"], parts["image"]], dim=-1))


def test_forward_returns_logits_b2(fusion_model, multimodal_batch):
    assert fusion_model(multimodal_batch).shape == (2, 2)


def test_forward_uses_both_modalities(fusion_model, multimodal_batch):
    """이미지를 바꾸면 출력이 바뀐다 — 텍스트만 보고 있지 않은지 확인."""
    fusion_model.eval()
    with torch.no_grad():
        base = fusion_model(multimodal_batch)
        perturbed = dict(multimodal_batch, image=multimodal_batch["image"] + 5.0)
        assert not torch.allclose(base, fusion_model(perturbed))

        perturbed_text = dict(multimodal_batch, input_ids=multimodal_batch["input_ids"] * 0 + 3)
        assert not torch.allclose(base, fusion_model(perturbed_text))


def test_predict_proba_in_unit_range_and_restores_mode(fusion_model, multimodal_batch):
    fusion_model.train()
    probs = fusion_model.predict_proba(multimodal_batch)
    assert probs.shape == (2,)
    assert bool(((probs >= 0) & (probs <= 1)).all())
    assert fusion_model.training


@pytest.mark.parametrize("hidden_dims", [None, 16, [32, 16]])
def test_head_hidden_dims_variants(tiny_text_encoder, tiny_image_encoder, multimodal_batch, hidden_dims):
    model = LateFusionClassifier(
        tiny_text_encoder, tiny_image_encoder, head_hidden_dims=hidden_dims, head_dropout=0.0
    )
    assert model(multimodal_batch).shape == (2, 2)


# ---------------------------------------------------------------------------
# AC2 — frozen / end-to-end 학습 모드
# ---------------------------------------------------------------------------
def test_freeze_encoders_sets_requires_grad_false(fusion_model):
    fusion_model.freeze_encoders()
    assert fusion_model.encoders_frozen
    assert all(not p.requires_grad for p in fusion_model.text_encoder.parameters())
    assert all(not p.requires_grad for p in fusion_model.image_encoder.parameters())


def test_unfreeze_encoders_restores_grad(fusion_model):
    fusion_model.freeze_encoders().unfreeze_encoders()
    assert not fusion_model.encoders_frozen
    assert all(p.requires_grad for p in fusion_model.text_encoder.parameters())


def test_frozen_encoders_receive_no_gradient(fusion_model, multimodal_batch):
    fusion_model.freeze_encoders()
    fusion_model(multimodal_batch).sum().backward()
    assert all(p.grad is None for p in fusion_model.text_encoder.parameters())
    assert all(p.grad is None for p in fusion_model.image_encoder.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in fusion_model.head.parameters())


def test_end_to_end_mode_updates_encoders(fusion_model, multimodal_batch):
    fusion_model.unfreeze_encoders()
    fusion_model(multimodal_batch).sum().backward()
    assert any(p.grad is not None for p in fusion_model.text_encoder.parameters())
    assert any(p.grad is not None for p in fusion_model.image_encoder.parameters())


def test_param_groups_split_encoder_and_head_lr(fusion_model):
    groups = fusion_model.param_groups(encoder_lr=2e-5, head_lr=1e-3)
    assert [g["lr"] for g in groups] == [2e-5, 1e-3]
    # 두 인코더의 파라미터가 모두 한 그룹에 들어간다
    expected = len(list(fusion_model.text_encoder.parameters())) + len(
        list(fusion_model.image_encoder.parameters())
    )
    assert len(groups[0]["params"]) == expected


def test_param_groups_skip_frozen_encoders(fusion_model):
    fusion_model.freeze_encoders()
    groups = fusion_model.param_groups(encoder_lr=2e-5, head_lr=1e-3)
    assert len(groups) == 1 and groups[0]["lr"] == 1e-3


def test_frozen_mode_leaves_encoder_weights_unchanged(fusion_model, multimodal_batch):
    fusion_model.freeze_encoders()
    before = fusion_model.text_encoder.encoder.embeddings.word_embeddings.weight.detach().clone()
    optimizer = torch.optim.AdamW(fusion_model.param_groups(encoder_lr=1e-3, head_lr=1e-2))
    loss = nn.CrossEntropyLoss()(fusion_model(multimodal_batch), multimodal_batch["label"])
    loss.backward()
    optimizer.step()
    after = fusion_model.text_encoder.encoder.embeddings.word_embeddings.weight
    torch.testing.assert_close(before, after)


# ---------------------------------------------------------------------------
# 레지스트리 / config (AC1, AC2) — 기존 등록을 깨지 않는지 포함
# ---------------------------------------------------------------------------
def test_build_model_late_fusion_from_config(multimodal_batch):
    model = build_model(_fusion_cfg())
    assert isinstance(model, LateFusionClassifier)
    assert model.feature_dim == TINY_BERT["hidden_size"] + 512  # resnet18 pooled 512
    assert model(multimodal_batch).shape == (2, 2)


def test_build_model_freeze_encoders_flag(multimodal_batch):
    frozen = build_model(_fusion_cfg(freeze_encoders=True))
    assert frozen.encoders_frozen
    thawed = build_model(_fusion_cfg(freeze_encoders=False))
    assert not thawed.encoders_frozen


def test_build_model_default_fusion_head_has_hidden_layer():
    """head 블록을 생략하면 기본 MLP head(2816 -> 512 -> 2)가 붙는다."""
    cfg = Config(
        {
            "exp_name": "lf",
            "seed": 42,
            "model": {
                "name": "late_fusion",
                "text": {"pretrained": False, "encoder_config": TINY_BERT},
                "image": {"arch": "resnet18", "pretrained": False},
            },
        }
    )
    model = build_model(cfg)
    linears = [m for m in model.head.net if isinstance(m, nn.Linear)]
    assert [m.out_features for m in linears] == [512, 2]


def test_build_model_explicit_null_head_is_single_linear():
    cfg = _fusion_cfg(head={"hidden_dims": None, "dropout": 0.0})
    model = build_model(cfg)
    linears = [m for m in model.head.net if isinstance(m, nn.Linear)]
    assert len(linears) == 1 and linears[0].out_features == 2


def test_existing_model_registrations_still_work(multimodal_batch):
    """dummy / text_only / image_only 등록이 깨지지 않았는지 (회귀)."""
    dummy = build_model(Config({"exp_name": "d", "seed": 1, "model": {"name": "dummy", "hidden_dim": 8}}))
    assert dummy(multimodal_batch).shape == (2, 2)

    text_only = build_model(
        Config(
            {
                "exp_name": "t",
                "seed": 1,
                "model": {"name": "text_only", "text": {"pretrained": False, "encoder_config": TINY_BERT}},
            }
        )
    )
    assert isinstance(text_only, SingleModalClassifier)
    assert text_only(multimodal_batch).shape == (2, 2)

    image_only = build_model(
        Config(
            {
                "exp_name": "i",
                "seed": 1,
                "model": {"name": "image_only", "image": {"arch": "resnet18", "pretrained": False}},
            }
        )
    )
    assert image_only(multimodal_batch).shape == (2, 2)


def test_shipped_late_fusion_config():
    from src.utils.config import load_config

    cfg = load_config("configs/late_fusion.yaml")
    assert cfg.get("model.name") == "late_fusion"
    assert cfg.get("data.modality") == "both"          # 두 모달 모두 로딩
    assert cfg.get("model.freeze_encoders") is True    # 기본은 학습 모드 (a)
    assert cfg.get("model.head.hidden_dims") == [512]
    assert cfg.get("model.text.proj_dim") is None and cfg.get("model.image.proj_dim") is None
    assert cfg.get("data.num_workers") == 0            # Windows 기본
    assert cfg.get("train.head_lr") is not None        # param group 분리 경로


def test_ablation_interface_matches_single_modal_models():
    """A1~A3 비교를 위해 세 모델이 같은 인터페이스를 노출하는지 (NFR6)."""
    models = [
        build_model(
            Config(
                {
                    "exp_name": "t",
                    "seed": 1,
                    "model": {"name": "text_only", "text": {"pretrained": False, "encoder_config": TINY_BERT}},
                }
            )
        ),
        build_model(
            Config(
                {
                    "exp_name": "i",
                    "seed": 1,
                    "model": {"name": "image_only", "image": {"arch": "resnet18", "pretrained": False}},
                }
            )
        ),
        build_model(_fusion_cfg()),
    ]
    for model in models:
        for attribute in ("forward", "encode", "predict_proba", "param_groups", "feature_dim"):
            assert hasattr(model, attribute), (type(model).__name__, attribute)


# ---------------------------------------------------------------------------
# Story 1.3 인코더 checkpoint 재사용
# ---------------------------------------------------------------------------
def test_load_encoders_from_story_1_3_checkpoints(tmp_path, multimodal_batch):
    text_source = SingleModalClassifier(TextEncoder(pretrained=False, config_overrides=TINY_BERT))
    image_source = SingleModalClassifier(ImageEncoder(encoder=TinyImageBackbone(), feature_dim=8))
    text_ckpt = text_source.save_encoder(tmp_path / "text_encoder.pt")
    image_ckpt = image_source.save_encoder(tmp_path / "image_encoder.pt")

    fusion = LateFusionClassifier(
        TextEncoder(pretrained=False, config_overrides=TINY_BERT),
        ImageEncoder(encoder=TinyImageBackbone(), feature_dim=8),
    )
    fusion.load_encoders(text_path=text_ckpt, image_path=image_ckpt).eval()
    text_source.eval(), image_source.eval()
    with torch.no_grad():
        parts = fusion.encode_modalities(multimodal_batch)
        torch.testing.assert_close(parts["text"], text_source.encode(multimodal_batch))
        torch.testing.assert_close(parts["image"], image_source.encode(multimodal_batch))


def test_checkpoint_paths_from_config(tmp_path, multimodal_batch):
    reference = LateFusionClassifier(
        TextEncoder(pretrained=False, config_overrides=TINY_BERT),
        ImageEncoder(arch="resnet18", pretrained=False),
        head_hidden_dims=[16],
        head_dropout=0.0,
    )
    saved = reference.save_encoders(tmp_path / "t.pt", tmp_path / "i.pt")

    cfg = _fusion_cfg(
        text={"pretrained": False, "encoder_config": TINY_BERT, "checkpoint": str(saved["text"])},
        image={"arch": "resnet18", "pretrained": False, "checkpoint": str(saved["image"])},
    )
    model = build_model(cfg).eval()
    reference.eval()
    with torch.no_grad():
        torch.testing.assert_close(model.encode(multimodal_batch), reference.encode(multimodal_batch))


def test_save_encoders_skips_none_paths(tmp_path, fusion_model):
    saved = fusion_model.save_encoders(text_path=tmp_path / "only_text.pt")
    assert set(saved) == {"text"} and saved["text"].is_file()


# ---------------------------------------------------------------------------
# AC6 — 단일 샘플 추론
# ---------------------------------------------------------------------------
def test_predict_single_returns_probability(fusion_model, tmp_path, write_image, fake_tokenizer):
    image_path = write_image(tmp_path / "sample.jpg")
    result = fusion_model.predict(
        image_path=image_path,
        text="tesla stock soars after earnings call",
        tokenizer=fake_tokenizer,
        image_size=32,
    )
    assert 0.0 <= result["fake_prob"] <= 1.0
    assert result["label"] in (0, 1)
    assert result["label_name"] == ("FAKE" if result["label"] else "REAL")
    assert result["elapsed_sec"] >= 0.0
    assert result["elapsed_sec"] < 5.0  # NFR2 사전 검증 (CPU, 소형 인코더 기준)


def test_predict_single_restores_training_mode(fusion_model, tmp_path, write_image, fake_tokenizer):
    fusion_model.train()
    fusion_model.predict(
        image_path=write_image(tmp_path / "s.jpg"), text="hello", tokenizer=fake_tokenizer, image_size=32
    )
    assert fusion_model.training


def test_predict_single_is_deterministic_in_eval(fusion_model, tmp_path, write_image, fake_tokenizer):
    image_path = write_image(tmp_path / "s.jpg")
    kwargs = dict(image_path=image_path, text="market cap surges", tokenizer=fake_tokenizer, image_size=32)
    first = fusion_model.predict(**kwargs)["fake_prob"]
    second = fusion_model.predict(**kwargs)["fake_prob"]
    assert first == pytest.approx(second)


def test_predict_single_works_for_single_modal_model(tmp_path, write_image):
    """추론 함수는 모델 종류에 의존하지 않는다 (image_only에도 적용)."""
    model = SingleModalClassifier(ImageEncoder(encoder=TinyImageBackbone(), feature_dim=8))
    result = predict_single(model, image_path=write_image(tmp_path / "s.jpg"), image_size=32)
    assert 0.0 <= result["fake_prob"] <= 1.0


def test_build_single_sample_batch_accepts_tensor_image(fake_tokenizer):
    batch = build_single_sample_batch(
        image_path=torch.randn(3, 16, 16), text="a b c", tokenizer=fake_tokenizer
    )
    assert batch["image"].shape == (1, 3, 16, 16)
    assert batch["input_ids"].shape[0] == 1
    assert batch["attention_mask"].shape == batch["input_ids"].shape


def test_build_single_sample_batch_without_tokenizer_keeps_raw_text():
    batch = build_single_sample_batch(text="raw text only")
    assert batch["text"] == ["raw text only"] and "input_ids" not in batch


def test_build_single_sample_batch_requires_some_input():
    with pytest.raises(ValueError, match="image_path 또는 text"):
        build_single_sample_batch()


# ---------------------------------------------------------------------------
# AC3/AC7 — 공통 Trainer + Story 1.2 분할 연동, 재현성
# ---------------------------------------------------------------------------
def test_late_fusion_trains_with_common_trainer(
    tmp_path, synthetic_manifest, synthetic_image_dir, fake_tokenizer
):
    assert synthetic_image_dir.is_dir()
    dataset = FakedditDataset(
        synthetic_manifest, modality="both", tokenizer=fake_tokenizer, image_size=32
    )
    loader = DataLoader(dataset, batch_size=4, collate_fn=dataset.collate_fn)
    trainer = Trainer(
        model=build_model(_fusion_cfg()),
        cfg=TrainerConfig(epochs=1, lr=1e-3, log_every=0),
        out_dir=tmp_path,
    )
    summary = trainer.fit(loader, loader)
    assert summary["best_epoch"] == 1
    assert (tmp_path / "best.pt").is_file()

    result = trainer.evaluate(loader, prefix="test")
    assert set(result) >= {"test_accuracy", "test_precision", "test_recall", "test_f1", "test_auroc"}


def test_same_seed_gives_identical_logits(multimodal_batch):
    """AC7 재현성 — 동일 seed로 만든 모델은 동일 출력을 낸다."""
    from src.utils.seed import set_seed

    outputs = []
    for _ in range(2):
        set_seed(42)
        model = build_model(_fusion_cfg()).eval()
        with torch.no_grad():
            outputs.append(model(multimodal_batch))
    torch.testing.assert_close(outputs[0], outputs[1])


# ---------------------------------------------------------------------------
# AC4, AC5 — ablation table
# ---------------------------------------------------------------------------
def _write_metrics(root, exp_name: str, split: str = "test", **values) -> None:
    directory = root / exp_name
    directory.mkdir(parents=True, exist_ok=True)
    payload = {f"{split}_{key}": value for key, value in values.items()}
    payload["split"] = split
    (directory / f"metrics_{split}.json").write_text(json.dumps(payload), encoding="utf-8")


def test_epic1_rows_cover_three_ablation_configs():
    assert [row.exp_name for row in EPIC1_ROWS] == ["bert_only", "resnet_only", "late_fusion"]


def test_load_metrics_strips_split_prefix(tmp_path):
    _write_metrics(tmp_path, "bert_only", accuracy=0.7, precision=0.7, recall=0.7, f1=0.7, auroc=0.75)
    metrics = load_metrics(tmp_path, "bert_only", split="test")
    assert metrics["f1"] == 0.7 and metrics["auroc"] == 0.75


def test_load_metrics_returns_none_when_missing(tmp_path):
    assert load_metrics(tmp_path, "nope") is None


def test_collect_ablation_marks_missing_rows(tmp_path):
    _write_metrics(tmp_path, "bert_only", accuracy=0.7, precision=0.7, recall=0.7, f1=0.70, auroc=0.75)
    table = collect_ablation(tmp_path)
    assert list(table["model"]) == [row.model for row in EPIC1_ROWS]
    assert list(table["status"]) == ["ok", "missing", "missing"]
    assert table.loc[0, "f1"] == 0.70
    assert table.loc[1, "f1"] != table.loc[1, "f1"]  # NaN


def test_f1_comparison_reports_improvement(tmp_path):
    _write_metrics(tmp_path, "bert_only", accuracy=0.7, precision=0.7, recall=0.7, f1=0.70, auroc=0.75)
    _write_metrics(tmp_path, "resnet_only", accuracy=0.6, precision=0.6, recall=0.6, f1=0.60, auroc=0.65)
    _write_metrics(tmp_path, "late_fusion", accuracy=0.8, precision=0.8, recall=0.8, f1=0.78, auroc=0.85)

    comparison = f1_comparison(collect_ablation(tmp_path))
    assert comparison["best_single_modal"] == "bert_only"
    assert comparison["improved"] is True
    assert comparison["deltas"]["bert_only"] == pytest.approx(0.08)
    assert comparison["deltas"]["resnet_only"] == pytest.approx(0.18)


def test_f1_comparison_detects_no_improvement(tmp_path):
    _write_metrics(tmp_path, "bert_only", accuracy=0.7, precision=0.7, recall=0.7, f1=0.80, auroc=0.75)
    _write_metrics(tmp_path, "resnet_only", accuracy=0.6, precision=0.6, recall=0.6, f1=0.60, auroc=0.65)
    _write_metrics(tmp_path, "late_fusion", accuracy=0.7, precision=0.7, recall=0.7, f1=0.75, auroc=0.80)
    comparison = f1_comparison(collect_ablation(tmp_path))
    assert comparison["improved"] is False


def test_f1_comparison_is_none_when_incomplete(tmp_path):
    assert f1_comparison(collect_ablation(tmp_path))["improved"] is None


def test_write_ablation_creates_csv_and_markdown(tmp_path):
    _write_metrics(tmp_path, "bert_only", accuracy=0.7, precision=0.7, recall=0.7, f1=0.70, auroc=0.75)
    _write_metrics(tmp_path, "resnet_only", accuracy=0.6, precision=0.6, recall=0.6, f1=0.60, auroc=0.65)
    _write_metrics(tmp_path, "late_fusion", accuracy=0.8, precision=0.8, recall=0.8, f1=0.78, auroc=0.85)

    result = write_ablation(
        output_dir=tmp_path,
        csv_path=tmp_path / "ablation_table.csv",
        markdown_path=tmp_path / "ablation.md",
        notes="테스트 코멘트",
    )
    csv_text = (tmp_path / "ablation_table.csv").read_text(encoding="utf-8")
    markdown = (tmp_path / "ablation.md").read_text(encoding="utf-8")
    assert "late_fusion" in csv_text and "0.78" in csv_text
    assert "BERT+Image (late fusion)" in markdown
    assert "개선됨" in markdown and "테스트 코멘트" in markdown
    assert result["comparison"]["improved"] is True


def test_render_markdown_marks_missing_rows(tmp_path):
    markdown = render_markdown(collect_ablation(tmp_path))
    assert "missing" in markdown and "—" in markdown


def test_custom_rows_extend_table(tmp_path):
    _write_metrics(tmp_path, "epic2_xattn", accuracy=0.9, precision=0.9, recall=0.9, f1=0.9, auroc=0.9)
    table = collect_ablation(tmp_path, rows=(AblationRow("+Cross-attention", "epic2_xattn"),))
    assert len(table) == 1 and table.loc[0, "f1"] == 0.9


def test_collect_ablation_script_main(tmp_path, capsys):
    import sys

    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
    from scripts.collect_ablation import main

    _write_metrics(tmp_path, "late_fusion", accuracy=0.8, precision=0.8, recall=0.8, f1=0.8, auroc=0.8)
    result = main(
        [
            "--output-dir",
            str(tmp_path),
            "--csv",
            str(tmp_path / "t.csv"),
            "--markdown",
            str(tmp_path / "t.md"),
        ]
    )
    assert (tmp_path / "t.csv").is_file() and (tmp_path / "t.md").is_file()
    assert list(result["table"]["status"]) == ["missing", "missing", "ok"]
