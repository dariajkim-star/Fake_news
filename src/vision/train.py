"""EfficientNet-B0 deepfake 분류 학습 (V0 full-frame / V1 face-crop).

§8.2 스케줄: 1단계 backbone freeze(head만, lr 1e-3) → 2단계 상위 블록 unfreeze(lr 1e-4).
AMP, early stopping(val video-level AUROC), seed 42.
모델 계약: model(batch) -> logits [B, 2].

사용:
    python -m src.vision.train --variant v0 --frames data/model/processed/frames
    python -m src.vision.train --variant v1 --frames data/model/processed/faces
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.vision.dataset import FrameDataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class DeepfakeClassifier(nn.Module):
    """model(batch) -> logits [B, 2] 계약 준수 래퍼."""

    def __init__(self):
        super().__init__()
        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
        self.net = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.net.classifier[1] = nn.Linear(self.net.classifier[1].in_features, 2)

    def forward(self, batch: dict) -> torch.Tensor:
        return self.net(batch["image"])

    def set_stage(self, stage: int) -> None:
        if stage == 1:                                  # head만 학습
            for p in self.net.features.parameters():
                p.requires_grad = False
        else:                                           # 상위 블록 unfreeze
            for p in self.net.features[-3:].parameters():
                p.requires_grad = True


@torch.no_grad()
def video_auroc(model, loader, device) -> tuple[float, dict]:
    """frame prob → video median → AUROC (§6.1: 평가는 video 단위)."""
    from sklearn.metrics import roc_auc_score
    model.eval()
    probs, labels = defaultdict(list), {}
    for batch in loader:
        batch["image"] = batch["image"].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            p = torch.softmax(model(batch), dim=1)[:, 1]
        for v, pi, li in zip(batch["video"], p.cpu().tolist(), batch["label"].tolist()):
            probs[v].append(pi)
            labels[v] = li
    vids = sorted(probs)
    y = [labels[v] for v in vids]
    s = [float(np.median(probs[v])) for v in vids]
    return roc_auc_score(y, s), {v: float(np.median(probs[v])) for v in vids}


def run_stage(model, loader, val_loader, device, epochs, lr, scaler, tag) -> float:
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr)
    crit = nn.CrossEntropyLoss()
    best = 0.0
    for ep in range(epochs):
        model.train()
        t0, tot = time.time(), 0.0
        for batch in loader:
            batch["image"] = batch["image"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = crit(model(batch), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            tot += loss.item()
        auc, _ = video_auroc(model, val_loader, device)
        best = max(best, auc)
        print(f"  [{tag} ep{ep+1}/{epochs}] loss {tot/len(loader):.4f} "
              f"val vAUROC {auc:.4f} ({time.time()-t0:.0f}s)", flush=True)
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", choices=["v0", "v1"], required=True)
    ap.add_argument("--frames", type=Path, required=True)
    ap.add_argument("--splits", type=Path, default=Path("data/model/splits"))
    ap.add_argument("--out", type=Path, default=Path("models"))
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--stage1-epochs", type=int, default=2)
    ap.add_argument("--stage2-epochs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda"
    tr = FrameDataset(args.splits / "train.csv", args.frames, train=True)
    va = FrameDataset(args.splits / "val.csv", args.frames)
    print(f"{args.variant}: train {len(tr):,} frames / val {len(va):,} frames "
          f"(root: {args.frames})")
    dl = dict(num_workers=4, pin_memory=True, persistent_workers=True)
    tr_l = DataLoader(tr, batch_size=args.batch, shuffle=True, **dl)
    va_l = DataLoader(va, batch_size=args.batch, **dl)

    model = DeepfakeClassifier().to(device)
    scaler = torch.amp.GradScaler("cuda")

    model.set_stage(1)
    run_stage(model, tr_l, va_l, device, args.stage1_epochs, 1e-3, scaler, "s1")
    model.set_stage(2)
    best = run_stage(model, tr_l, va_l, device, args.stage2_epochs, 1e-4, scaler, "s2")

    name = f"deepfake_{args.variant}{('_'+args.tag) if args.tag else ''}"
    ckpt = args.out / f"{name}.pt"
    torch.save(model.state_dict(), ckpt)
    meta = {"variant": args.variant, "frames_root": str(args.frames), "seed": args.seed,
            "batch": args.batch, "best_val_video_auroc": round(best, 4),
            "epochs": [args.stage1_epochs, args.stage2_epochs]}
    Path("results").mkdir(exist_ok=True)
    (Path("results") / f"{name}_train.json").write_text(json.dumps(meta, indent=2))
    print(f"저장: {ckpt}  (best val vAUROC {best:.4f})")


if __name__ == "__main__":
    main()
