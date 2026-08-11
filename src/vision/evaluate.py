"""V0/V1 video-level 평가 + family 단위 bootstrap CI + paired 비교 (README §6).

원칙:
  - 모든 지표는 video 단위 (frame prob → median 집계). §6.1 함정 1.
  - bootstrap resampling 단위는 video가 아니라 **family**다. 같은 원본에서 파생된
    영상들은 독립 관측이 아니다. §6.1 함정 3.
  - V1 − V0은 같은 test 영상에 대한 paired bootstrap. CI가 0을 포함하면
    "개선 없음"으로 보고한다 — 사전에 못 박은 규칙 (§6.1).

사용:
    python -m src.vision.evaluate \
        --v0 models/deepfake_v0_d1crops.pt --v0-frames data/model/processed/frames \
        --v1 models/deepfake_v1_d1crops.pt --v1-frames data/model/processed/faces \
        --out results/vision_metrics_d1crops.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.vision.dataset import FrameDataset
from src.vision.train import DeepfakeClassifier

N_BOOT = 2000
SEED = 42


@torch.no_grad()
def video_scores(ckpt: Path, frames_root: Path, split_csv: Path):
    """체크포인트 → test 영상별 median P(fake) + 라벨 + family."""
    device = "cuda"
    model = DeepfakeClassifier().to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()
    ds = FrameDataset(split_csv, frames_root)
    loader = DataLoader(ds, batch_size=64, num_workers=4, pin_memory=True)
    probs, labels, fams = defaultdict(list), {}, {}
    for batch in loader:
        batch["image"] = batch["image"].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            p = torch.softmax(model(batch), dim=1)[:, 1]
        for v, f, pi, li in zip(batch["video"], batch["family"],
                                p.cpu().tolist(), batch["label"].tolist()):
            probs[v].append(pi)
            labels[v], fams[v] = li, f
    vids = sorted(probs)
    return (np.array([float(np.median(probs[v])) for v in vids]),
            np.array([labels[v] for v in vids]),
            np.array([fams[v] for v in vids]), vids)


def metrics(y, s) -> dict:
    from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                                 f1_score, recall_score, roc_auc_score)
    pred = (s >= 0.5).astype(int)
    return {
        "auroc": roc_auc_score(y, s),
        "auprc": average_precision_score(y, s),
        "macro_f1": f1_score(y, pred, average="macro"),
        "fake_recall": recall_score(y, pred, pos_label=1),
        "balanced_acc": balanced_accuracy_score(y, pred),
        "accuracy": float((pred == y).mean()),
    }


def family_bootstrap(y, s, fams, stat_fn, n=N_BOOT, seed=SEED):
    """family 단위 resampling. 한 family의 영상들은 함께 뽑히거나 함께 빠진다."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(fams)
    idx_of = {f: np.where(fams == f)[0] for f in uniq}
    out = []
    for _ in range(n):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_of[f] for f in pick])
        yy, ss = y[idx], s[idx]
        if len(np.unique(yy)) < 2:
            continue
        out.append(stat_fn(yy, ss))
    lo, hi = np.percentile(out, [2.5, 97.5])
    return float(lo), float(hi)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--v0", type=Path, required=True)
    ap.add_argument("--v0-frames", type=Path, required=True)
    ap.add_argument("--v1", type=Path, required=True)
    ap.add_argument("--v1-frames", type=Path, required=True)
    ap.add_argument("--split", type=Path, default=Path("data/model/splits/test.csv"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    from sklearn.metrics import roc_auc_score
    res: dict = {"split": str(args.split), "n_boot": N_BOOT, "seed": SEED,
                 "bootstrap_unit": "family"}

    scores = {}
    for tag, ckpt, root in (("v0", args.v0, args.v0_frames), ("v1", args.v1, args.v1_frames)):
        s, y, fams, vids = video_scores(ckpt, root, args.split)
        m = metrics(y, s)
        lo, hi = family_bootstrap(y, s, fams, lambda a, b: roc_auc_score(a, b))
        m["auroc_ci95"] = [round(lo, 4), round(hi, 4)]
        res[tag] = {k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()}
        scores[tag] = (s, y, fams, vids)
        print(f"{tag}: AUROC {m['auroc']:.4f} [{lo:.4f}, {hi:.4f}]  "
              f"AUPRC {m['auprc']:.4f}  FAKE recall {m['fake_recall']:.4f}")

    # paired V1 − V0: 동일 test 영상 정렬 확인 후 family 단위 paired bootstrap
    s0, y0, f0, v0ids = scores["v0"]
    s1, y1, f1, v1ids = scores["v1"]
    common = sorted(set(v0ids) & set(v1ids))
    i0 = [v0ids.index(v) for v in common]
    i1 = [v1ids.index(v) for v in common]
    s0c, s1c, yc, fc = s0[i0], s1[i1], y0[i0], f0[i0]
    delta = roc_auc_score(yc, s1c) - roc_auc_score(yc, s0c)

    rng = np.random.default_rng(SEED)
    uniq = np.unique(fc)
    idx_of = {f: np.where(fc == f)[0] for f in uniq}
    deltas = []
    for _ in range(N_BOOT):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_of[f] for f in pick])
        if len(np.unique(yc[idx])) < 2:
            continue
        deltas.append(roc_auc_score(yc[idx], s1c[idx]) - roc_auc_score(yc[idx], s0c[idx]))
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    improved = lo > 0
    res["paired_v1_minus_v0"] = {
        "videos": len(common), "delta_auroc": round(float(delta), 4),
        "ci95": [round(float(lo), 4), round(float(hi), 4)],
        "verdict": "개선" if improved else ("악화" if hi < 0 else "개선 없음 (CI가 0 포함)"),
    }
    print(f"paired ΔAUROC (V1−V0): {delta:+.4f} [{lo:+.4f}, {hi:+.4f}] "
          f"→ {res['paired_v1_minus_v0']['verdict']}")

    args.out.parent.mkdir(exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"저장: {args.out}")


if __name__ == "__main__":
    main()
