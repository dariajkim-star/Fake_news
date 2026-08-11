"""DFDC frame 데이터셋 (V0 full-frame / V1 face-crop 공용).

split CSV(video, label, family)를 읽고, 영상 디렉토리의 프레임 jpg를 frame 단위
샘플로 편다. 라벨은 video 라벨을 상속한다. 평가는 반드시 video 단위로 집계해야
하므로(§6.1 함정 1) 각 샘플에 video id를 같이 반환한다.
"""
from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class FrameDataset(Dataset):
    def __init__(self, split_csv: Path, frames_root: Path, size: int = 224,
                 train: bool = False):
        self.size = size
        self.train = train
        self.items: list[tuple[Path, int, str, str]] = []   # (jpg, label, video, family)
        with open(split_csv, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                stem = Path(r["video"]).stem
                vdir = frames_root / stem
                if not vdir.is_dir():
                    continue
                for jpg in sorted(vdir.glob("f_*.jpg")):
                    self.items.append((jpg, int(r["label"]), stem, r["family"]))
        if not self.items:
            raise RuntimeError(f"샘플 0개: {split_csv} × {frames_root}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        path, label, video, family = self.items[i]
        img = cv2.imread(str(path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.shape[0] != self.size or img.shape[1] != self.size:
            img = cv2.resize(img, (self.size, self.size))
        if self.train and np.random.rand() < 0.5:           # 수평 반전만 (§8.1)
            img = img[:, ::-1]
        x = (img.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        return {
            "image": torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))),
            "label": torch.tensor(label),
            "video": video,
            "family": family,
        }
