"""DFDC family 단위 split 생성기.

핵심 원칙 — README §5.1 / §6.1:
  DFDC의 fake 영상은 특정 real 영상에서 파생된다. 파생 관계가 split을 가로지르면
  모델은 조작 흔적이 아니라 인물 얼굴과 배경을 외운다. 그러면 test AUROC가
  비현실적으로 높게 나오고 아무도 눈치채지 못한 채 "성공"으로 보고된다.
  따라서 **family(고유 원본) 전체를 하나의 단위로** train/val/test에 배정한다.

독립 표본은 영상이 아니라 family다. 이 스크립트는 영상 수와 family 수를 항상
함께 보고하며, leakage 검사를 통과하지 못하면 결과를 쓰지 않고 실패한다.

사용:
    python -m src.preprocess.build_splits --data-root data/model/raw/dfdc
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

# 내부 라벨 규약 (README §5.1). Fakeddit 시절과 달리 DFDC는 문자열 라벨이다.
LABEL_TO_INT = {"REAL": 0, "FAKE": 1}
SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}


@dataclass
class Family:
    """하나의 REAL 원본과 그로부터 파생된 FAKE들."""

    original: str
    real: str | None = None
    fakes: list[str] = field(default_factory=list)

    @property
    def size(self) -> int:
        return (1 if self.real else 0) + len(self.fakes)


def load_metadata(data_root: Path) -> dict[str, dict]:
    """data_root 아래 모든 part의 metadata.json을 병합한다.

    같은 파일명이 여러 part에 나타나면 DFDC 배포 오류이므로 즉시 실패시킨다.
    """
    merged: dict[str, dict] = {}
    paths = sorted(data_root.rglob("metadata.json"))
    if not paths:
        raise SystemExit(f"metadata.json을 찾지 못했다: {data_root}")

    for p in paths:
        part = p.relative_to(data_root).parts[0]
        with p.open(encoding="utf-8") as f:
            entries = json.load(f)
        for name, meta in entries.items():
            if name in merged and merged[name]["_part"] != part:
                raise SystemExit(f"중복 영상명 {name}: {merged[name]['_part']} vs {part}")
            merged[name] = {**meta, "_part": part}
        print(f"  {part}: {len(entries):,} entries")
    return merged


def build_families(meta: dict[str, dict], present: set[str] | None) -> dict[str, Family]:
    """metadata를 family 단위로 묶는다.

    present가 주어지면 실제로 디스크에 있는 영상만 사용한다(부분 다운로드 대응).
    original이 이 배포분에 없는 FAKE는 고아(orphan)로, 자기 original 이름을
    key로 하는 family에 남긴다 — REAL이 없을 뿐 leakage 위험은 없다.
    """
    fams: dict[str, Family] = {}
    skipped = 0
    for name, m in meta.items():
        if present is not None and name not in present:
            skipped += 1
            continue
        label = m["label"]
        if label == "REAL":
            fam = fams.setdefault(name, Family(original=name))
            fam.real = name
        elif label == "FAKE":
            origin = m.get("original")
            if not origin:
                skipped += 1
                continue
            fam = fams.setdefault(origin, Family(original=origin))
            fam.fakes.append(name)
        else:
            raise SystemExit(f"알 수 없는 라벨: {label!r} ({name})")
    if skipped:
        print(f"  제외된 영상 {skipped:,}개 (미다운로드 또는 original 누락)")
    return fams


def assign_splits(fams: dict[str, Family], seed: int) -> dict[str, str]:
    """family를 train/val/test에 배정한다.

    **분할 비율은 family 수에 적용한다.** 독립 표본이 family이기 때문이다.
    영상 수를 기준으로 나누면 family 크기 편차(1~36)와 train 전용 FAKE 상한이
    겹쳐 train 비중이 절반 아래로 떨어진다(실측: 46/27/27).

    **family 크기로 층화(stratify)한다.** 크기 내림차순으로 훑으면서 단순히
    "남은 자리가 많은 split"에 넣으면 train 쿼터가 먼저 차면서 큰 family가 전부
    train으로 몰리고 val/test는 최소 크기 family만 받는다(실측: val/test가 1:1).

    그래서 크기순으로 정렬한 뒤 20개 단위 블록마다 14/3/3 패턴을 섞어 배정한다.
    각 블록이 비슷한 크기의 family로 구성되므로, 모든 split이 크기 분포를 고르게 받는다.
    """
    rng = random.Random(seed)
    keys = sorted(fams)          # 파일시스템 순서에 의존하지 않도록 정렬 후
    rng.shuffle(keys)            # 고정 시드로 섞는다 (동일 크기 내 순서 무작위화)
    keys.sort(key=lambda k: fams[k].size, reverse=True)

    # 비율을 정수 패턴으로 환산 (70/15/15 → train 14, val 3, test 3 / 블록 20)
    block = 20
    pattern: list[str] = []
    for s, frac in SPLIT_FRACTIONS.items():
        pattern += [s] * round(block * frac)
    pattern += ["train"] * (block - len(pattern))   # 반올림 잔여는 train으로

    assignment: dict[str, str] = {}
    for i in range(0, len(keys), block):
        chunk = keys[i:i + block]
        slots = pattern[:len(chunk)]
        rng.shuffle(slots)       # 블록 내 위치 편향 제거
        assignment.update(zip(chunk, slots))
    return assignment


def cap_fakes(fam: Family, k: int | None, rng: random.Random) -> list[str]:
    """family당 FAKE 수를 k개로 제한한다 (train split 전용).

    한 원본에서 파생된 36개 fake는 독립 관측 36건이 아니다. 상한은 학습시간·
    family 편중·FAKE 클래스 지배를 동시에 줄인다. test/val에는 적용하지 않는다 —
    평가 대상 분포를 인위적으로 바꾸면 지표의 의미가 사라진다.
    """
    if k is None or len(fam.fakes) <= k:
        return list(fam.fakes)
    return rng.sample(sorted(fam.fakes), k)


def verify_no_leakage(rows: list[dict]) -> None:
    """family가 split을 가로지르지 않는지, 영상이 중복 배정되지 않았는지 검사한다."""
    fam_splits: dict[str, set[str]] = defaultdict(set)
    video_splits: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        fam_splits[r["family"]].add(r["split"])
        video_splits[r["video"]].add(r["split"])

    bad_fams = {f: s for f, s in fam_splits.items() if len(s) > 1}
    bad_vids = {v: s for v, s in video_splits.items() if len(s) > 1}
    if bad_fams:
        sample = list(bad_fams.items())[:3]
        raise SystemExit(f"LEAKAGE: family가 여러 split에 걸쳐 있다 — {sample}")
    if bad_vids:
        sample = list(bad_vids.items())[:3]
        raise SystemExit(f"LEAKAGE: 영상이 중복 배정됐다 — {sample}")
    print("  ✅ leakage 검사 통과 (family·영상 모두 단일 split)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, default=Path("data/model/raw/dfdc"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/model/splits"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fake-cap", type=int, default=5,
                    help="train split의 family당 FAKE 상한 (0이면 무제한)")
    ap.add_argument("--require-files", action="store_true",
                    help="디스크에 실제로 존재하는 mp4만 사용 (부분 다운로드 대응)")
    args = ap.parse_args()

    print("metadata 병합:")
    meta = load_metadata(args.data_root)

    present = None
    if args.require_files:
        present = {p.name for p in args.data_root.rglob("*.mp4")}
        print(f"  디스크상 mp4 {len(present):,}개")

    fams = build_families(meta, present)
    n_videos = sum(f.size for f in fams.values())
    print(f"\nfamily {len(fams):,}개 / 영상 {n_videos:,}개")
    if not fams:
        raise SystemExit("family가 하나도 만들어지지 않았다")

    assignment = assign_splits(fams, args.seed)
    rng = random.Random(args.seed)
    cap = args.fake_cap or None

    rows: list[dict] = []
    for key, fam in sorted(fams.items()):
        split = assignment[key]
        fakes = cap_fakes(fam, cap, rng) if split == "train" else list(fam.fakes)
        videos = ([fam.real] if fam.real else []) + fakes
        for v in videos:
            rows.append({
                "video": v,
                "part": meta[v]["_part"],
                "label": LABEL_TO_INT[meta[v]["label"]],
                "label_str": meta[v]["label"],
                "family": key,
                "split": split,
            })

    print("\nleakage 검사:")
    verify_no_leakage(rows)

    # ── 리포트 ────────────────────────────────────────────────────────────
    report: dict = {
        "seed": args.seed,
        "fake_cap_train": cap,
        "source_parts": sorted({m["_part"] for m in meta.values()}),
        "totals": {"families": len(fams), "videos_before_cap": n_videos,
                   "videos_after_cap": len(rows)},
        "splits": {},
    }
    print(f"\n{'split':>6} {'family':>8} {'video':>7} {'REAL':>6} {'FAKE':>6} {'ratio':>8}")
    print("-" * 48)
    for s in SPLIT_FRACTIONS:
        sub = [r for r in rows if r["split"] == s]
        real = sum(1 for r in sub if r["label"] == 0)
        fake = len(sub) - real
        nf = len({r["family"] for r in sub})
        ratio = f"{fake/real:.1f}:1" if real else "—"
        print(f"{s:>6} {nf:>8} {len(sub):>7} {real:>6} {fake:>6} {ratio:>8}")
        report["splits"][s] = {"families": nf, "videos": len(sub),
                               "real": real, "fake": fake}

    # 재현성: split 자체의 해시를 남겨 이후 실험이 같은 분할인지 대조할 수 있게 한다.
    digest = hashlib.sha256(
        json.dumps([(r["video"], r["split"]) for r in sorted(rows, key=lambda x: x["video"])],
                   ensure_ascii=False).encode()
    ).hexdigest()
    report["split_sha256"] = digest

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for s in SPLIT_FRACTIONS:
        sub = [r for r in rows if r["split"] == s]
        with (args.out_dir / f"{s}.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["video", "part", "label", "label_str", "family"])
            w.writeheader()
            w.writerows({k: r[k] for k in w.fieldnames} for r in sub)

    with (args.out_dir / "split_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\nsplit_sha256: {digest[:16]}…")
    print(f"저장 완료 → {args.out_dir}/")


if __name__ == "__main__":
    main()
