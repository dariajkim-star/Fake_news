"""DFDC part별 family 규모 계산 — 영상을 받기 전에 몇 part가 필요한지 결정한다.

독립 표본은 영상이 아니라 family(고유 REAL 원본)다. 이 스크립트는 metadata만으로
누적 family 수와 70/15/15 분할 시 test family 수를 추정한다.
"""
from __future__ import annotations
import io, json, sys
from collections import Counter

from huggingface_hub import hf_hub_download

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
REPO = "scarlettss/dfdc_metadata"
TEST_FRAC = 0.15
TARGET_TEST_FAMILIES = 30


def load(i: int) -> dict:
    p = hf_hub_download(REPO, f"dfdc_metadata/metadata_{i:02d}.json", repo_type="dataset")
    return json.load(open(p, encoding="utf-8"))


def main():
    n_parts = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    cum_fams: set[str] = set()
    cum_real = cum_fake = 0
    print(f"{'part':>4} {'videos':>7} {'REAL':>6} {'FAKE':>6} {'ratio':>7} "
          f"{'fam':>5} {'누적fam':>7} {'누적test':>8}")
    print("-" * 62)
    reached = None
    for i in range(n_parts):
        try:
            m = load(i)
        except Exception as e:
            print(f"{i:>4}  로드 실패: {type(e).__name__}")
            break
        lab = Counter(v["label"] for v in m.values())
        real, fake = lab.get("REAL", 0), lab.get("FAKE", 0)
        # family = 이 part 안의 고유 원본. REAL 영상 자체가 원본이다.
        fams = {k for k, v in m.items() if v["label"] == "REAL"}
        fams |= {v["original"] for v in m.values() if v.get("original")}
        cum_fams |= fams
        cum_real += real
        cum_fake += fake
        test_est = int(len(cum_fams) * TEST_FRAC)
        if reached is None and test_est >= TARGET_TEST_FAMILIES:
            reached = i
        ratio = f"{fake/real:.1f}:1" if real else "—"
        print(f"{i:>4} {len(m):>7} {real:>6} {fake:>6} {ratio:>7} "
              f"{len(fams):>5} {len(cum_fams):>7} {test_est:>8}")

    print("-" * 62)
    print(f"누적: 영상 {cum_real+cum_fake:,} (REAL {cum_real:,} / FAKE {cum_fake:,}), "
          f"family {len(cum_fams):,}")
    print(f"전체 불균형: {cum_fake/max(cum_real,1):.1f} : 1  "
          f"→ 전부 FAKE로 찍을 때 Accuracy {cum_fake/(cum_real+cum_fake)*100:.1f}%")
    if reached is not None:
        print(f"\n✅ test family {TARGET_TEST_FAMILIES}개 도달: part_00 ~ part_{reached:02d} "
              f"({reached+1}개 part, 약 {(reached+1)*10.6:.0f}GB)")
    else:
        print(f"\n⚠️ {n_parts}개 part로도 test family {TARGET_TEST_FAMILIES} 미달")

    # family당 FAKE 수 분포 — K개 상한 정책(K=3~5) 검토용
    m0 = load(0)
    per = Counter(v["original"] for v in m0.values() if v.get("original"))
    vals = sorted(per.values())
    if vals:
        print(f"\npart_00 family당 FAKE 수: 최소 {vals[0]} / 중앙 {vals[len(vals)//2]} / 최대 {vals[-1]}")
        for K in (3, 5):
            kept = sum(min(v, K) for v in vals)
            print(f"  K={K} 상한 시 FAKE {sum(vals):,} → {kept:,} "
                  f"(불균형 {kept/len(vals):.1f}:1, {kept/sum(vals)*100:.0f}% 유지)")


if __name__ == "__main__":
    main()
