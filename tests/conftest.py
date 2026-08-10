"""pytest가 repo 루트를 import 경로로 인식하게 한다 (`src.*` 절대 import 사용)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402
import pytest  # noqa: E402

# ---------------------------------------------------------------------------
# Story 1.2 — 합성(synthetic) Fakeddit 픽스처.
# 실데이터는 로컬/CI에 없으므로 전처리·분할·필터·Dataset 로직은 모두
# 아래 합성 manifest/이미지로 검증한다.
# ---------------------------------------------------------------------------

SYNTHETIC_TITLES = [
    "tesla stock soars after earnings call",  # financial
    "cute puppy sleeping on a couch",
    "ceo announces merger with rival bank",  # financial
    "restocking the shelves at the local store",  # 'stock' 부분문자열 — 금융 아님
    "bitcoin price hits new record",  # financial
    "homemade bread recipe goes viral",
    "ipo filing reveals revenue growth",  # financial
    "sunset over the mountains",
    "investor confidence drops sharply",  # financial
    "a cat wearing tiny sunglasses",
]


@pytest.fixture
def synthetic_raw() -> pd.DataFrame:
    """Fakeddit 원본 tsv 형태의 합성 DataFrame (`2_way_label` 원본 인코딩 사용)."""
    rows = []
    for index, title in enumerate(SYNTHETIC_TITLES):
        rows.append(
            {
                "id": f"s{index:03d}",
                "clean_title": title,
                # 원본 인코딩: 1=true(real), 0=fake — 짝수는 real, 홀수는 fake
                "2_way_label": 1 if index % 2 == 0 else 0,
                "image_url": f"https://example.invalid/{index}.jpg",
                "hasImage": True,
            }
        )
    return pd.DataFrame(rows)


def _write_image(path: Path, color: tuple[int, int, int] = (128, 64, 32)) -> Path:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (40, 30), color=color).save(path)
    return path


@pytest.fixture
def synthetic_image_dir(tmp_path: Path) -> Path:
    """모든 합성 샘플에 대응하는 정상 jpg 이미지를 만든다."""
    image_dir = tmp_path / "images"
    for index in range(len(SYNTHETIC_TITLES)):
        _write_image(image_dir / f"s{index:03d}.jpg", color=(index * 20 % 256, 100, 50))
    return image_dir


@pytest.fixture
def write_image():
    """테스트가 임의 위치에 정상 이미지를 만들 수 있게 하는 헬퍼."""
    return _write_image


@pytest.fixture
def synthetic_manifest(synthetic_raw: pd.DataFrame, synthetic_image_dir: Path) -> pd.DataFrame:
    """전처리를 통과한 클린 manifest (내부 라벨 규약 0=REAL, 1=FAKE)."""
    from src.data.preprocess import build_manifest

    manifest, _ = build_manifest(synthetic_raw, image_dir=synthetic_image_dir)
    return manifest


class FakeTokenizer:
    """네트워크 없이 Dataset 토큰화 경로를 검증하기 위한 가짜 tokenizer.

    HuggingFace tokenizer의 호출 규약(`__call__` -> dict, `pad_token_id`) 중
    Dataset이 실제로 쓰는 부분만 흉내낸다.
    """

    pad_token_id = 0

    def __call__(
        self,
        text: str,
        max_length: int = 128,
        truncation: bool = True,
        padding: bool | str = False,
        return_tensors=None,
    ) -> dict[str, list[int]]:
        # 단어당 토큰 1개 + [CLS]/[SEP] 흉내 (id는 1~99로 결정적 해싱)
        tokens = [1] + [(sum(map(ord, word)) % 98) + 2 for word in str(text).split()] + [1]
        if truncation:
            tokens = tokens[: int(max_length)]
        return {"input_ids": tokens, "attention_mask": [1] * len(tokens)}


@pytest.fixture
def fake_tokenizer() -> FakeTokenizer:
    return FakeTokenizer()
