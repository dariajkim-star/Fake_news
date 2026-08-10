"""재현성 유틸 (NFR3).

동일 config로 두 번 실행하면 같은 metric이 나와야 하므로, 난수원 전체와
cuDNN 비결정 커널까지 함께 고정한다. DataLoader worker는 별도 프로세스라
`worker_init_fn`/`generator`로 seed를 명시적으로 전달해야 한다.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> int:
    """random / numpy / torch(cuda 포함) seed를 고정한다."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # 일부 연산은 결정적 커널이 없어 예외가 나므로 warn_only로 학습을 막지 않는다.
        torch.use_deterministic_algorithms(True, warn_only=True)
    return seed


def seed_worker(worker_id: int) -> None:
    """DataLoader worker의 python/numpy 난수를 고정한다 (`worker_init_fn`)."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def torch_generator(seed: int) -> torch.Generator:
    """DataLoader shuffle 순서를 고정하기 위한 generator."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator
