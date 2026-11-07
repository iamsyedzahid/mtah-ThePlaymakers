import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mtah.synthetic import make_synthetic  # noqa: E402

N_TEST_FRAMES = 3_000


@pytest.fixture(scope="session")
def syn():
    return make_synthetic(N_TEST_FRAMES, seed=0)
