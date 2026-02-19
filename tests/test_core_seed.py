import numpy as np

from mlb.core.seed import set_seed


def test_set_seed_reproducible_numpy() -> None:
    set_seed(123)
    a = np.random.rand(5)
    set_seed(123)
    b = np.random.rand(5)
    assert np.allclose(a, b)
