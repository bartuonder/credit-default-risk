import pytest

from credit_risk.config import load_config
from tests.synthetic import make_synthetic_frame


@pytest.fixture(scope="module")
def frame():
    return make_synthetic_frame()


@pytest.fixture(scope="module")
def cfg():
    return load_config()
