import numpy as np

from credit_risk import schema
from credit_risk.split import make_split


def test_split_is_disjoint_and_complete(frame):
    train, test = make_split(frame, 0.2, 1)
    assert set(train).isdisjoint(test)
    assert len(train) + len(test) == len(frame)
    assert len(test) == round(0.2 * len(frame))


def test_split_is_stratified(frame):
    train, test = make_split(frame, 0.2, 1)
    overall = frame[schema.TARGET].mean()
    assert abs(frame.loc[train, schema.TARGET].mean() - overall) < 0.01
    assert abs(frame.loc[test, schema.TARGET].mean() - overall) < 0.01


def test_split_is_deterministic(frame):
    a = make_split(frame, 0.2, 7)
    b = make_split(frame, 0.2, 7)
    assert np.array_equal(a[1], b[1])
    assert not np.array_equal(a[1], make_split(frame, 0.2, 8)[1])
