import pandas as pd
import pytest

from credit_risk import schema
from credit_risk.features import ENGINEERED_BLOCKS, FeatureBuilder, column_plan, engineer


def _row(**overrides):
    base = dict.fromkeys(schema.RAW_FEATURES, 0)
    base.update({schema.LIMIT: 100_000, schema.SEX: 1, schema.EDUCATION: 2, schema.MARRIAGE: 1})
    base.update(overrides)
    return pd.DataFrame([base])


def test_engineering_is_row_independent(frame):
    X = frame.drop(columns=schema.TARGET)
    full = engineer(X)
    single = engineer(X.iloc[[10]])
    pd.testing.assert_frame_equal(full.iloc[[10]], single)


def test_utilisation_and_delay_counts():
    X = _row(
        BILL_AMT1=50_000, BILL_AMT2=100_000, PAY_1=2, PAY_2=1, PAY_3=0, PAY_4=-1, PAY_5=-2, PAY_6=0
    )
    out = engineer(X).iloc[0]
    assert out["util_1"] == pytest.approx(0.5)
    assert out["util_max"] == pytest.approx(1.0)
    assert out["n_delayed"] == 2
    assert out["max_delay"] == 2
    assert out["mean_delay"] == pytest.approx(3 / 6)


def test_status_trend_sign():
    worsening = _row(PAY_6=-1, PAY_5=-1, PAY_4=0, PAY_3=1, PAY_2=2, PAY_1=3)
    improving = _row(PAY_6=3, PAY_5=2, PAY_4=1, PAY_3=0, PAY_2=-1, PAY_1=-1)
    assert engineer(worsening)["status_trend"].iloc[0] > 0
    assert engineer(improving)["status_trend"].iloc[0] < 0


def test_payment_ratio_uses_previous_bill_and_is_bounded():
    X = _row(BILL_AMT2=1000, PAY_AMT1=250, BILL_AMT3=0, PAY_AMT2=500)
    out = engineer(X).iloc[0]
    assert out["payratio_1"] == pytest.approx(0.25)
    assert 0.0 <= out["payratio_total"] <= 1.0
    paid_more = _row(BILL_AMT2=100, PAY_AMT1=900)
    assert engineer(paid_more)["payratio_1"].iloc[0] == 1.0


def test_column_plan_drops_group_with_engineered_columns(cfg):
    full = column_plan(cfg)
    reduced = column_plan(cfg, drop_groups=("repayment_status",))
    assert "PAY_1" in full["columns"] and "PAY_1" not in reduced["columns"]
    assert "n_delayed" in full["columns"] and "n_delayed" not in reduced["columns"]
    assert len(reduced["columns"]) < len(full["columns"])


def test_feature_builder_output_columns(cfg, frame):
    plan = column_plan(cfg)
    out = FeatureBuilder(tuple(plan["columns"]), tuple(plan["engineered"])).transform(
        frame.drop(columns=schema.TARGET)
    )
    assert list(out.columns) == plan["columns"]
    expected = {c for cols in ENGINEERED_BLOCKS.values() for c in cols}
    assert expected <= set(out.columns)
    assert not out.isna().any().any()
