import pytest

from credit_risk import schema
from credit_risk.data import clean, validate
from tests.synthetic import make_raw_like, make_synthetic_frame


def test_clean_merges_undocumented_codes(cfg):
    out = clean(make_raw_like(), cfg)
    assert set(out[schema.EDUCATION].unique()) <= set(schema.CATEGORY_VALUES[schema.EDUCATION])
    assert set(out[schema.MARRIAGE].unique()) <= set(schema.CATEGORY_VALUES[schema.MARRIAGE])
    assert out.loc[1, schema.EDUCATION] == 4
    assert out.loc[4, schema.MARRIAGE] == 3


def test_clean_renames_columns(cfg):
    out = clean(make_raw_like(), cfg)
    assert "PAY_1" in out.columns and "PAY_0" not in out.columns
    assert out.index.name == "id"
    assert schema.TARGET in out.columns


def test_validate_accepts_clean_frame():
    validate(make_synthetic_frame())


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        (schema.SEX, 3, "SEX"),
        (schema.EDUCATION, 0, "EDUCATION"),
        (schema.AGE, 5, "AGE"),
        (schema.LIMIT, 0, "LIMIT_BAL"),
        ("PAY_2", 12, "PAY_2"),
        ("PAY_AMT1", -1, "negative"),
        (schema.TARGET, 2, "binary"),
    ],
)
def test_validate_rejects_bad_values(column, value, message):
    df = make_synthetic_frame()
    df.loc[df.index[0], column] = value
    with pytest.raises(ValueError, match=message):
        validate(df)


def test_validate_reports_missing_columns():
    df = make_synthetic_frame().drop(columns=["AGE"])
    with pytest.raises(ValueError, match="missing columns"):
        validate(df)
