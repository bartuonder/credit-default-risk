import copy

import pytest
from fastapi.testclient import TestClient

from credit_risk.api.main import create_app
from credit_risk.api.schemas import EXAMPLE_HIGH_RISK, EXAMPLE_LOW_RISK
from credit_risk.api.settings import Settings


@pytest.fixture(scope="module")
def client(artifacts):
    settings = Settings(
        model_path=artifacts / "model.joblib",
        model_card_path=artifacts / "model_card.json",
        max_batch_size=5,
        max_body_bytes=20_000,
    )
    with TestClient(create_app(settings)) as c:
        yield c


def payload(**overrides):
    body = copy.deepcopy(EXAMPLE_LOW_RISK)
    body.update(overrides)
    return body


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok" and r.json()["model_loaded"] is True


def test_model_info(client):
    body = client.get("/model-info").json()
    assert body["threshold"] == 0.25
    assert body["cost_assumption"]["false_negative"] == 5.0
    assert body["calibration_curve_test"]["count"] == [10]
    assert any(f["name"] == "AGE" for f in body["feature_schema"])


def test_predict_valid(client):
    r = client.post("/predict", json=EXAMPLE_LOW_RISK)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["default_probability"] <= 1
    assert body["decision"] in {"flag as likely default", "do not flag"}
    assert body["flagged"] == (body["default_probability"] >= body["threshold"])
    assert body["risk_band"] in {"low", "medium", "high"}
    assert "explanation" not in body


def test_high_risk_scores_above_low_risk(client):
    low = client.post("/predict", json=EXAMPLE_LOW_RISK).json()["default_probability"]
    high = client.post("/predict", json=EXAMPLE_HIGH_RISK).json()["default_probability"]
    assert high > low


def test_explanation_shape(client):
    r = client.post("/predict?explain=true&top_k=4", json=EXAMPLE_HIGH_RISK)
    assert r.status_code == 200
    body = r.json()
    assert len(body["explanation"]) == 4
    first = body["explanation"][0]
    assert {"feature", "label", "value", "contribution", "direction"} <= set(first)
    sizes = [abs(e["contribution"]) for e in body["explanation"]]
    assert sizes == sorted(sizes, reverse=True)
    assert body["explanation_unit"] == "log-odds"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("age", 17),
        ("age", 101),
        ("sex", 3),
        ("education", 0),
        ("marriage", 4),
        ("pay_1", 10),
        ("pay_3", -3),
        ("limit_bal", 0),
        ("pay_amt1", -1),
        ("bill_amt1", 10**9),
    ],
)
def test_out_of_range_values_are_rejected(client, field, value):
    r = client.post("/predict", json=payload(**{field: value}))
    assert r.status_code == 422
    error = r.json()["error"]
    assert error["code"] == "validation_error"
    assert error["details"][0]["field"] == field


@pytest.mark.parametrize(
    ("field", "value"),
    [("age", 18), ("age", 100), ("pay_1", -2), ("pay_1", 9), ("limit_bal", 1), ("pay_amt6", 0)],
)
def test_boundary_values_are_accepted(client, field, value):
    assert client.post("/predict", json=payload(**{field: value})).status_code == 200


def test_unknown_and_missing_fields(client):
    r = client.post("/predict", json=payload(extra_field=1))
    assert r.status_code == 422
    assert "extra_field" in r.json()["error"]["details"][0]["field"]
    body = payload()
    del body["age"]
    r = client.post("/predict", json=body)
    assert r.status_code == 422 and r.json()["error"]["details"][0]["field"] == "age"


def test_malformed_json_uses_error_format(client):
    r = client.post("/predict", content=b"{not json", headers={"content-type": "application/json"})
    assert r.status_code == 422
    assert set(r.json()["error"]) == {"code", "message", "details"}


def test_batch(client):
    r = client.post("/predict/batch", json={"applicants": [EXAMPLE_LOW_RISK, EXAMPLE_HIGH_RISK]})
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 2 and len(body["predictions"]) == 2
    assert (
        body["predictions"][1]["default_probability"]
        > body["predictions"][0]["default_probability"]
    )


def test_batch_with_explanations(client):
    r = client.post("/predict/batch?explain=true", json={"applicants": [EXAMPLE_HIGH_RISK] * 2})
    assert all(len(p["explanation"]) == 5 for p in r.json()["predictions"])


def test_batch_limits(client):
    assert (
        client.post("/predict/batch", json={"applicants": [EXAMPLE_LOW_RISK] * 5}).status_code
        == 200
    )
    too_many = client.post("/predict/batch", json={"applicants": [EXAMPLE_LOW_RISK] * 6})
    assert too_many.status_code == 422
    assert too_many.json()["error"]["code"] == "batch_too_large"
    assert client.post("/predict/batch", json={"applicants": []}).status_code == 422


def test_request_size_limit(client):
    r = client.post("/predict/batch", json={"applicants": [EXAMPLE_LOW_RISK] * 100})
    assert r.status_code == 413
    assert r.json()["error"]["code"] == "payload_too_large"


def test_unknown_route_and_method(client):
    assert client.get("/nope").json()["error"]["code"] == "not_found"
    assert client.get("/predict").status_code == 405


def test_request_id_header_and_no_payload_in_logs(client, caplog):
    with caplog.at_level("INFO", logger="credit_risk.access"):
        r = client.post("/predict", json=EXAMPLE_LOW_RISK)
    assert r.headers["x-request-id"]
    text = " ".join(rec.getMessage() + str(rec.__dict__) for rec in caplog.records)
    assert "limit_bal" not in text and "200000" not in text


def test_openapi_documents_examples(client):
    schema = client.get("/openapi.json").json()
    assert "/predict" in schema["paths"] and "/predict/batch" in schema["paths"]
    examples = schema["paths"]["/predict"]["post"]["requestBody"]["content"]["application/json"][
        "examples"
    ]
    assert {"low_risk", "high_risk"} <= set(examples)


def test_degraded_without_model(tmp_path):
    app = create_app(
        Settings(model_path=tmp_path / "x.joblib", model_card_path=tmp_path / "x.json")
    )
    with TestClient(app) as c:
        assert c.get("/health").json()["status"] == "degraded"
        r = c.post("/predict", json=EXAMPLE_LOW_RISK)
        assert r.status_code == 503 and r.json()["error"]["code"] == "http_error"


def test_cors_only_for_configured_origins(artifacts):
    settings = Settings(
        model_path=artifacts / "model.joblib",
        model_card_path=artifacts / "model_card.json",
        cors_origins=["https://example.org"],
    )
    with TestClient(create_app(settings)) as c:
        ok = c.get("/health", headers={"Origin": "https://example.org"})
        other = c.get("/health", headers={"Origin": "https://other.test"})
    assert ok.headers.get("access-control-allow-origin") == "https://example.org"
    assert "access-control-allow-origin" not in other.headers
