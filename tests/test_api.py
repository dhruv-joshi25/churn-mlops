from fastapi.testclient import TestClient

from churnkit.reference.api.main import app
from churnkit.reference.schema import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES

client = TestClient(app)

VALID = {
    "tenure": 2,
    "MonthlyCharges": 89.5,
    "TotalCharges": 179.0,
    "SeniorCitizen": 0,
    "gender": "Female",
    "Partner": "No",
    "Dependents": "No",
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
}


def test_health_responds():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] in {"ok", "degraded"}


def test_schema_has_no_duplicates():
    assert len(FEATURES) == len(set(FEATURES))
    assert not set(NUMERIC_FEATURES) & set(CATEGORICAL_FEATURES)


def test_request_example_covers_every_feature():
    assert set(VALID) == set(FEATURES)


def test_bad_category_rejected():
    bad = VALID | {"Contract": "Three year"}
    assert client.post("/predict", json=bad).status_code == 422


def test_negative_tenure_rejected():
    bad = VALID | {"tenure": -3}
    assert client.post("/predict", json=bad).status_code == 422


def test_predict_returns_200_or_503():
    # 503 when no model is registered yet; 200 once Sanjida's model lands.
    r = client.post("/predict", json=VALID)
    assert r.status_code in {200, 503}
    if r.status_code == 200:
        body = r.json()
        assert 0.0 <= body["churn_probability"] <= 1.0
        assert body["risk_band"] in {"Low", "Medium", "High"}


# ── /reload is a mutation, so it is guarded (ADR 0004) ────────────────────────
#
# There is no user authentication in this product and there is not meant to be
# (ADR 0001). These tests pin the two halves of a CSRF defence instead: an
# origin allowlist, and a custom header that a cross-origin page cannot set
# without a preflight the allowlist then refuses.

ADMIN_HEADER = "X-Churnkit-Admin"

PORTAL_ORIGIN = "http://localhost:8501"
HOSTILE_ORIGIN = "https://evil.example"


def test_reload_without_the_admin_header_is_refused():
    """A page the operator merely visits must not be able to swap the model."""
    r = client.post("/reload")
    assert r.status_code == 403
    assert ADMIN_HEADER.lower() in r.text.lower()


def test_reload_with_the_admin_header_gets_past_the_guard():
    # 503 when no model is registered, 200 once one is. Either answer proves
    # the guard let the request through, which is all this test asserts.
    r = client.post("/reload", headers={ADMIN_HEADER: "1"})
    assert r.status_code in {200, 503}


def test_the_guard_is_only_on_the_mutation():
    """Scoring stays open; locking it would break the portal for no gain."""
    assert client.get("/health").status_code == 200
    assert client.post("/predict", json=VALID).status_code in {200, 503}


def test_the_wildcard_origin_is_gone():
    r = client.get("/health", headers={"Origin": HOSTILE_ORIGIN})
    assert r.headers.get("access-control-allow-origin") != "*"


def test_an_unknown_origin_is_not_granted_cors_access():
    r = client.options(
        "/reload",
        headers={
            "Origin": HOSTILE_ORIGIN,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": ADMIN_HEADER,
        },
    )
    assert r.headers.get("access-control-allow-origin") not in {"*", HOSTILE_ORIGIN}


def test_the_portal_origin_is_granted_cors_access():
    """The allowlist has to admit the UI it ships with, or it is just an outage."""
    r = client.options(
        "/reload",
        headers={
            "Origin": PORTAL_ORIGIN,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": ADMIN_HEADER,
        },
    )
    assert r.headers.get("access-control-allow-origin") == PORTAL_ORIGIN
