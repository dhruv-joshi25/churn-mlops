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
