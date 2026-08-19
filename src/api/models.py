from typing import Literal

from pydantic import BaseModel, Field

from src.schema import CATEGORY_VALUES as C


class Customer(BaseModel):
    tenure: int = Field(ge=0, le=100, description="Months with the company")
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float | None = Field(default=None, ge=0)
    SeniorCitizen: Literal[0, 1] = 0

    gender: Literal["Female", "Male"]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]

    model_config = {
        "json_schema_extra": {
            "example": {
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
        }
    }


class Reason(BaseModel):
    feature: str
    contribution: float
    direction: Literal["increases_risk", "decreases_risk"]


class Prediction(BaseModel):
    churn_probability: float
    risk_band: Literal["Low", "Medium", "High"]
    will_churn: bool
    threshold: float
    # Where the cutoff came from: the model's own run, an env override, or the
    # fallback. Makes an accidental 0.5 visible instead of silent.
    threshold_source: str
    top_reasons: list[Reason] = []
    retention_recommendation: str | None = None
    model_version: str


class BatchRequest(BaseModel):
    customers: list[Customer] = Field(min_length=1, max_length=1000)


class BatchResponse(BaseModel):
    predictions: list[Prediction]


class Health(BaseModel):
    status: str
    model_loaded: bool
    model_version: str | None = None
    threshold: float | None = None
    threshold_source: str | None = None
