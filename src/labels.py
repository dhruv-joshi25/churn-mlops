"""Human-readable names for model feature columns.

SHAP reports raw encoded column names like `Contract_Month-to-month`. Those are
fine in a notebook and unacceptable in a UI a non-technical reader will look at,
so the mapping lives here rather than being duplicated wherever it is needed.

Phrases are written to complete the sentence "this customer ...", which is how
they read in the explanation list.
"""

# Numeric columns pass through the encoder unchanged.
_NUMERIC = {
    "tenure": "Length of time as a customer",
    "MonthlyCharges": "Monthly charges",
    "TotalCharges": "Total charges to date",
    "SeniorCitizen": "Senior citizen",
}

# One-hot columns arrive as "Column_Value". Only the phrasings that read badly
# when derived mechanically are listed; everything else falls through to the
# generic rule below.
_CATEGORICAL = {
    "Contract_Month-to-month": "Month-to-month contract",
    "Contract_One year": "One-year contract",
    "Contract_Two year": "Two-year contract",
    "InternetService_Fiber optic": "Fibre optic internet",
    "InternetService_DSL": "DSL internet",
    "InternetService_No": "No internet service",
    "PaymentMethod_Electronic check": "Pays by electronic check",
    "PaymentMethod_Mailed check": "Pays by mailed check",
    "PaymentMethod_Bank transfer (automatic)": "Pays by automatic bank transfer",
    "PaymentMethod_Credit card (automatic)": "Pays by automatic credit card",
    "PaperlessBilling_Yes": "Paperless billing",
    "PaperlessBilling_No": "Paper billing",
    "TechSupport_No": "No tech support",
    "TechSupport_Yes": "Has tech support",
    "OnlineSecurity_No": "No online security",
    "OnlineSecurity_Yes": "Has online security",
    "OnlineBackup_No": "No online backup",
    "OnlineBackup_Yes": "Has online backup",
    "DeviceProtection_No": "No device protection",
    "DeviceProtection_Yes": "Has device protection",
    "Partner_Yes": "Has a partner",
    "Partner_No": "No partner",
    "Dependents_Yes": "Has dependents",
    "Dependents_No": "No dependents",
    "StreamingTV_Yes": "Streams TV",
    "StreamingMovies_Yes": "Streams movies",
    "MultipleLines_Yes": "Multiple phone lines",
    "MultipleLines_No phone service": "No phone service",
    "PhoneService_No": "No phone service",
    "gender_Female": "Gender: female",
    "gender_Male": "Gender: male",
}


def readable(feature: str) -> str:
    """Turn an encoded column name into something a person can read.

    Unknown columns degrade to "Column: value" rather than raising, so a schema
    change shows up as slightly clumsy wording instead of a broken page.
    """
    if feature in _NUMERIC:
        return _NUMERIC[feature]
    if feature in _CATEGORICAL:
        return _CATEGORICAL[feature]
    if "_" in feature:
        column, _, value = feature.partition("_")
        return f"{column}: {value}"
    return feature
