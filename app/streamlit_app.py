"""Churn prediction portal.

This UI has no model of its own. Every prediction is an HTTP call to the API,
which is the whole point of the architecture: one model, loaded in one place,
with one threshold. A UI that imported the pipeline directly would be a second
copy that could silently drift from the served one.
"""

import io
import os

import pandas as pd
import requests
import streamlit as st

from src.labels import readable
from src.schema import CATEGORY_VALUES, FEATURES

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")
TIMEOUT = 30
BATCH_LIMIT = 1000  # matches BatchRequest's max_length

st.set_page_config(page_title="Churn Prediction", page_icon="📉", layout="wide")


# ── API helpers ───────────────────────────────────────────────────────────────


@st.cache_data(ttl=10, show_spinner=False)
def get_health() -> dict | None:
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        r.raise_for_status()
        return r.json()
    except requests.RequestException:
        return None


def post_predict(payload: dict, explain: bool = True) -> dict:
    r = requests.post(
        f"{API_URL}/predict", json=payload, params={"explain": explain}, timeout=TIMEOUT
    )
    r.raise_for_status()
    return r.json()


def post_batch(rows: list[dict], explain: bool = False) -> list[dict]:
    """Chunked so an upload larger than the API's cap still works."""
    out = []
    for start in range(0, len(rows), BATCH_LIMIT):
        chunk = rows[start : start + BATCH_LIMIT]
        r = requests.post(
            f"{API_URL}/predict/batch",
            json={"customers": chunk},
            params={"explain": explain},
            timeout=TIMEOUT * 4,
        )
        r.raise_for_status()
        out.extend(r.json()["predictions"])
    return out


# ── Shared rendering ──────────────────────────────────────────────────────────

BAND_COLOUR = {"High": "#c0392b", "Medium": "#c87f0a", "Low": "#1e7e34"}


def render_prediction(result: dict) -> None:
    prob = result["churn_probability"]
    band = result["risk_band"]

    left, right = st.columns([1, 2])

    with left:
        st.metric("Churn probability", f"{prob:.1%}")
        st.markdown(
            f"<div style='padding:.5rem 1rem;border-radius:6px;color:#fff;"
            f"background:{BAND_COLOUR.get(band, '#555')};display:inline-block;"
            f"font-weight:600'>{band} risk</div>",
            unsafe_allow_html=True,
        )
        st.progress(min(max(prob, 0.0), 1.0))
        st.caption(
            f"Flagged at a threshold of {result['threshold']:.2f} — "
            f"{'will churn' if result['will_churn'] else 'will not churn'}"
        )

    with right:
        reasons = result.get("top_reasons") or []
        if reasons:
            st.markdown("**Why the model says this**")
            df = pd.DataFrame(
                [
                    {
                        "Factor": readable(r["feature"]),
                        "Effect": r["contribution"],
                        "Direction": (
                            "▲ increases risk"
                            if r["direction"] == "increases_risk"
                            else "▼ reduces risk"
                        ),
                    }
                    for r in reasons
                ]
            )
            st.dataframe(
                df,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Effect": st.column_config.NumberColumn(format="%.3f")
                },
            )
        else:
            st.info("No explanation returned for this prediction.")

    rec = result.get("retention_recommendation")
    if rec:
        st.success(f"**Suggested action:** {rec}")

    st.caption(
        f"Model: `{result['model_version']}` · "
        f"threshold from {result.get('threshold_source', 'unknown')}"
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Service")
    st.caption(f"`{API_URL}`")

    health = get_health()
    if health is None:
        st.error("API unreachable")
        st.caption("Start it with `make api`, or set `API_URL`.")
    elif not health.get("model_loaded"):
        st.warning("API up, no model loaded")
    else:
        st.success("API healthy")
        st.caption(f"Model: `{health.get('model_version')}`")
        st.caption(f"Threshold: **{health.get('threshold')}**")
        st.caption(f"Source: {health.get('threshold_source')}")

    if st.button("Reload model", use_container_width=True):
        try:
            r = requests.post(f"{API_URL}/reload", timeout=TIMEOUT)
            r.raise_for_status()
            get_health.clear()
            st.success(f"Now serving {r.json()['model_version']}")
        except requests.RequestException as exc:
            st.error(f"Reload failed: {exc}")

    st.divider()
    st.caption(
        "Recommendations are hypotheses drawn from correlational SHAP "
        "attributions, not proven causes of churn. Validating them needs an "
        "A/B test."
    )


# ── Main ──────────────────────────────────────────────────────────────────────

st.title("Customer churn prediction")

single_tab, batch_tab = st.tabs(["Single customer", "Upload a CSV"])


with single_tab:
    with st.form("customer"):
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("**Account**")
            tenure = st.number_input("Tenure (months)", 0, 100, 2)
            monthly = st.number_input("Monthly charges", 0.0, 500.0, 95.7, step=0.05)
            total = st.number_input("Total charges", 0.0, 100_000.0, 191.4, step=0.05)
            contract = st.selectbox("Contract", CATEGORY_VALUES["Contract"])
            payment = st.selectbox("Payment method", CATEGORY_VALUES["PaymentMethod"])
            paperless = st.selectbox(
                "Paperless billing", CATEGORY_VALUES["PaperlessBilling"]
            )

        with c2:
            st.markdown("**Services**")
            internet = st.selectbox(
                "Internet service", CATEGORY_VALUES["InternetService"]
            )
            phone = st.selectbox("Phone service", CATEGORY_VALUES["PhoneService"])
            lines = st.selectbox("Multiple lines", CATEGORY_VALUES["MultipleLines"])
            security = st.selectbox(
                "Online security", CATEGORY_VALUES["OnlineSecurity"]
            )
            backup = st.selectbox("Online backup", CATEGORY_VALUES["OnlineBackup"])
            protection = st.selectbox(
                "Device protection", CATEGORY_VALUES["DeviceProtection"]
            )

        with c3:
            st.markdown("**Household & extras**")
            support = st.selectbox("Tech support", CATEGORY_VALUES["TechSupport"])
            tv = st.selectbox("Streaming TV", CATEGORY_VALUES["StreamingTV"])
            movies = st.selectbox(
                "Streaming movies", CATEGORY_VALUES["StreamingMovies"]
            )
            gender = st.selectbox("Gender", CATEGORY_VALUES["gender"])
            partner = st.selectbox("Partner", CATEGORY_VALUES["Partner"])
            dependents = st.selectbox("Dependents", CATEGORY_VALUES["Dependents"])
            senior = st.selectbox("Senior citizen", [0, 1])

        submitted = st.form_submit_button("Predict", type="primary")

    if submitted:
        payload = {
            "tenure": int(tenure),
            "MonthlyCharges": float(monthly),
            "TotalCharges": float(total),
            "SeniorCitizen": int(senior),
            "gender": gender,
            "Partner": partner,
            "Dependents": dependents,
            "PhoneService": phone,
            "MultipleLines": lines,
            "InternetService": internet,
            "OnlineSecurity": security,
            "OnlineBackup": backup,
            "DeviceProtection": protection,
            "TechSupport": support,
            "StreamingTV": tv,
            "StreamingMovies": movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
        }
        try:
            with st.spinner("Scoring…"):
                render_prediction(post_predict(payload))
        except requests.HTTPError as exc:
            st.error(f"API rejected the request ({exc.response.status_code})")
            st.code(exc.response.text)
        except requests.RequestException as exc:
            st.error(f"Could not reach the API: {exc}")


with batch_tab:
    st.markdown(
        "Upload a CSV with the raw Telco columns. `customerID` and `Churn` are "
        "ignored if present, so the original dataset works unmodified."
    )
    uploaded = st.file_uploader("CSV file", type=["csv"])
    explain_batch = st.checkbox(
        "Include explanations (slower)",
        value=False,
        help="SHAP runs per row, so this is left off for large files.",
    )

    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.caption(f"{len(df):,} rows read")

        missing = [c for c in FEATURES if c not in df.columns]
        if missing:
            st.error(f"Missing required columns: {', '.join(missing)}")
        else:
            frame = df[FEATURES].copy()
            # Blank TotalCharges is the dataset's known quirk; the pipeline
            # imputes it, so send null rather than dropping the row.
            frame["TotalCharges"] = pd.to_numeric(
                frame["TotalCharges"], errors="coerce"
            )
            rows = frame.where(pd.notna(frame), None).to_dict("records")

            if st.button("Score all rows", type="primary"):
                try:
                    with st.spinner(f"Scoring {len(rows):,} customers…"):
                        preds = post_batch(rows, explain=explain_batch)
                except requests.RequestException as exc:
                    st.error(f"Batch scoring failed: {exc}")
                else:
                    scored = df.copy()
                    scored["churn_probability"] = [
                        p["churn_probability"] for p in preds
                    ]
                    scored["risk_band"] = [p["risk_band"] for p in preds]
                    scored["will_churn"] = [p["will_churn"] for p in preds]
                    scored["recommendation"] = [
                        p.get("retention_recommendation") for p in preds
                    ]
                    scored = scored.sort_values(
                        "churn_probability", ascending=False
                    )

                    flagged = int(scored["will_churn"].sum())
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Customers scored", f"{len(scored):,}")
                    m2.metric(
                        "Flagged as at-risk",
                        f"{flagged:,}",
                        f"{flagged / len(scored):.1%} of file",
                    )
                    m3.metric(
                        "High risk band",
                        f"{int((scored['risk_band'] == 'High').sum()):,}",
                    )

                    st.dataframe(scored, use_container_width=True, height=420)

                    buf = io.StringIO()
                    scored.to_csv(buf, index=False)
                    st.download_button(
                        "Download scored CSV",
                        buf.getvalue(),
                        file_name="churn_scored.csv",
                        mime="text/csv",
                    )
