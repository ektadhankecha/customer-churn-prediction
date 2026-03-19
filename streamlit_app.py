import os

import joblib
import pandas as pd
import streamlit as st

from src.pipeline import ensure_artifacts

MODEL_PATH = "models/churn_model.pkl"
FEATURES_PATH = "data/processed/feature_names.csv"
ENCODERS_PATH = "data/processed/label_encoders.pkl"
SCALER_PATH = "data/processed/scaler.pkl"
SAMPLE_DATA_PATH = "data/sample/customer_churn_sample.csv"


st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📉",
    layout="centered",
)


def artifacts_ready() -> bool:
    return all(
        os.path.exists(p)
        for p in [
            MODEL_PATH,
            FEATURES_PATH,
            ENCODERS_PATH,
            SCALER_PATH,
        ]
    )


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_encoders():
    return joblib.load(ENCODERS_PATH)


@st.cache_data
def load_features():
    return pd.read_csv(FEATURES_PATH, header=None)[0].tolist()


@st.cache_resource
def load_scaler():
    return joblib.load(SCALER_PATH)


st.title("📉 Customer Churn Prediction")

if not artifacts_ready():
    st.warning(
        "Setup required: model files are missing. "
        "Click the button below to generate them automatically (uses a small bundled dataset)."
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("⚙️ Setup: Generate model files"):
            with st.spinner("Generating processed data, scaler, and model..."):
                ensure_artifacts()
            st.success("Setup completed. Reloading app...")
            st.cache_resource.clear()
            st.cache_data.clear()
            st.rerun()

    with col2:
        st.info(
            "Expected files:\n"
            "- models/churn_model.pkl\n"
            "- data/processed/feature_names.csv\n"
            "- data/processed/label_encoders.pkl\n"
            "- data/processed/scaler.pkl\n"
        )

    st.stop()


model = load_model()
encoders = load_encoders()
feature_names = load_features()
scaler = load_scaler()

st.write("Fill in customer details to predict whether the customer is likely to leave the service.")

st.markdown("---")
st.subheader("🧾 Customer Information")

with st.expander("✅ Quick self-test (optional)"):
    st.write("This checks that the model can load and produce predictions on sample rows.")
    if st.button("Run self-test"):
        try:
            sample_df = pd.read_csv(SAMPLE_DATA_PATH)
            for col in list(sample_df.columns):
                if "id" in col.lower():
                    sample_df = sample_df.drop(columns=[col])

            X_sample = sample_df.drop(columns=["Churn"])
            y_true = sample_df["Churn"].values

            for col, le in encoders.items():
                if col in X_sample.columns:
                    X_sample[col] = le.transform(X_sample[col])

            X_scaled = scaler.transform(X_sample[feature_names])
            X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names)
            preds = model.predict(X_scaled_df)
            acc = float((preds == y_true).mean())

            st.success(f"Self-test completed. Accuracy on bundled sample rows: {acc:.2f}")
            st.dataframe(pd.DataFrame({"y_true": y_true, "y_pred": preds}).head(10))
        except Exception as e:
            st.error(f"Self-test failed: {e}")


FIELD_META = {
    "Age": {"label": "Customer Age", "help": "Age of the customer in years"},
    "Gender": {"label": "Gender", "help": "Gender of the customer"},
    "Tenure": {"label": "Tenure (Months)", "help": "How long the customer has been using the service"},
    "Usage Frequency": {"label": "Usage Frequency", "help": "How frequently the customer uses the service"},
    "Support Calls": {"label": "Support Calls", "help": "Number of times the customer contacted customer support"},
    "Payment Delay": {"label": "Payment Delay", "help": "Number of times the customer delayed payments"},
    "Subscription Type": {"label": "Subscription Type", "help": "Type of subscription plan chosen by the customer"},
    "Contract Length": {"label": "Contract Length", "help": "Duration of the customer contract"},
    "Total Spend": {"label": "Total Spend", "help": "Total amount spent by the customer so far"},
    "Last Interaction": {"label": "Last Interaction (Days)", "help": "Days since the customer last interacted with the service"},
}


user_input = {}
for feature in feature_names:
    meta = FIELD_META.get(feature, {})
    label = meta.get("label", feature)
    help_text = meta.get("help", "Enter customer information")

    if feature in encoders:
        options = list(encoders[feature].classes_)
        selected = st.selectbox(label, options, help=help_text)
        user_input[feature] = encoders[feature].transform([selected])[0]
    elif feature in ["Usage Frequency", "Support Calls", "Payment Delay", "Last Interaction"]:
        user_input[feature] = st.number_input(label, min_value=0, value=0, step=1, help=help_text)
    elif feature == "Age":
        user_input[feature] = st.slider(label, 18, 100, 30, help=help_text)
    elif feature == "Tenure":
        user_input[feature] = st.slider(label, 0, 120, 12, help=help_text)
    elif feature == "Total Spend":
        user_input[feature] = st.slider(label, min_value=100, max_value=1000, value=500, step=1, help=help_text)
    else:
        user_input[feature] = st.number_input(label, min_value=0, value=0, step=1, help=help_text)


st.markdown("---")

if st.button("🔍 Predict Churn"):
    input_df = pd.DataFrame([user_input])
    input_scaled = scaler.transform(input_df)
    input_scaled_df = pd.DataFrame(input_scaled, columns=feature_names)
    prediction = int(model.predict(input_scaled_df)[0])
    probability = float(model.predict_proba(input_scaled_df)[0][1])

    if prediction == 1:
        st.error(
            f"**Prediction: 1** (Churn)\n\n"
            f"⚠️ This customer is likely to churn.\n\n"
            f"Churn probability: {probability:.2f}"
        )
    else:
        st.success(
            f"**Prediction: 0** (No churn)\n\n"
            f"✅ This customer is not likely to churn.\n\n"
            f"Churn probability: {probability:.2f}"
        )
