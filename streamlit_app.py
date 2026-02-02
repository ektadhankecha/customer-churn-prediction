import streamlit as st
import pandas as pd
import joblib

MODEL_PATH = "models/churn_model.pkl"
FEATURES_PATH = "data/processed/feature_names.csv"
ENCODERS_PATH = "data/processed/label_encoders.pkl"

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📉",
    layout="centered"
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

model = load_model()
encoders = load_encoders()
feature_names = load_features()

st.set_page_config(page_title="Customer Churn Prediction", layout="centered")

st.title("📉 Customer Churn Prediction")
st.write("Fill in customer details to predict whether the customer is likely to leave the service.")

st.markdown("---")
st.subheader("🧾 Customer Information")

FIELD_META = {
    "Age": {
        "label": "Customer Age",
        "help": "Age of the customer in years"
    },
    "Gender": {
        "label": "Gender",
        "help": "Gender of the customer"
    },
    "Tenure": {
        "label": "Tenure (Months)",
        "help": "How long the customer has been using the service"
    },
    "Usage Frequency": {
        "label": "Usage Frequency",
        "help": "How frequently the customer uses the service"
    },
    "Support Calls": {
        "label": "Support Calls",
        "help": "Number of times the customer contacted customer support"
    },
    "Payment Delay": {
        "label": "Payment Delay",
        "help": "Number of times the customer delayed payments"
    },
    "Subscription Type": {
        "label": "Subscription Type",
        "help": "Type of subscription plan chosen by the customer"
    },
    "Contract Length": {
        "label": "Contract Length",
        "help": "Duration of the customer contract"
    },
    "Total Spend": {
        "label": "Total Spend",
        "help": "Total amount spent by the customer so far"
    },
    "Last Interaction": {
        "label": "Last Interaction (Days)",
        "help": "Days since the customer last interacted with the service"
    }
}


user_input = {}

for feature in feature_names:
    meta = FIELD_META.get(feature, {})
    label = meta.get("label", feature)
    help_text = meta.get("help", "Enter customer information")

    # 🔹 Categorical → dropdown
    if feature in encoders:
        options = list(encoders[feature].classes_)
        selected = st.selectbox(label, options, help=help_text)
        user_input[feature] = encoders[feature].transform([selected])[0]

    # 🔹 Integer-based numerical fields
    elif feature in [
        "Usage Frequency",
        "Support Calls",
        "Payment Delay",
        "Last Interaction"
    ]:
        user_input[feature] = st.number_input(
            label,
            min_value=0,
            value=0,
            step=1,
            help=help_text
        )

    # 🔹 Sliders for better UX
    elif feature == "Age":
        user_input[feature] = st.slider(label, 18, 100, 30, help=help_text)

    elif feature == "Tenure":
        user_input[feature] = st.slider(label, 0, 120, 12, help=help_text)

    elif feature == "Total Spend":
        user_input[feature] = st.slider(
            label,
            min_value=100,
            max_value=1000,
            value=500,
            step=1,
            help=help_text
        )

    # 🔹 Fallback (safe)
    else:
        user_input[feature] = st.number_input(
            label,
            min_value=0,
            value=0,
            step=1,
            help=help_text
        )

st.markdown("---")

if st.button("🔍 Predict Churn"):
    input_df = pd.DataFrame([user_input])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(
            f"⚠️ This customer is likely to churn.\n\n"
            f"Estimated probability: {probability:.2f}"
        )
    else:
        st.success(
            f"✅ This customer is not likely to churn.\n\n"
            f"Estimated probability: {probability:.2f}"
        )
