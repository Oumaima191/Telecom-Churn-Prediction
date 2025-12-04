import streamlit as st
import pandas as pd
import joblib

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Telecom Churn Prediction",
    page_icon="📡",
    layout="wide"
)

# -------------------- HEADER --------------------
st.markdown("<h1 style='text-align:center;'>📡 Telecom Customer Churn Prediction Dashboard</h1>", unsafe_allow_html=True)
st.write("")

# -------------------- SIDEBAR --------------------
st.sidebar.header("⚙️ Configuration")
uploaded_file = st.sidebar.file_uploader("📥 Upload CSV file", type=["csv"])

model_choice = st.sidebar.selectbox(
    "🤖 Choose a Model",
    ["XGBoost", "Random Forest", "Logistic Regression"]
)

st.sidebar.markdown("---")
st.sidebar.info("🔎 The app will analyze your dataset and predict churn probability.")

# -------------------- MAIN CONTENT --------------------
if uploaded_file:
    st.success("📌 File uploaded successfully!")

    data = pd.read_csv(uploaded_file)

    # Show sample of dataset
    st.subheader("📊 Preview of Uploaded Data")
    st.dataframe(data.head(), use_container_width=True)

    # Load components
    imputer = joblib.load("imputer.pkl")
    scaler = joblib.load("scaler.pkl")

    model_xgb = joblib.load("xgb_model.pkl")
    rf = joblib.load("rf_model.pkl")
    logreg = joblib.load("logreg_model.pkl")

    # Preprocess
    data_imputed = imputer.transform(data)
    data_preprocessed = pd.DataFrame(data_imputed, columns=data.columns)

    # Predict
    if model_choice == "XGBoost":
        preds = model_xgb.predict(data_preprocessed)
        probs = model_xgb.predict_proba(data_preprocessed)[:, 1]

    elif model_choice == "Random Forest":
        preds = rf.predict(data_preprocessed)
        probs = rf.predict_proba(data_preprocessed)[:, 1]

    else:  # Logistic Regression
        scaled = scaler.transform(data_preprocessed)
        preds = logreg.predict(scaled)
        probs = logreg.predict_proba(scaled)[:, 1]

    data["Churn Prediction"] = preds
    data["Churn Probability (%)"] = (probs * 100).round(2)

    # ---------------- METRICS ----------------
    st.subheader("📈 Churn Summary")

    col1, col2 = st.columns(2)
    churn_rate = (preds.sum() / len(preds)) * 100

    col1.metric("🔴 % of Customers Likely to Churn", f"{churn_rate:.2f}%")
    col2.metric("🟢 Total Customers Analyzed", len(preds))

    # ---------------- RESULT TABLE ----------------
    st.subheader("🧾 Prediction Results")
    st.dataframe(data, use_container_width=True)

    # Highlight churn customers
    st.write("### 🔍 Highlight Churn Customers Only")
    churn_only = data[data["Churn Prediction"] == 1]

    if churn_only.empty:
        st.info("🎉 No customers predicted to churn.")
    else:
        st.error("⚠️ Customers at Risk of Churn")
        st.dataframe(churn_only, use_container_width=True)

    # ---------------- DOWNLOAD BUTTON ----------------
    st.download_button(
        "📥 Download Full Predictions",
        data.to_csv(index=False),
        "predictions.csv",
        mime="text/csv"
    )

else:
    st.warning("📁 Please upload a CSV file to begin analysis.")
