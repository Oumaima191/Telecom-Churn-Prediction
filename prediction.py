import pandas as pd
import joblib

# Load preprocessing
imputer = joblib.load("imputer.pkl")
scaler = joblib.load("scaler.pkl")

# Load models
model_xgb = joblib.load("xgb_model.pkl")
logreg = joblib.load("logreg_model.pkl")
rf = joblib.load("rf_model.pkl")

def predict_from_csv(input_csv, model_name="xgb"):
    new_data = pd.read_csv(input_csv)

    # Apply imputer
    new_data_imputed = imputer.transform(new_data)

    # Convert back to DataFrame with correct column names
    new_data = pd.DataFrame(new_data_imputed, columns=new_data.columns)

    # Choose model
    if model_name == "xgb":
        preds = model_xgb.predict(new_data)

    elif model_name == "rf":
        preds = rf.predict(new_data)

    elif model_name == "logreg":
        new_data_scaled = scaler.transform(new_data)
        preds = logreg.predict(new_data_scaled)

    else:
        raise ValueError("Unknown model name. Choose: xgb, logreg, rf")

    return preds

if __name__ == "__main__":
    print("XGB Predictions:", predict_from_csv("new_customers.csv", "xgb"))
    print("LogReg Predictions:", predict_from_csv("new_customers.csv", "logreg"))
    print("RF Predictions:", predict_from_csv("new_customers.csv", "rf"))
