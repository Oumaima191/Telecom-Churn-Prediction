from fastapi import FastAPI, UploadFile, File
import pandas as pd
import joblib
from fastapi.responses import JSONResponse

app = FastAPI(title="Telecom Churn Prediction API")

# Load preprocessing
imputer = joblib.load("imputer.pkl")
scaler = joblib.load("scaler.pkl")

# Load models
model_xgb = joblib.load("xgb_model.pkl")
logreg = joblib.load("logreg_model.pkl")
rf = joblib.load("rf_model.pkl")

# Preprocessing
def preprocess(data):
    data_imputed = imputer.transform(data)
    return pd.DataFrame(data_imputed, columns=data.columns)

# Prediction function
def prediction(data, model_name="xgb"):
    data_preprocessed = preprocess(data)

    if model_name == "xgb":
        preds = model_xgb.predict(data_preprocessed)
    elif model_name == "rf":
        preds = rf.predict(data_preprocessed)
    elif model_name == "logreg":
        data_scaled = scaler.transform(data_preprocessed)
        preds = logreg.predict(data_scaled)
    else:
        raise ValueError("Unknown model name. Choose: xgb, logreg, rf")
    
    return preds.tolist()

# API endpoint
@app.post("/prediction")
async def predict_api(file: UploadFile = File(...), model: str = "xgb"):
    df = pd.read_csv(file.file)
    predictions = prediction(df, model_name=model)
    return JSONResponse(content={"predictions": predictions})

# Optional root route
@app.get("/")
def root():
    return {"message": "Telecom Churn Prediction API is running!"}