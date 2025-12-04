#%%
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
# %%
data = pd.read_csv('telecom_raw.csv')
# %%
print(data.head())
# %%
print(data.isnull().sum())
# %%
data.info()
#%%
X = data.drop('Churn', axis=1)
y = data['Churn']
# %%
num_cols = X.select_dtypes(include=['float', 'int']).columns

# Numeric columns → median
imputer_num = SimpleImputer(strategy='median')
X[num_cols] = imputer_num.fit_transform(X[num_cols])
# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
# %%
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train[num_cols])
X_test_scaled  = scaler.transform(X_test[num_cols])
# %%
model_xgb = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.9,
    colsample_bytree=0.8,
    eval_metric='logloss',
    random_state=42
)
# %%
logreg = LogisticRegression(class_weight='balanced', max_iter=1000)
# %%
rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight='balanced'
)
# %%
# Entraînement
model_xgb.fit(X_train, y_train)
# %%
logreg.fit(X_train_scaled, y_train)
# %%
rf.fit(X_train, y_train)
# %%
def evaluate(model, X, y_true, scaled=False):
    if scaled:
        X = scaler.transform(X)
        
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:,1]

    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred))
    print("F1 Score:", f1_score(y_true, y_pred))
    print("ROC-AUC:", roc_auc_score(y_true, y_prob))

# %%
print("----- XGBClassifier -----")
evaluate(model_xgb, X_test, y_test, scaled=True)

print("----- Logistic Regression -----")
evaluate(logreg, X_test, y_test, scaled=True)

print("----- Random Forest -----")
evaluate(rf, X_test, y_test, scaled=False)
# %%
import joblib

# Save preprocessing
joblib.dump(imputer_num, "imputer.pkl")
joblib.dump(scaler, "scaler.pkl")

# Save models
joblib.dump(model_xgb, "xgb_model.pkl")
joblib.dump(logreg, "logreg_model.pkl")
joblib.dump(rf, "rf_model.pkl")

print("All models saved successfully.")
# %%
xgb_loaded = joblib.load("xgb_model.pkl")
logreg_loaded = joblib.load("logreg_model.pkl")
rf_loaded = joblib.load("rf_model.pkl")
print("All models loaded successfully.")