import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
df = pd.read_csv("D:\Machine Failure Prediction\predictive_maintenance_balanced.csv")
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

print("Columns in dataset:", df.columns.tolist())
if "target" in df.columns:
    failure_status_col = "target"
elif "failure" in df.columns:
    failure_status_col = "failure"
else:
    raise ValueError("Failure status column not found")

if "failure_type" in df.columns:
    failure_type_col = "failure_type"
else:
    raise ValueError("Failure type column not found")

X = df.drop([failure_status_col, failure_type_col], axis=1)
y_failure_status = df[failure_status_col]
y_failure_type = df[failure_type_col]
label_encoders = {}
categorical_cols = X.select_dtypes(include=["object"]).columns

for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le
failure_status_encoder = LabelEncoder()
y_failure_status_encoded = failure_status_encoder.fit_transform(y_failure_status)

target_encoder = LabelEncoder()
y_failure_type_encoded = target_encoder.fit_transform(y_failure_type)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, "../model/scaler.pkl")
joblib.dump(label_encoders, "../model/label_encoders.pkl")
joblib.dump(target_encoder, "../model/target_encoder.pkl")
joblib.dump(failure_status_encoder, "../model/failure_status_encoder.pkl")
joblib.dump(X.columns.tolist(), "../model/feature_columns.pkl")

print("Feature shape:", X_scaled.shape)