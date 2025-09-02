import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer

# Load a small sample
df = pd.read_csv('data/UNSW-NB15_1.csv', nrows=5)
print(f"1. Original data shape: {df.shape}")

# Take first 48 columns
X = df.iloc[:, :48].copy()
print(f"2. After taking first 48 columns: {X.shape}")

# Handle categorical columns
print(f"3. Categorical columns: {X.select_dtypes(include=['object']).columns.tolist()}")
categorical_cols = X.select_dtypes(include=['object']).columns
for col in categorical_cols:
    X[col] = X[col].astype(str)
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

print(f"4. After encoding categoricals: {X.shape}")

# Convert to numeric
X_numeric = X.apply(pd.to_numeric, errors='coerce')
print(f"5. After numeric conversion: {X_numeric.shape}")
print(f"   NaN columns: {X_numeric.isnull().all().sum()}")

# Drop columns that are all NaN
X_clean = X_numeric.dropna(axis=1, how='all')
print(f"6. After dropping all-NaN columns: {X_clean.shape}")

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X_clean)
print(f"7. After imputation: {X_imputed.shape}")

# Scale features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_imputed)
print(f"8. Final scaled shape: {X_scaled.shape}")

# Show which column was dropped
if X_clean.shape[1] < 48:
    dropped_cols = set(range(48)) - set(range(X_clean.shape[1]))
    print(f"Dropped column indices: {dropped_cols}")
    print(f"Original column {list(dropped_cols)[0]} content: {X.iloc[:, list(dropped_cols)[0]].tolist()}")
