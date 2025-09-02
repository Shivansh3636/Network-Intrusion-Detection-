import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer

# Read the actual data file
print("Reading training data file...")
df = pd.read_csv('data/UNSW_NB15_training-set.csv', header=None, nrows=10)
print(f"Raw data shape: {df.shape}")

# Check the first row to see if it has headers
print("\nFirst row (potential headers):")
print(df.iloc[0].tolist())

# The first row looks like headers, so let's read it properly
print("\nReading with headers...")
df_with_headers = pd.read_csv('data/UNSW_NB15_training-set.csv', nrows=10)
print(f"Data with headers shape: {df_with_headers.shape}")
print("Actual column names:")
for i, col in enumerate(df_with_headers.columns):
    print(f"{i+1:2d}. {col}")

# Now let's simulate the preprocessing logic
print(f"\nSimulating preprocessing logic...")
print(f"Original columns: {len(df_with_headers.columns)}")

# Drop the label column if it exists
if 'label' in df_with_headers.columns:
    X = df_with_headers.drop(columns=['label'])
    print(f"After dropping 'label': {len(X.columns)} columns")
else:
    X = df_with_headers
    print("No 'label' column found")

# Check for categorical columns
categorical_cols = X.select_dtypes(include=['object']).columns
print(f"Categorical columns: {list(categorical_cols)}")

# Apply label encoding to categorical columns
for col in categorical_cols:
    X[col] = X[col].astype(str)
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

print(f"After label encoding: {X.shape[1]} features")

# Load the actual preprocessed data to compare
X_train_actual = np.load('preprocessed/X_train.npy')
print(f"Actual preprocessed data: {X_train_actual.shape[1]} features")

# The difference might be due to the imputer or other processing
# Let's check if there are any NaN values that get handled
print(f"\nNaN values in processed data: {X.isnull().sum().sum()}")

# Apply imputer
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)
print(f"After imputation: {X_imputed.shape[1]} features")

# Apply scaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_imputed)
print(f"After scaling: {X_scaled.shape[1]} features")

print(f"\nFinal result: {X_scaled.shape[1]} features")
print(f"Expected by model: 48 features")
print(f"Match: {X_scaled.shape[1] == 48}")

