import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# Define column names for UNSW-NB15 dataset
column_names = [
    'srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', 'dur', 'sbytes', 'dbytes',
    'sttl', 'dttl', 'sloss', 'dloss', 'service', 'Sload', 'Dload', 'Spkts', 'Dpkts',
    'swin', 'dwin', 'stcpb', 'dtcpb', 'smeansz', 'dmeansz', 'trans_depth', 'res_bdy_len',
    'Sjit', 'Djit', 'Stime', 'Ltime', 'Sintpkt', 'Dintpkt', 'tcprtt', 'synack', 'ackdat',
    'is_sm_ips_ports', 'ct_state_ttl', 'ct_flw_http_mthd', 'is_ftp_login', 'ct_ftp_cmd',
    'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ltm', 'ct_src_dport_ltm',
    'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'attack_cat', 'label'
]

data_dir = 'data'
preprocessed_dir = 'preprocessed'
os.makedirs(preprocessed_dir, exist_ok=True)
all_features, all_labels = [], []

for filename in os.listdir(data_dir):
    if filename.lower().endswith('.csv'):
        print(f"Processing file: {filename}")
        df = pd.read_csv(os.path.join(data_dir, filename), header=None, names=column_names, low_memory=False, encoding='latin1')
        df = df.dropna(subset=['label'])
        if df.shape[0] == 0:
            continue
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col] = df[col].astype(str)
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
        X = df.drop(columns=['label'])
        y = df['label']
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        all_features.append(X_scaled)
        all_labels.append(y.values)

X_combined = np.vstack(all_features)
y_combined = np.hstack(all_labels)

# Remove rows where label is NaN (defensive)
mask_ok = ~pd.isnull(y_combined)
X = X_combined[mask_ok]
y = y_combined[mask_ok]

# Impute missing features (mean)
imp = SimpleImputer(strategy='mean')
X = imp.fit_transform(X)

# Balance with SMOTE
smote = SMOTE(random_state=42)
X_bal, y_bal = smote.fit_resample(X, y)

# Split 80/20 stratified
X_train, X_test, y_train, y_test = train_test_split(
    X_bal, y_bal, test_size=0.2, random_state=42, stratify=y_bal
)

# Save train/test splits
np.save(f'{preprocessed_dir}/X_train.npy', X_train)
np.save(f'{preprocessed_dir}/y_train.npy', y_train)
np.save(f'{preprocessed_dir}/X_test.npy', X_test)
np.save(f'{preprocessed_dir}/y_test.npy', y_test)
print(f"Saved: {preprocessed_dir}/X_train.npy, X_test.npy, y_train.npy, y_test.npy")
