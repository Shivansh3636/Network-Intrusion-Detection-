from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer

app = Flask(__name__)

# Load pretrained model at app startup
model = joblib.load('models/rf_unsw_nb15.joblib')

# Load feature names from saved file
try:
    feature_cols = np.load('models/feature_names.npy', allow_pickle=True).tolist()
    print(f"Loaded {len(feature_cols)} feature columns from saved file")
except FileNotFoundError:
    # Fallback to hardcoded feature columns if file doesn't exist
    feature_cols = [
        'srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', 'dur', 'sbytes', 'dbytes',
        'sttl', 'dttl', 'sloss', 'dloss', 'service', 'Sload', 'Dload', 'Spkts', 'Dpkts',
        'swin', 'dwin', 'stcpb', 'dtcpb', 'smeansz', 'dmeansz', 'trans_depth', 'res_bdy_len',
        'Sjit', 'Djit', 'Stime', 'Ltime', 'Sintpkt', 'Dintpkt', 'tcprtt', 'synack', 'ackdat',
        'is_sm_ips_ports', 'ct_state_ttl', 'ct_flw_http_mthd', 'is_ftp_login', 'ct_ftp_cmd',
        'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ltm', 'ct_src_dport_ltm',
        'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'attack_cat'
    ]
    print(f"Using fallback feature columns: {len(feature_cols)} features")

@app.route('/', methods=['GET'])
def index():
    return f'''
    <h2>Upload CSV File for Network Intrusion Detection</h2>
    <p><strong>Expected columns ({len(feature_cols)} features):</strong></p>
    <ul>
        {''.join([f'<li>{col}</li>' for col in feature_cols])}
    </ul>
    <p><strong>CSV Format Options:</strong></p>
    <ul>
        <li><strong>Option 1:</strong> CSV with header row containing the exact column names above</li>
        <li><strong>Option 2:</strong> CSV without headers - the system will automatically assign column names to the first {len(feature_cols)} columns</li>
    </ul>
    <p><strong>Note:</strong> If your CSV has more than {len(feature_cols)} columns, only the first {len(feature_cols)} will be used.</p>
    <p><strong>Important:</strong> The 'attack_cat' column is included as a feature in the model, so your CSV should include this column.</p>
    <form action="/predict" method="post" enctype="multipart/form-data">
      <input type="file" name="file" accept=".csv" required>
      <input type="submit" value="Predict">
    </form>
    <p><a href="/columns">View expected columns as JSON</a></p>
    <p><a href="/sample">Download sample CSV format</a></p>
    '''

@app.route('/columns', methods=['GET'])
def get_columns():
    return jsonify({
        "expected_columns": feature_cols,
        "total_features": len(feature_cols)
    })

@app.route('/sample', methods=['GET'])
def download_sample():
    from flask import send_file
    import io
    
    # Create a sample CSV with headers
    sample_data = pd.DataFrame(columns=feature_cols)
    sample_data.loc[0] = [0.0] * len(feature_cols)  # Add a sample row with zeros
    
    # Create CSV in memory
    output = io.StringIO()
    sample_data.to_csv(output, index=False)
    output.seek(0)
    
    return send_file(
        io.BytesIO(output.getvalue().encode('utf-8')),
        mimetype='text/csv',
        as_attachment=True,
        download_name='sample_network_data.csv'
    )

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"})

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"})

    try:
        # Try to read CSV with headers first
        try:
            df_raw = pd.read_csv(file, encoding='utf-8')
            has_headers = True
        except:
            # If that fails, try without headers and assign column names
            df_raw = pd.read_csv(file, header=None, encoding='utf-8')
            has_headers = False
            
        # If no headers, we need to assign the correct column names
        if not has_headers or len(df_raw.columns) != len(feature_cols):
            # Check if the data has the right number of columns
            if len(df_raw.columns) >= len(feature_cols):
                # Assign the feature column names to the first N columns
                df_raw.columns = feature_cols + [f'extra_{i}' for i in range(len(df_raw.columns) - len(feature_cols))]
            else:
                return jsonify({
                    "error": f"CSV file has {len(df_raw.columns)} columns but {len(feature_cols)} are required",
                    "expected_columns": feature_cols,
                    "uploaded_columns": list(df_raw.columns),
                    "total_expected": len(feature_cols),
                    "total_uploaded": len(df_raw.columns)
                })

        # Validate required feature columns are in the uploaded data
        missing_cols = [col for col in feature_cols if col not in df_raw.columns]
        if missing_cols:
            return jsonify({
                "error": f"Missing columns for prediction: {missing_cols}",
                "expected_columns": feature_cols,
                "uploaded_columns": list(df_raw.columns),
                "total_expected": len(feature_cols),
                "total_uploaded": len(df_raw.columns)
            })

        # Select only the features your model expects and enforce column order
        df = df_raw[feature_cols].copy()

        # Preprocess categorical columns: convert to string and label encode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col] = df[col].astype(str)
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])

        # Impute missing values with mean
        imputer = SimpleImputer(strategy='mean')
        X_imputed = imputer.fit_transform(df)

        # Normalize features
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X_imputed)

        # Predict
        preds = model.predict(X_scaled)

        return jsonify({"predictions": preds.tolist()})

    except Exception as e:
        return jsonify({"error": "Failed to process file", "details": str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
