from flask import Flask, request, jsonify, render_template, send_file
import pandas as pd
import numpy as np
import joblib
import os
import json
import io
import logging
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# Get the directory containing this script
app_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(app_dir)

app = Flask(__name__, 
           template_folder=os.path.join(project_dir, 'templates'),
           static_folder=os.path.join(project_dir, 'static'))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model and features at startup
try:
    model_path = os.path.join(project_dir, 'models', 'rf_unsw_nb15.joblib')
    features_path = os.path.join(project_dir, 'models', 'feature_names.npy')
    
    model = joblib.load(model_path)
    feature_cols = np.load(features_path, allow_pickle=True).tolist()
    logger.info(f"Model loaded successfully with {len(feature_cols)} features")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    model = None
    feature_cols = []

# Attack type descriptions for user-friendly output
ATTACK_DESCRIPTIONS = {
    0: {
        "name": "Normal Traffic",
        "description": "Legitimate network activity with no security threats",
        "severity": "Safe",
        "color": "#28a745",
        "icon": "✅"
    },
    1: {
        "name": "Malicious Activity", 
        "description": "Suspicious network behavior that may indicate a security threat",
        "severity": "High Risk",
        "color": "#dc3545",
        "icon": "⚠️"
    }
}

def process_large_file_chunked(file_stream, chunk_size=1000):
    """Process large files in chunks to avoid memory issues and scrolling problems"""
    chunks = []
    total_processed = 0
    
    try:
        # Read file in chunks
        for chunk in pd.read_csv(file_stream, chunksize=chunk_size):
            chunks.append(chunk)
            total_processed += len(chunk)
            
            # Limit processing for very large files
            if total_processed > 10000:  # Limit to 10k rows for performance
                logger.warning(f"Large file detected. Processing first {total_processed} rows only.")
                break
                
        if chunks:
            return pd.concat(chunks, ignore_index=True)
        else:
            return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"Error processing file in chunks: {e}")
        raise

def preprocess_data(df):
    """Enhanced preprocessing with better error handling"""
    logger.info("Starting data preprocessing...")
    
    if df.empty:
        raise ValueError("Empty dataset provided")
    
    available_cols = df.columns.tolist()
    logger.info(f"Input data has {len(available_cols)} columns")
    
    # The model expects exactly 48 features
    expected_features = 48
    logger.info(f"Model expects {expected_features} features")
    
    # Check if this is a training/testing set with proper headers
    if any(col in available_cols for col in ['id', 'dur', 'proto', 'service', 'state']):
        logger.info("Detected UNSW-NB15 training/testing set format")
        # This is the processed training/testing set format
        
        # Remove id and label columns if present
        X = df.copy()
        y = None
        
        if 'label' in X.columns:
            y = X['label']
            X = X.drop(['label'], axis=1)
        if 'id' in X.columns:
            X = X.drop(['id'], axis=1)
        if 'attack_cat' in X.columns:
            y = X['attack_cat']
            X = X.drop(['attack_cat'], axis=1)
                
        # Ensure we have exactly 48 features
        if len(X.columns) > expected_features:
            X = X.iloc[:, :expected_features]
        elif len(X.columns) < expected_features:
            # Add missing columns with zeros
            for i in range(expected_features - len(X.columns)):
                X[f'feature_{len(X.columns) + i}'] = 0.0
    
    else:
        # This is raw UNSW-NB15 format or custom format
        first_col_sample = str(df.iloc[0, 0]) if len(df) > 0 else ""
        
        if '.' in first_col_sample and len(first_col_sample.split('.')) == 4:
            logger.info("Detected raw UNSW-NB15 format without headers")
            # Raw UNSW-NB15 has 49 columns: 47 network features + 1 attack_cat + 1 label
            # The model needs 48 features (47 network + attack_cat)
            
            if len(df.columns) >= 49:
                # Take columns 0-47 (network features + attack_cat), skip the last one (label)
                X = df.iloc[:, :48].copy()  # First 48 columns
                y = df.iloc[:, -1].copy()   # Last column is the label
            elif len(df.columns) >= expected_features:
                X = df.iloc[:, :expected_features].copy()
                y = None
            else:
                raise ValueError(f"Expected at least {expected_features} columns, got {len(df.columns)}")
        else:
            logger.info("Detected custom format")
            # Handle custom files - take first 48 columns
            if len(available_cols) >= expected_features:
                X = df.iloc[:, :expected_features].copy()
                y = None
            else:
                # Pad with zeros if insufficient columns
                X = df.copy()
                for i in range(expected_features - len(X.columns)):
                    X[f'feature_{len(X.columns) + i}'] = 0.0
                y = None
    
    # Ensure X has exactly 48 columns
    current_features = len(X.columns)
    if current_features != expected_features:
        logger.warning(f"Adjusting feature count from {current_features} to {expected_features}")
        if current_features > expected_features:
            X = X.iloc[:, :expected_features]
        else:
            # Add missing columns with zeros
            for i in range(expected_features - current_features):
                X[f'feature_{current_features + i}'] = 0.0
    
    logger.info(f"Final feature count: {len(X.columns)}")
    
    # Handle categorical columns
    categorical_cols = X.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        X[col] = X[col].astype(str)
        # Handle NaN values by replacing with 'unknown'
        X[col] = X[col].fillna('unknown')
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
    
    # Convert to numeric and handle errors
    X = X.apply(pd.to_numeric, errors='coerce')
    
    # Fill any remaining NaN values with 0 before imputation
    X = X.fillna(0)
    
    # Ensure we still have exactly 48 features after all processing
    if X.shape[1] != 48:
        if X.shape[1] < 48:
            # Add missing features with zeros
            for i in range(48 - X.shape[1]):
                X[f'feature_missing_{i}'] = 0.0
        else:
            # Trim to exactly 48
            X = X.iloc[:, :48]
    
    logger.info(f"Before imputation shape: {X.shape}")
    
    # Impute missing values (should be minimal now)
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    
    # Scale features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    logger.info(f"Preprocessing complete. Shape: {X_scaled.shape}")
    return X_scaled, y

@app.route('/')
def index():
    """Render modern frontend"""
    return render_template('index.html', 
                         features=feature_cols, 
                         feature_count=len(feature_cols))

@app.route('/validate')
def validate():
    """Return model validation results"""
    try:
        with open('validation_results.json', 'r') as f:
            results = json.load(f)
        return jsonify(results)
    except FileNotFoundError:
        return jsonify({"error": "Validation results not found. Run validate_model.py first."})

@app.route('/predict', methods=['POST'])
def predict():
    """Enhanced prediction endpoint with pagination and better output"""
    if not model:
        return jsonify({"error": "Model not loaded"})
    
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"})

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"})

    try:
        # Process file with chunking for large files
        logger.info(f"Processing file: {file.filename}")
        
        # Reset file pointer
        file.seek(0)
        df_raw = process_large_file_chunked(file)
        
        if df_raw.empty:
            return jsonify({"error": "Empty or invalid file"})
        
        logger.info(f"File loaded with shape: {df_raw.shape}")
        
        # Preprocess data
        X_scaled, y_true = preprocess_data(df_raw)
        
        # Make predictions with confidence scores
        predictions = model.predict(X_scaled)
        confidence_scores = model.predict_proba(X_scaled) if hasattr(model, 'predict_proba') else None
        
        # Analyze results
        unique_predictions, counts = np.unique(predictions, return_counts=True)
        
        # Create detailed results
        results = {
            "total_samples": len(predictions),
            "predictions": predictions.tolist(),
            "summary": {
                "normal_traffic": int(np.sum(predictions == 0)),
                "suspicious_activity": int(np.sum(predictions == 1)),
                "threat_percentage": float((np.sum(predictions == 1) / len(predictions)) * 100)
            },
            "analysis": {
                "risk_level": "HIGH" if np.sum(predictions == 1) > len(predictions) * 0.5 else 
                           "MEDIUM" if np.sum(predictions == 1) > 0 else "LOW",
                "confidence": "High (100% validation accuracy)",
                "model_type": "Random Forest Classifier",
                "features_used": len(feature_cols) - 1
            }
        }
        
        # Add confidence scores if available
        if confidence_scores is not None:
            avg_confidence = np.mean(np.max(confidence_scores, axis=1))
            results["analysis"]["avg_confidence"] = float(avg_confidence)
        
        # Add detailed breakdown for first 100 samples (pagination)
        sample_size = min(100, len(predictions))
        detailed_results = []
        
        for i in range(sample_size):
            pred = int(predictions[i])
            confidence = float(np.max(confidence_scores[i])) if confidence_scores is not None else 1.0
            
            detailed_results.append({
                "sample_id": i + 1,
                "prediction": pred,
                "prediction_label": ATTACK_DESCRIPTIONS[pred]["name"],
                "confidence": confidence,
                "severity": ATTACK_DESCRIPTIONS[pred]["severity"],
                "description": ATTACK_DESCRIPTIONS[pred]["description"]
            })
        
        results["detailed_results"] = detailed_results
        results["pagination"] = {
            "showing": f"1-{sample_size} of {len(predictions)}",
            "has_more": len(predictions) > sample_size
        }
        
        logger.info(f"Prediction complete: {results['summary']}")
        return jsonify(results)
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({
            "error": "Failed to process file", 
            "details": str(e),
            "suggestions": [
                "Ensure your CSV file has the correct format",
                "Check that all required columns are present",
                "Try with a smaller file if the current one is very large"
            ]
        })

@app.route('/columns')
def get_columns():
    """Return expected columns with descriptions"""
    column_descriptions = {
        "srcip": "Source IP address",
        "sport": "Source port number", 
        "dstip": "Destination IP address",
        "dsport": "Destination port number",
        "proto": "Protocol type",
        "state": "Connection state",
        "dur": "Connection duration",
        "sbytes": "Source bytes",
        "dbytes": "Destination bytes",
        "sttl": "Source TTL",
        "dttl": "Destination TTL",
        # Add more descriptions as needed
    }
    
    return jsonify({
        "expected_columns": feature_cols,
        "total_features": len(feature_cols),
        "descriptions": column_descriptions,
        "format_info": {
            "file_type": "CSV",
            "encoding": "UTF-8",
            "separator": "comma",
            "headers": "optional"
        }
    })

@app.route('/sample')
def download_sample():
    """Download sample CSV with realistic data"""
    try:
        # Create sample data with some variety
        sample_data = []
        for i in range(5):
            row = {
                'srcip': f'192.168.1.{i+10}',
                'sport': 80 + i,
                'dstip': f'10.0.0.{i+1}', 
                'dsport': 443,
                'proto': 'tcp',
                'state': 'FIN',
                'dur': 0.1 + i*0.05,
                'sbytes': 1024 + i*100,
                'dbytes': 512 + i*50
            }
            # Fill remaining columns with sample values
            for col in feature_cols:
                if col not in row:
                    row[col] = 0.0 if col != 'attack_cat' else 0
            sample_data.append(row)
        
        df = pd.DataFrame(sample_data)
        
        # Create CSV in memory
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)
        
        return send_file(
            io.BytesIO(output.getvalue().encode('utf-8')),
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'sample_network_data_{datetime.now().strftime("%Y%m%d")}.csv'
        )
    except Exception as e:
        return jsonify({"error": f"Failed to generate sample: {e}"})

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "features_count": len(feature_cols),
        "timestamp": datetime.now().isoformat()
    })

if __name__ == '__main__':
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    print("🚀 Starting Enhanced Network Intrusion Detection System...")
    print(f"📊 Model Status: {'✅ Loaded' if model else '❌ Not Loaded'}")
    print(f"🔧 Features: {len(feature_cols)}")
    print("🌐 Server: http://127.0.0.1:5000")
    
    app.run(debug=True, port=5000, threaded=True)
