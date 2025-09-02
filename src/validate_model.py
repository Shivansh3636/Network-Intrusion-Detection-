import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def load_test_data():
    """Load and preprocess test data"""
    print("Loading test data...")
    
    # Load the preprocessed test data if available
    try:
        X_test = np.load('preprocessed/X_test.npy')
        y_test = np.load('preprocessed/y_test.npy')
        print(f"Loaded preprocessed test data: {X_test.shape}")
        return X_test, y_test
    except FileNotFoundError:
        print("Preprocessed data not found. Loading raw test data...")
        # Load raw test data
        df = pd.read_csv('data/UNSW_NB15_testing-set.csv')
        return preprocess_data(df)

def preprocess_data(df):
    """Preprocess raw data for model validation"""
    print("Preprocessing data...")
    
    # Load feature names
    feature_cols = np.load('models/feature_names.npy', allow_pickle=True).tolist()
    
    # Separate features and target
    X = df[feature_cols[:-1]]  # All except 'attack_cat'
    y = df['attack_cat'] if 'attack_cat' in df.columns else df['label']
    
    # Handle categorical columns
    categorical_cols = X.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        X[col] = X[col].astype(str)
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
    
    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    
    # Scale features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    return X_scaled, y

def validate_model():
    """Comprehensive model validation"""
    print("="*50)
    print("NETWORK INTRUSION DETECTION MODEL VALIDATION")
    print("="*50)
    
    # Load model
    print("\n1. Loading trained model...")
    model = joblib.load('models/rf_unsw_nb15.joblib')
    print(f"✓ Model loaded: {type(model).__name__}")
    
    # Load test data
    print("\n2. Loading test data...")
    X_test, y_test = load_test_data()
    print(f"✓ Test data shape: {X_test.shape}")
    print(f"✓ Test labels shape: {y_test.shape}")
    
    # Make predictions
    print("\n3. Making predictions...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
    
    # Calculate metrics
    print("\n4. Calculating performance metrics...")
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print(f"\n📊 PERFORMANCE METRICS:")
    print(f"   Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"   Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"   F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
    
    # Detailed classification report
    print(f"\n📋 DETAILED CLASSIFICATION REPORT:")
    print(classification_report(y_test, y_pred))
    
    # Attack type distribution
    unique_attacks = np.unique(y_test)
    print(f"\n🎯 ATTACK TYPES DETECTED:")
    for attack in unique_attacks:
        count = np.sum(y_test == attack)
        percentage = (count / len(y_test)) * 100
        print(f"   {attack}: {count} samples ({percentage:.2f}%)")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n🔍 CONFUSION MATRIX:")
    print(cm)
    
    # Save validation results
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'predictions_sample': y_pred[:10].tolist(),
        'attack_types': unique_attacks.tolist()
    }
    
    import json
    with open('validation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Validation complete! Results saved to 'validation_results.json'")
    return results

if __name__ == "__main__":
    validate_model()
