import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

preprocessed_dir = 'preprocessed'
models_dir = 'models'
os.makedirs(models_dir, exist_ok=True)

# Load data
X_train = np.load(f'{preprocessed_dir}/X_train.npy')
y_train = np.load(f'{preprocessed_dir}/y_train.npy')
X_test = np.load(f'{preprocessed_dir}/X_test.npy')
y_test = np.load(f'{preprocessed_dir}/y_test.npy')

# Train Random Forest
clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)

# Evaluate on test set
y_pred = clf.predict(X_test)
print("Test accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save the trained model
joblib.dump(clf, f'{models_dir}/rf_unsw_nb15.joblib')
print(f"Random Forest model saved to {models_dir}/rf_unsw_nb15.joblib")
