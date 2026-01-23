
import json
import joblib
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

def train_model(data_path="apps/api/data/synthetic_affect_data.json", model_path="apps/api/data/affect_model.joblib"):
    print(f"Loading data from {data_path}...")
    
    with open(data_path, 'r') as f:
        data = json.load(f)
        
    # extract features and labels
    X = []
    y = []
    
    for item in data:
        features = [
            item["velocity_mean"],
            item["velocity_variance"],
            item["click_rate"],
            item["idle_time_ratio"],
            item["scroll_depth_rate"]
        ]
        X.append(features)
        y.append(item["label"])
        
    X = np.array(X)
    y = np.array(y)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training RandomForestClassifier on {len(X_train)} samples...")
    clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    clf.fit(X_train, y_train)
    
    # Evaluate
    preds = clf.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    print(f"Model Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, preds, target_names=["Flow", "Frustrated", "Bored"]))
    
    # Save
    joblib.dump(clf, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    # Ensure sklearn is installed
    # pip install scikit-learn
    train_model()
