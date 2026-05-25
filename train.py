"""
Train the fraud detection model
Downloads Kaggle Credit Card Fraud dataset, trains Logistic Regression model
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    precision_score, recall_score, f1_score
)
import joblib
import requests
import os
from io import StringIO

def download_dataset():
    """Download Kaggle credit card fraud dataset"""
    url = "https://raw.githubusercontent.com/datasets/creditcard/master/data.csv"
    
    print("Downloading dataset...")
    response = requests.get(url)
    df = pd.read_csv(StringIO(response.text))
    
    print(f"Dataset shape: {df.shape}")
    print(f"Fraud cases: {df['Class'].sum()} ({df['Class'].mean()*100:.2f}%)")
    
    return df

def train_model(df):
    """Train Logistic Regression model"""
    
    # Separate features and target
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Split data (80-20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Logistic Regression
    print("\nTraining Logistic Regression model...")
    model = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    print("\n" + "="*50)
    print("MODEL PERFORMANCE")
    print("="*50)
    
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud']))
    
    # Save model and scaler
    joblib.dump(model, 'model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    print("\n✅ Model saved as model.pkl")
    print("✅ Scaler saved as scaler.pkl")
    
    return model, scaler

if __name__ == "__main__":
    df = download_dataset()
    model, scaler = train_model(df)
