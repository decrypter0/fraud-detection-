"""
Flask API for real-time fraud detection
Endpoints:
  POST /predict - Predict fraud probability for a transaction
  GET /health - Health check
"""
from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load model and scaler
try:
    model = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')
    print("✅ Model loaded successfully")
except FileNotFoundError:
    print("❌ Model files not found. Run 'python train.py' first")
    model = None
    scaler = None

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    }), 200

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict fraud probability for a transaction
    
    Expected JSON payload (30 features):
    {
        "features": [f1, f2, ..., f30]  # 30 numeric values from PCA
    }
    
    Returns:
    {
        "fraud_probability": 0.95,
        "is_fraud": true,
        "confidence": "high"
    }
    """
    if model is None or scaler is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        
        if 'features' not in data:
            return jsonify({'error': 'Missing "features" key in request'}), 400
        
        features = np.array(data['features']).reshape(1, -1)
        
        if features.shape[1] != 30:
            return jsonify({
                'error': f'Expected 30 features, got {features.shape[1]}'
            }), 400
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Predict
        fraud_prob = model.predict_proba(features_scaled)[0, 1]
        is_fraud = model.predict(features_scaled)[0] == 1
        
        # Confidence level
        confidence = 'high' if fraud_prob > 0.7 else 'medium' if fraud_prob > 0.4 else 'low'
        
        return jsonify({
            'fraud_probability': float(fraud_prob),
            'is_fraud': bool(is_fraud),
            'confidence': confidence,
            'threshold': 0.5
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/batch-predict', methods=['POST'])
def batch_predict():
    """
    Batch prediction for multiple transactions
    
    Expected JSON payload:
    {
        "transactions": [
            {"features": [f1, f2, ..., f30]},
            {"features": [f1, f2, ..., f30]},
            ...
        ]
    }
    """
    if model is None or scaler is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        transactions = data.get('transactions', [])
        
        if not transactions:
            return jsonify({'error': 'No transactions provided'}), 400
        
        results = []
        for txn in transactions:
            features = np.array(txn['features']).reshape(1, -1)
            features_scaled = scaler.transform(features)
            fraud_prob = model.predict_proba(features_scaled)[0, 1]
            is_fraud = model.predict(features_scaled)[0] == 1
            
            results.append({
                'fraud_probability': float(fraud_prob),
                'is_fraud': bool(is_fraud)
            })
        
        return jsonify({
            'total': len(results),
            'fraudulent': sum(1 for r in results if r['is_fraud']),
            'results': results
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
