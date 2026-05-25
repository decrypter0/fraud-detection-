# Credit Card Fraud Detection System

A minimal, production-ready fraud detection system using Logistic Regression and Flask API.

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model
```bash
python train.py
```
This will:
- Download the Kaggle Credit Card Fraud dataset (~150K transactions)
- Train a Logistic Regression model
- Save `model.pkl` and `scaler.pkl`
- Display performance metrics

**Expected Results:**
- Precision: ~0.88
- Recall: ~0.63
- ROC-AUC: ~0.98

### 3. Start the API Server
```bash
python app.py
```
Server runs at `http://localhost:5000`

## API Endpoints

### Health Check
```bash
curl http://localhost:5000/health
```

### Single Prediction
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": [0.1, -0.5, 1.2, 0.0, ..., 0.3]  # 30 numeric values
  }'
```

**Response:**
```json
{
  "fraud_probability": 0.95,
  "is_fraud": true,
  "confidence": "high",
  "threshold": 0.5
}
```

### Batch Prediction
```bash
curl -X POST http://localhost:5000/batch-predict \
  -H "Content-Type: application/json" \
  -d '{
    "transactions": [
      {"features": [0.1, -0.5, ..., 0.3]},
      {"features": [0.2, 0.1, ..., 0.5]}
    ]
  }'
```

**Response:**
```json
{
  "total": 2,
  "fraudulent": 1,
  "results": [
    {"fraud_probability": 0.95, "is_fraud": true},
    {"fraud_probability": 0.12, "is_fraud": false}
  ]
}
```

## Model Details

- **Algorithm**: Logistic Regression
- **Features**: 30 PCA components (from original transaction data)
- **Training Data**: 227,846 transactions (0.17% fraudulent)
- **Threshold**: 0.5 (adjustable)
- **Performance**: ROC-AUC ~0.98

## Dataset

Uses the [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud):
- Transactions from September 2013
- European cardholders
- 30 anonymized features (PCA-transformed)
- Highly imbalanced (99.83% legitimate)

## Project Structure

```
fraud-detection/
├── train.py          # Training script
├── app.py            # Flask API
├── requirements.txt  # Dependencies
├── model.pkl         # Trained model (generated)
├── scaler.pkl        # Feature scaler (generated)
└── README.md         # This file
```

## Usage Example (Python)

```python
import requests
import json

# Send prediction request
response = requests.post(
    'http://localhost:5000/predict',
    json={'features': [0.1, -0.5, 1.2, 0.0, ...]}  # 30 values
)

result = response.json()
print(f"Fraud Probability: {result['fraud_probability']}")
print(f"Is Fraudulent: {result['is_fraud']}")
```

## Optimization Tips

1. **Adjust threshold** in `app.py` based on business needs:
   - Higher threshold → fewer false positives, more fraud slips through
   - Lower threshold → more false positives, better fraud catch

2. **Add rate limiting** for production

3. **Add logging** for fraud alerts

4. **Retrain monthly** with new transaction data

## Files Generated After Training

- `model.pkl` - Trained Logistic Regression model
- `scaler.pkl` - StandardScaler for feature normalization

## Next Steps

- Add database integration (PostgreSQL, MongoDB)
- Add real-time monitoring dashboard
- Implement model versioning
- Add A/B testing for threshold optimization
- Deploy to Docker/Kubernetes
