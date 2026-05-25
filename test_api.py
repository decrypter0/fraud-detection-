"""
Comprehensive test suite for fraud detection API
Tests all endpoints and validates model performance
"""
import requests
import json
import time
from datetime import datetime

BASE_URL = "http://localhost:5000"

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'

def print_header(text):
    print(f"\n{Colors.BLUE}{'='*60}")
    print(f"{text}")
    print(f"{'='*60}{Colors.END}\n")

def print_success(text):
    print(f"{Colors.GREEN}✅ {text}{Colors.END}")

def print_error(text):
    print(f"{Colors.RED}❌ {text}{Colors.END}")

def print_info(text):
    print(f"{Colors.YELLOW}ℹ️  {text}{Colors.END}")

# Sample transaction features (30 PCA components)
LEGITIMATE_TRANSACTION = [
    0.0, -0.576574, -0.462325, -0.582771, 1.642434, -0.611480, 
    -0.991390, -0.763066, 0.320198, 0.405572, 0.251784, -0.228365, 
    0.405935, 0.009724, 0.798368, -0.137458, 0.141267, -0.206010, 
    0.502292, 0.219422, 0.215153, 0.073863, 0.005000, -0.036081, 
    0.639599, 0.265623, -0.113937, -0.030331, -0.099139, -0.233236
]

FRAUDULENT_TRANSACTION = [
    -1.358354, -1.340163, 1.773209, 0.379780, -0.503198, 1.800499, 
    0.791461, 0.247676, -1.514654, 0.207643, 0.624501, 0.066084, 
    0.228278, -0.010703, 0.277837, -0.110474, 0.066352, 0.128539, 
    -0.189114, 0.133558, -0.021053, 0.003725, 0.835836, 0.464960, 
    -0.099540, -0.123542, -0.098437, 0.056817, -0.169625, 0.056835
]

def test_health_check():
    """Test 1: Health Check Endpoint"""
    print_header("TEST 1: Health Check")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        
        data = response.json()
        assert data['status'] == 'healthy', "Status should be 'healthy'"
        assert data['model_loaded'] == True, "Model should be loaded"
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(data, indent=2)}")
        print_success("Health check passed")
        return True
    except Exception as e:
        print_error(f"Health check failed: {str(e)}")
        return False

def test_legitimate_transaction():
    """Test 2: Predict Legitimate Transaction"""
    print_header("TEST 2: Legitimate Transaction Prediction")
    try:
        payload = {"features": LEGITIMATE_TRANSACTION}
        response = requests.post(
            f"{BASE_URL}/predict",
            json=payload,
            timeout=5
        )
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        
        data = response.json()
        assert 'fraud_probability' in data, "Missing fraud_probability"
        assert 'is_fraud' in data, "Missing is_fraud"
        assert 'confidence' in data, "Missing confidence"
        
        # Legitimate transactions should have low fraud probability
        assert data['fraud_probability'] < 0.5, "Legitimate transaction should have low fraud probability"
        assert data['is_fraud'] == False, "Should be classified as legitimate"
        
        print(f"Status Code: {response.status_code}")
        print(f"Fraud Probability: {data['fraud_probability']:.4f}")
        print(f"Is Fraud: {data['is_fraud']}")
        print(f"Confidence: {data['confidence']}")
        print_success("Legitimate transaction correctly classified")
        return True
    except Exception as e:
        print_error(f"Legitimate transaction test failed: {str(e)}")
        return False

def test_fraudulent_transaction():
    """Test 3: Predict Fraudulent Transaction"""
    print_header("TEST 3: Fraudulent Transaction Prediction")
    try:
        payload = {"features": FRAUDULENT_TRANSACTION}
        response = requests.post(
            f"{BASE_URL}/predict",
            json=payload,
            timeout=5
        )
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        
        data = response.json()
        assert 'fraud_probability' in data, "Missing fraud_probability"
        assert 'is_fraud' in data, "Missing is_fraud"
        assert 'confidence' in data, "Missing confidence"
        
        # Fraudulent transactions should have high fraud probability
        assert data['fraud_probability'] > 0.5, "Fraudulent transaction should have high fraud probability"
        assert data['is_fraud'] == True, "Should be classified as fraudulent"
        
        print(f"Status Code: {response.status_code}")
        print(f"Fraud Probability: {data['fraud_probability']:.4f}")
        print(f"Is Fraud: {data['is_fraud']}")
        print(f"Confidence: {data['confidence']}")
        print_success("Fraudulent transaction correctly classified")
        return True
    except Exception as e:
        print_error(f"Fraudulent transaction test failed: {str(e)}")
        return False

def test_batch_prediction():
    """Test 4: Batch Prediction"""
    print_header("TEST 4: Batch Prediction")
    try:
        payload = {
            "transactions": [
                {"features": LEGITIMATE_TRANSACTION},
                {"features": FRAUDULENT_TRANSACTION},
                {"features": LEGITIMATE_TRANSACTION}
            ]
        }
        response = requests.post(
            f"{BASE_URL}/batch-predict",
            json=payload,
            timeout=5
        )
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        
        data = response.json()
        assert 'total' in data, "Missing total"
        assert 'fraudulent' in data, "Missing fraudulent count"
        assert 'results' in data, "Missing results"
        assert len(data['results']) == 3, "Should have 3 results"
        assert data['fraudulent'] >= 1, "Should detect at least 1 fraudulent transaction"
        
        print(f"Status Code: {response.status_code}")
        print(f"Total Transactions: {data['total']}")
        print(f"Fraudulent Detected: {data['fraudulent']}")
        print(f"Legitimate Detected: {data['total'] - data['fraudulent']}")
        print("\nDetailed Results:")
        for i, result in enumerate(data['results'], 1):
            print(f"  Transaction {i}: Fraud={result['is_fraud']}, Prob={result['fraud_probability']:.4f}")
        print_success("Batch prediction processed successfully")
        return True
    except Exception as e:
        print_error(f"Batch prediction test failed: {str(e)}")
        return False

def test_invalid_feature_count():
    """Test 5: Invalid Feature Count Handling"""
    print_header("TEST 5: Invalid Feature Count Handling")
    try:
        payload = {"features": [0.1, 0.2, 0.3]}  # Only 3 features instead of 30
        response = requests.post(
            f"{BASE_URL}/predict",
            json=payload,
            timeout=5
        )
        assert response.status_code == 400, f"Expected 400, got {response.status_code}"
        
        data = response.json()
        assert 'error' in data, "Should return error message"
        
        print(f"Status Code: {response.status_code}")
        print(f"Error Message: {data['error']}")
        print_success("Invalid feature count correctly rejected")
        return True
    except Exception as e:
        print_error(f"Invalid feature count test failed: {str(e)}")
        return False

def test_missing_features():
    """Test 6: Missing Features Key"""
    print_header("TEST 6: Missing Features Key")
    try:
        payload = {"invalid_key": [0.1, 0.2, 0.3]}
        response = requests.post(
            f"{BASE_URL}/predict",
            json=payload,
            timeout=5
        )
        assert response.status_code == 400, f"Expected 400, got {response.status_code}"
        
        data = response.json()
        assert 'error' in data, "Should return error message"
        
        print(f"Status Code: {response.status_code}")
        print(f"Error Message: {data['error']}")
        print_success("Missing features key correctly rejected")
        return True
    except Exception as e:
        print_error(f"Missing features test failed: {str(e)}")
        return False

def test_response_time():
    """Test 7: Response Time Performance"""
    print_header("TEST 7: Response Time Performance")
    try:
        payload = {"features": LEGITIMATE_TRANSACTION}
        
        start_time = time.time()
        response = requests.post(
            f"{BASE_URL}/predict",
            json=payload,
            timeout=5
        )
        end_time = time.time()
        
        response_time = (end_time - start_time) * 1000  # Convert to milliseconds
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        
        print(f"Response Time: {response_time:.2f}ms")
        if response_time < 100:
            print_success(f"Excellent response time (< 100ms)")
        elif response_time < 500:
            print_success(f"Good response time (< 500ms)")
        else:
            print_info(f"Response time is {response_time:.2f}ms")
        
        return True
    except Exception as e:
        print_error(f"Response time test failed: {str(e)}")
        return False

def run_all_tests():
    """Run all tests"""
    print_header("FRAUD DETECTION API - COMPREHENSIVE TEST SUITE")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Base URL: {BASE_URL}")
    
    # Check if server is running
    try:
        requests.get(f"{BASE_URL}/health", timeout=2)
    except requests.exceptions.ConnectionError:
        print_error(f"Cannot connect to API server at {BASE_URL}")
        print_info("Make sure to run 'python app.py' first")
        return False
    
    tests = [
        test_health_check,
        test_legitimate_transaction,
        test_fraudulent_transaction,
        test_batch_prediction,
        test_invalid_feature_count,
        test_missing_features,
        test_response_time
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    # Print summary
    print_header("TEST SUMMARY")
    passed = sum(results)
    total = len(results)
    
    print(f"Total Tests: {total}")
    print(f"Passed: {Colors.GREEN}{passed}{Colors.END}")
    print(f"Failed: {Colors.RED}{total - passed}{Colors.END}")
    print(f"Success Rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print_success("All tests passed! 🎉")
    else:
        print_error(f"{total - passed} test(s) failed")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
