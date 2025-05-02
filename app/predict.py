import requests

def test_prediction():
    url = "http://localhost:8000/predict"
    
    # Create a sample input with 754 features (fill with 0.1 as placeholder)
    sample_input = [0.1] * 754  # Replace with real features if available
    
    try:
        response = requests.post(
            url,
            json={"features": sample_input},
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        print(f"Prediction successful: {response.json()}")
    except requests.exceptions.RequestException as e:
        print(f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    test_prediction()