import urllib.request
import urllib.parse
import json

def trigger_training():
    print("Logging in as admin to retrieve token...")
    login_data = urllib.parse.urlencode({
        "username": "admin@mindmesh.edu",
        "password": "Admin123!"
    }).encode('utf-8')
    
    req = urllib.request.Request("http://localhost:8000/auth/login", data=login_data)
    req.add_header("Content-Type", "application/x-www-form-urlencoded")
    
    try:
        response = urllib.request.urlopen(req)
        token_data = json.loads(response.read())
        token = token_data["access_token"]
        print("Token retrieved successfully! Triggering ML Pipeline...")
        
        train_req = urllib.request.Request("http://localhost:8000/models/train", method="POST")
        train_req.add_header("Content-Type", "application/json")
        train_req.add_header("Authorization", f"Bearer {token}")
        
        # Start training
        payload = json.dumps({"model_name": "all", "corpus_size": 500, "feature_size": 100}).encode("utf-8")
        train_res = urllib.request.urlopen(train_req, data=payload)
        train_output = json.loads(train_res.read())
        print("Training successful!")
        print(f"Trained {train_output['total_models']} Core Machine Learning algorithms.")
        
    except Exception as e:
        print(f"Error during training: {e}")

if __name__ == "__main__":
    trigger_training()
