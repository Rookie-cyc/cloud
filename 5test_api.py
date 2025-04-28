import requests
import json

url = "http://127.0.0.1:5003/predict"
print(f"Sending request to: {url}")  

data = {
    "features": [63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]
  }
  

response = requests.post(url, headers={"Content-Type": "application/json"}, data=json.dumps(data))

print("Status Code:", response.status_code)
print("Response:", response.json())
