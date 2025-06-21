import requests
import json
from time import sleep
url = "http://210.28.135.197:8080/test_plan"
request_data = {
    "startCity": "苏州",
    "destinationCity": "南京",
    "peopleCount": 2,
    "daysCount": 2,
    "additionalRequirements": "想吃烤鸭"
}
response = requests.post(url, json=request_data)
response_data = response.json()
print(response_data)

input("Press Enter to continue...")

get_url = f"http://210.28.135.197:8080/get_plan"
params = {"task_id": response_data["task_id"]}
response = requests.get(get_url, params=params)
response_data = response.json()
print(response_data)
