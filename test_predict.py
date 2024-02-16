import requests

endpoint = 'http://localhost:8080/predict'

input_data= {
    'user_id': '100',
    'n_visits':10,


}

response = requests.post(url=endpoint, json= input_data)
print(response)
print(response.json())
