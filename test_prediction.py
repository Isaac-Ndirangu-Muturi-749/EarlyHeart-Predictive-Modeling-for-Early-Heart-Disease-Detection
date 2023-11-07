import requests
import json

# Define the sample patient data
sample_patient_data = {
    "Sex": "Male",
    "ChestPainType": "Typical Angina",
    "RestingBP": 130,
    "Cholesterol": 210,
    "FastingBS": 0,
    "RestingECG": "ST-T wave abnormality",
    "MaxHR": 160,
    "ExerciseAngina": "No",
    "Oldpeak": 1.2,
    "ST_Slope": "Upsloping"
}

# URL of your local API endpoint
url = "http://heart-serving-env.eba-pahrexjp.eu-west-3.elasticbeanstalk.com/predict"


# Send a POST request to the API with the sample patient data
response = requests.post(url, json=sample_patient_data).json()

# Print the response from the API
print("API Response:")
print(json.dumps(response, indent=2))
print()

# Check the predicted outcome for the sample patient
if response.get('Has_heart_disease', False):
    print("Test Result: The patient is predicted to have heart disease.")
else:
    print("Test Result: The patient is predicted to not have heart disease.")
