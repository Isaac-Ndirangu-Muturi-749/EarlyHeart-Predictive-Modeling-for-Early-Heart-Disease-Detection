import pickle

output_file="random_forest_model.pkl"

# Load the saved model and DictVectorizer from the file
with open(output_file, 'rb') as file:
    loaded_model_data = pickle.load(file)
print("load model and vectorizer")
print()
# Retrieve the loaded model and vectorizer
loaded_rf_model = loaded_model_data['model']
loaded_vectorizer = loaded_model_data['vectorizer']

# Create a sample patient data dictionary
sample_patient_data = {
    'Sex': 'Male',
    'ChestPainType': 'Typical Angina',
    'RestingBP': 130,
    'Cholesterol': 210,
    'FastingBS': 0,
    'RestingECG': 'ST-T wave abnormality',
    'MaxHR': 160,
    'ExerciseAngina': 'No',
    'Oldpeak': 1.2,
    'ST_Slope': 'Upsloping'
}

print( "input:", sample_patient_data)
print()
# Transform the dictionary into a feature vector using the loaded DictVectorizer
sample_patient_vector = loaded_vectorizer.transform([sample_patient_data])

print(f"Heart disease Probability: {loaded_rf_model.predict_proba(sample_patient_vector)[0, 1]:.2f}")

print()
# Make predictions using the loaded Random Forest model
predictions = loaded_rf_model.predict(sample_patient_vector)

# Print the predicted outcome for the sample patient
if predictions[0] == 0:
    print("The patient does not have heart disease.")
else:
    print("The patient has heart disease.")
