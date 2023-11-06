import pickle
from flask import Flask, request, jsonify

model_file = 'random_forest_model.pkl'
# Load the model and vectorizer
with open(model_file, 'rb') as file:
    loaded_model_data = pickle.load(file)

# Retrieve the loaded model and vectorizer
loaded_rf_model = loaded_model_data['model']
loaded_vectorizer = loaded_model_data['vectorizer']

app = Flask('heart_disease')

@app.route('/app', methods=['POST'])
def predict():
    if request.method == 'POST':
        data = request.get_json()

        def predict_heart_disease(sample_patient_data, loaded_rf_model, loaded_vectorizer):
            # Transform the dictionary into a feature vector using the loaded DictVectorizer
            sample_patient_vector = loaded_vectorizer.transform([sample_patient_data])

            # Make predictions using the loaded Random Forest model
            predictions = loaded_rf_model.predict(sample_patient_vector)

            y_pred = loaded_rf_model.predict_proba(sample_patient_vector)[0, 1]
            heart_disease = y_pred >= 0.5

            result = {
                'Heart disease Probability': y_pred,
                'Has_heart_disease': bool(heart_disease)
            }

            return jsonify(result)

        # Call the prediction function and return the result
        return predict_heart_disease(data, loaded_rf_model, loaded_vectorizer)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)
