from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('risk_model.pkl')

@app.route('/')
def home():
    return "🏥 Health Risk AI is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    try:
        hr = float(data['heart_rate'])
        spo2 = float(data['spo2'])
        temp = float(data['temperature'])

        input_data = np.array([[hr, spo2, temp]])
        prediction = model.predict(input_data)[0]

        result = "RISK DETECTED" if prediction == 1 else "No Risk"
        return jsonify({"input": data, "prediction": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
