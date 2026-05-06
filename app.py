from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load model & scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route("/")
def home():
    return "API is running"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        values = [
            data["Age"],
            data["DistanceFromHome"],
            data["Education"],
            data["EnvironmentSatisfaction"],
            data["JobInvolvement"],
            data["JobLevel"],
            data["JobSatisfaction"],
            data["MonthlyIncome"],
            data["NumCompaniesWorked"],
            data["OverTime"],
            data["PercentSalaryHike"],
            data["TotalWorkingYears"],
            data["TrainingTimesLastYear"],
            data["WorkLifeBalance"],
            data["YearsAtCompany"],
            data["YearsSinceLastPromotion"],
            data["YearsWithCurrManager"]
        ]

        values = np.array(values).reshape(1, -1)
        values_scaled = scaler.transform(values)

        prediction = model.predict(values_scaled)[0]

        return jsonify({
            "prediction": int(prediction)
        })

    except Exception as e:
        return jsonify({"error": str(e)})