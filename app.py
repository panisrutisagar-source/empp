from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["features"]          # list of 17 numbers
    scaled = scaler.transform([data])
    result = model.predict(scaled)[0]
    label = "WILL LEAVE" if result == 1 else "WILL STAY"
    return jsonify({"prediction": label})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)