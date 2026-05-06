from flask import Flask, request, jsonify, render_template_string
import pickle
import numpy as np
import os

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Employee Attrition Predictor</title>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #0f0e17;
    --card: #1a1826;
    --border: #2e2b3e;
    --accent: #ff6b6b;
    --accent2: #ffd93d;
    --green: #6bcb77;
    --text: #fffffe;
    --muted: #a7a9be;
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: 'DM Sans', sans-serif;
    min-height: 100vh;
    padding: 0 0 60px;
  }

  header {
    background: linear-gradient(135deg, #1a1826 0%, #0f0e17 100%);
    border-bottom: 1px solid var(--border);
    padding: 28px 24px 20px;
    text-align: center;
    position: relative;
    overflow: hidden;
  }

  header::before {
    content: '';
    position: absolute;
    top: -40px; left: 50%;
    transform: translateX(-50%);
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(255,107,107,0.15) 0%, transparent 70%);
    pointer-events: none;
  }

  header h1 {
    font-family: 'Syne', sans-serif;
    font-size: 1.7rem;
    font-weight: 800;
    letter-spacing: -0.5px;
    position: relative;
  }

  header h1 span { color: var(--accent); }

  header p {
    color: var(--muted);
    font-size: 0.85rem;
    margin-top: 6px;
    position: relative;
  }

  .container { padding: 24px 16px; max-width: 500px; margin: 0 auto; }

  .section-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--accent);
    margin: 28px 0 14px;
    padding-left: 2px;
  }

  .field {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 14px 16px;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    transition: border-color 0.2s;
  }

  .field:focus-within {
    border-color: var(--accent);
  }

  .field-info { flex: 1; }

  .field-name {
    font-size: 0.9rem;
    font-weight: 500;
    color: var(--text);
  }

  .field-range {
    font-size: 0.72rem;
    color: var(--muted);
    margin-top: 2px;
  }

  .field input {
    width: 90px;
    background: rgba(255,255,255,0.05);
    border: 1px solid var(--border);
    border-radius: 8px;
    color: var(--text);
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 600;
    padding: 8px 10px;
    text-align: center;
    outline: none;
    transition: all 0.2s;
    -moz-appearance: textfield;
  }

  .field input::-webkit-outer-spin-button,
  .field input::-webkit-inner-spin-button { -webkit-appearance: none; }

  .field input:focus {
    border-color: var(--accent);
    background: rgba(255,107,107,0.08);
  }

  .predict-btn {
    width: 100%;
    margin-top: 32px;
    padding: 18px;
    background: var(--accent);
    color: #fff;
    border: none;
    border-radius: 14px;
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    letter-spacing: 1px;
    cursor: pointer;
    transition: all 0.2s;
    text-transform: uppercase;
  }

  .predict-btn:active { transform: scale(0.98); opacity: 0.9; }
  .predict-btn:disabled { opacity: 0.5; cursor: not-allowed; }

  .result-box {
    margin-top: 20px;
    border-radius: 14px;
    padding: 22px;
    text-align: center;
    display: none;
    animation: fadeIn 0.4s ease;
  }

  @keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
  }

  .result-box.leave {
    background: rgba(255,107,107,0.12);
    border: 1px solid var(--accent);
    display: block;
  }

  .result-box.stay {
    background: rgba(107,203,119,0.12);
    border: 1px solid var(--green);
    display: block;
  }

  .result-emoji { font-size: 2.5rem; margin-bottom: 8px; }

  .result-text {
    font-family: 'Syne', sans-serif;
    font-size: 1.3rem;
    font-weight: 800;
  }

  .result-box.leave .result-text { color: var(--accent); }
  .result-box.stay  .result-text { color: var(--green); }

  .result-sub { color: var(--muted); font-size: 0.82rem; margin-top: 6px; }

  .loading { display: none; text-align: center; padding: 20px; color: var(--muted); font-size: 0.9rem; }
  .loading.active { display: block; }

  .clear-btn {
    width: 100%;
    margin-top: 10px;
    padding: 14px;
    background: transparent;
    color: var(--muted);
    border: 1px solid var(--border);
    border-radius: 14px;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.9rem;
    cursor: pointer;
    transition: all 0.2s;
  }
  .clear-btn:active { opacity: 0.7; }
</style>
</head>
<body>

<header>
  <h1>Attrition <span>Predictor</span></h1>
  <p>Powered by Random Forest · ML Model</p>
</header>

<div class="container">

  <div class="section-title">Personal Info</div>
  <div class="field"><div class="field-info"><div class="field-name">Age</div><div class="field-range">18 – 60</div></div><input type="number" id="Age" placeholder="35"></div>
  <div class="field"><div class="field-info"><div class="field-name">Distance From Home</div><div class="field-range">1 – 29 km</div></div><input type="number" id="DistanceFromHome" placeholder="10"></div>
  <div class="field"><div class="field-info"><div class="field-name">Education</div><div class="field-range">1 – 5</div></div><input type="number" id="Education" placeholder="3"></div>
  <div class="field"><div class="field-info"><div class="field-name">Num Companies Worked</div><div class="field-range">0 – 9</div></div><input type="number" id="NumCompaniesWorked" placeholder="2"></div>
  <div class="field"><div class="field-info"><div class="field-name">Total Working Years</div><div class="field-range">0 – 40</div></div><input type="number" id="TotalWorkingYears" placeholder="10"></div>

  <div class="section-title">Job Details</div>
  <div class="field"><div class="field-info"><div class="field-name">Job Level</div><div class="field-range">1 – 5</div></div><input type="number" id="JobLevel" placeholder="2"></div>
  <div class="field"><div class="field-info"><div class="field-name">Job Involvement</div><div class="field-range">1 – 4</div></div><input type="number" id="JobInvolvement" placeholder="3"></div>
  <div class="field"><div class="field-info"><div class="field-name">Job Satisfaction</div><div class="field-range">1 – 4</div></div><input type="number" id="JobSatisfaction" placeholder="3"></div>
  <div class="field"><div class="field-info"><div class="field-name">Monthly Income</div><div class="field-range">1009 – 19999</div></div><input type="number" id="MonthlyIncome" placeholder="5000"></div>
  <div class="field"><div class="field-info"><div class="field-name">OverTime</div><div class="field-range">0 = No, 1 = Yes</div></div><input type="number" id="OverTime" placeholder="0"></div>
  <div class="field"><div class="field-info"><div class="field-name">Percent Salary Hike</div><div class="field-range">11 – 25</div></div><input type="number" id="PercentSalaryHike" placeholder="15"></div>

  <div class="section-title">Satisfaction & Growth</div>
  <div class="field"><div class="field-info"><div class="field-name">Environment Satisfaction</div><div class="field-range">1 – 4</div></div><input type="number" id="EnvironmentSatisfaction" placeholder="3"></div>
  <div class="field"><div class="field-info"><div class="field-name">Work Life Balance</div><div class="field-range">1 – 4</div></div><input type="number" id="WorkLifeBalance" placeholder="3"></div>
  <div class="field"><div class="field-info"><div class="field-name">Training Times Last Year</div><div class="field-range">0 – 6</div></div><input type="number" id="TrainingTimesLastYear" placeholder="3"></div>
  <div class="field"><div class="field-info"><div class="field-name">Years At Company</div><div class="field-range">0 – 40</div></div><input type="number" id="YearsAtCompany" placeholder="5"></div>
  <div class="field"><div class="field-info"><div class="field-name">Years Since Last Promotion</div><div class="field-range">0 – 15</div></div><input type="number" id="YearsSinceLastPromotion" placeholder="2"></div>
  <div class="field"><div class="field-info"><div class="field-name">Years With Curr Manager</div><div class="field-range">0 – 17</div></div><input type="number" id="YearsWithCurrManager" placeholder="3"></div>

  <button class="predict-btn" onclick="predict()">PREDICT</button>
  <button class="clear-btn" onclick="clearAll()">CLEAR ALL</button>

  <div class="loading" id="loading">⏳ Analysing employee data...</div>

  <div class="result-box" id="result">
    <div class="result-emoji" id="result-emoji"></div>
    <div class="result-text" id="result-text"></div>
    <div class="result-sub" id="result-sub"></div>
  </div>

</div>

<script>
const features = ["Age","DistanceFromHome","Education","EnvironmentSatisfaction",
  "JobInvolvement","JobLevel","JobSatisfaction","MonthlyIncome",
  "NumCompaniesWorked","OverTime","PercentSalaryHike","TotalWorkingYears",
  "TrainingTimesLastYear","WorkLifeBalance","YearsAtCompany",
  "YearsSinceLastPromotion","YearsWithCurrManager"];

async function predict() {
  const values = features.map(f => parseFloat(document.getElementById(f).value));

  if (values.some(isNaN)) {
    alert("Please fill in all fields!");
    return;
  }

  const btn = document.querySelector('.predict-btn');
  btn.disabled = true;
  document.getElementById('loading').classList.add('active');
  document.getElementById('result').className = 'result-box';

  try {
    const res = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ features: values })
    });

    const data = await res.json();
    const leaving = data.prediction === "WILL LEAVE";

    const box = document.getElementById('result');
    box.className = 'result-box ' + (leaving ? 'leave' : 'stay');
    document.getElementById('result-emoji').textContent = leaving ? '🚨' : '✅';
    document.getElementById('result-text').textContent = data.prediction;
    document.getElementById('result-sub').textContent = leaving
      ? 'This employee is at high risk of leaving.'
      : 'This employee is likely to stay with the company.';

  } catch(e) {
    alert("Error connecting to server!");
  }

  btn.disabled = false;
  document.getElementById('loading').classList.remove('active');
}

function clearAll() {
  features.forEach(f => document.getElementById(f).value = '');
  document.getElementById('result').className = 'result-box';
}
</script>
</body>
</html>
"""

@app.route("/")
def home():
    return render_template_string(HTML)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["features"]
    scaled = scaler.transform([data])
    result = model.predict(scaled)[0]
    label = "WILL LEAVE" if result == 1 else "WILL STAY"
    return jsonify({"prediction": label})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)