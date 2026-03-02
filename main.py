from flask import Flask, render_template_string
import joblib
import numpy as np
import random
import json

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
target_encoder = joblib.load("target_encoder.pkl")
feature_columns = joblib.load("feature_columns.pkl")

app = Flask(__name__)

DEMO_FAILURES = [
    "Heat Dissipation Failure",
    "Overstrain Failure",
    "Tool Wear Failure"
]

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
<title>Smart Factory AI Dashboard</title>

<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

<meta http-equiv="refresh" content="5">

<style>
body{
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
    font-family: 'Segoe UI', sans-serif;
}

.card-custom{
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(10px);
    border-radius: 15px;
    padding: 20px;
    margin-bottom: 20px;
    box-shadow: 0 0 20px rgba(0,0,0,0.4);
}

.status-normal{
    color: #00ff9d;
    font-size: 28px;
    font-weight: bold;
}

.status-failure{
    color: #ff4d4d;
    font-size: 28px;
    font-weight: bold;
}

.big-title{
    font-size: 30px;
    font-weight: bold;
    margin-bottom: 20px;
}

.sensor-value{
    font-size: 22px;
    font-weight: bold;
}
</style>
</head>

<body>

<div class="container mt-5">

<div class="text-center big-title">
 SMART FACTORY AI MONITORING SYSTEM
</div>

<div class="card-custom text-center">
    <h4>Machine Status</h4>
    <div class="{{ 'status-failure' if status=='Failure' else 'status-normal' }}">
        {{status}}
    </div>
    <h5 class="mt-2">Failure Type: {{failure_type}}</h5>
</div>

<div class="row">
    {% for key,value in data.items() %}
    <div class="col-md-4">
        <div class="card-custom text-center">
            <h6>{{key}}</h6>
            <div class="sensor-value">{{value}}</div>
        </div>
    </div>
    {% endfor %}
</div>

<div class="row mt-4">
    <div class="col-md-6">
        <div class="card-custom">
            <canvas id="tempChart"></canvas>
        </div>
    </div>

    <div class="col-md-6">
        <div class="card-custom">
            <canvas id="rpmChart"></canvas>
        </div>
    </div>
</div>

</div>

<script>
new Chart(document.getElementById("tempChart"), {
    type: "bar",
    data: {
        labels: ["Air Temp", "Process Temp"],
        datasets: [{
            label: "Temperature",
            data: {{ temp_data|safe }},
            backgroundColor: ["#00c6ff", "#ff9966"]
        }]
    }
});

new Chart(document.getElementById("rpmChart"), {
    type: "doughnut",
    data: {
        labels: ["RPM"],
        datasets: [{
            data: {{ rpm_data|safe }},
            backgroundColor: ["#00ff9d"]
        }]
    }
});
</script>

</body>
</html>
"""

@app.route("/")
def home():

    values = []

    for col in feature_columns:
        name = col.lower()

        if "air" in name:
            values.append(round(random.uniform(290,330),2))
        elif "process" in name:
            values.append(round(random.uniform(300,350),2))
        elif "rotational" in name:
            values.append(random.randint(500,2500))
        elif "torque" in name:
            values.append(round(random.uniform(20,100),2))
        elif "tool" in name:
            values.append(random.randint(0,300))
        elif "type" in name:
            values.append(random.randint(0,2))
        else:
            values.append(0)

    input_array = np.array(values).reshape(1,-1)
    input_scaled = scaler.transform(input_array)

    pred = model.predict(input_scaled)
    predicted_label = target_encoder.inverse_transform(pred)[0]

    if predicted_label == "No Failure":
        if random.random() < 0.4:
            failure_type = random.choice(DEMO_FAILURES)
            status = "Failure"
        else:
            failure_type = "No Failure"
            status = "No Failure"
    else:
        failure_type = predicted_label
        status = "Failure"

    display_data = dict(zip(feature_columns, values))

    air_temp = next((v for k,v in display_data.items() if "air" in k.lower()), 0)
    process_temp = next((v for k,v in display_data.items() if "process" in k.lower()), 0)
    rpm = next((v for k,v in display_data.items() if "rotational" in k.lower()), 0)

    return render_template_string(
        HTML_PAGE,
        data=display_data,
        status=status,
        failure_type=failure_type,
        temp_data=json.dumps([air_temp, process_temp]),
        rpm_data=json.dumps([rpm])
    )

if __name__ == "__main__":
    app.run(debug=True)