# app.py
from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('population_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    area = float(request.form['area'])
    density = float(request.form['density'])
    growth_rate = float(request.form['growth_rate'])

    # Make prediction
    prediction = model.predict(np.array([[area, density, growth_rate]]))
    
    return render_template('result.html', prediction=prediction[0])

if __name__ == "__main__":
    app.run(debug=True)