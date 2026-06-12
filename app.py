from flask import Flask, render_template, request
import pickle
import numpy as np
from pyexpat import features

app = Flask(__name__)
model = pickle.load(open('multiple_regression_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    season = int(request.form['season'])
    yr = int(request.form['yr'])
    mnth =int(request.form['mnth'])
    holiday =int(request.form['holiday'])
    weekday=int(request.form['weekday'])
    workingday=int(request.form['workingday'])
    weather =int(request.form['weathersit'])
    temp = float(request.form['temp'])
    atemp = float(request.form['atemp'])
    hum= float(request.form['hum'])
    windspeed = float(request.form['windspeed'])
    features =np.array([[season,yr,mnth,holiday,weekday,workingday,weather,temp,atemp,hum,windspeed,]])
    prediction = model.predict(features)
    output = round(prediction[0],0)
    return render_template('index.html',
                           prediction_text=f'Estimated Bike Rentals:{output}')

if __name__ == "__main__":
    app.run(debug=True)