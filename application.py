from flask import Flask, jsonify, render_template, request
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application

# Import Ridge regressor and Standard Scaler
ridge_model = pickle.load(open("models/Ridge.pkl", "rb"))
scaler_model = pickle.load(open("models/Scaler.pkl", "rb"))

@app.route("/")
def index() :
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        Temperature = float(request.form["Temperature"])
        RH = float(request.form["RH"])
        Ws = float(request.form["Ws"])
        Rain = float(request.form["Rain"])
        FFMC = float(request.form["FFMC"])
        DMC = float(request.form["DMC"])
        ISI = float(request.form["ISI"])
        Classes = float(request.form["Classes"])
        Region = float(request.form["Region"])
        
        new_scaled_data = scaler_model.transform([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
        result = ridge_model.predict(new_scaled_data)[0]
        
        return render_template("home.html", result = result)
        
    return render_template("home.html")

if __name__ == "__main__":
    app.run()