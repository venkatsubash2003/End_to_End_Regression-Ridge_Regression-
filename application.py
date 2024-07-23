from flask import Flask,render_template,jsonify,request
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application

scaler = pickle.load(open("scaler.pkl","rb"))
model = pickle.load(open("model.pkl","rb"))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/datapredict",methods=["POST","GET"])
def predict_data():
    if request.method == "GET":
        return render_template("home.html")
    else:
        Temperature = float(request.form.get("Temperature"))
        RH = float(request.form.get("RH"))
        Ws = float(request.form.get("Ws"))
        Rain = float(request.form.get("Rain"))
        FFMC = float(request.form.get("FFMC"))
        DMC = float(request.form.get("DMC"))
        ISI = float(request.form.get("ISI"))
        Classes = float(request.form.get("Classes"))
        Region = float(request.form.get("Region"))

        test_data = scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        results = model.predict(test_data)
        return render_template("home.html", results = results[0])



if __name__ == "__main__":
    app.run(debug=True) 
