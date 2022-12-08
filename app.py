# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 21:21:44 2022

@author: Banaj
"""

import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__, template_folder='templates')

model = pickle.load(open("pipeline_pkl", "rb"))


@app.route('/')

def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    features = [np.array(features)]
    
    prediction = model.predict(features)
    output = "Hadron"
    if(prediction == 0):
        output = "Gamma"
    
    return render_template("index.html", prediction_text=" Class Detected : {} ".format(output))


if __name__ == '__main__':
    app.run()