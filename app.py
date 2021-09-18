import pickle

import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template

# create flask app

app = Flask(__name__)

# load pickle

model = pickle.load(open("model.pkl", "rb"))


@app.route('/')
def home():
    return render_template('index.html')


@app.route("/predict", methods=['POST'])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = np.array(float_features).reshape(1, -1)

    prediction = model.predict(features)

    if prediction == 1:
        res = "likely"
    else:
        res = "not likely"
        # return res
    return render_template('finalprediction.html', prediction=res)


if __name__ == '__main__':
    app.run()
    
