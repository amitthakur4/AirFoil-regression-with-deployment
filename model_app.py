import pickle
from flask import Flask, app, jsonify, request, url_for, render_template
from flask import Response
import numpy as np
import pandas as pd

app = Flask(__name__)
model= pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    #return 'Hello World'
    return render_template('home.html')

@app.route('/predict_api', methods=["POST"])
def predict_api():
    data = request.json["data"]
    print(data)
    new_data= list(data.values())
    prediction= model.predict(np.array(new_data).reshape(1,-1))

    output = prediction[0]
    return jsonify(output)


@app.route('/predict', methods=["POST"])
def predict():
    data = [float(x) for x in request.form.values()]
    final_features = [np.array(data)]
    print(data)

    output = model.predict(final_features)[0]
    print(output)
    return render_template('home.html', prediction_text="Airfoil pressure is  {}".format(output))

if __name__ == "__main__":
    app.run(debug=True, port=9090)