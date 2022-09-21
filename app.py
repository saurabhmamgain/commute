import numpy as np
from flask import Flask, request, render_template
import pickle
app = Flask(__name__)
with open('models/model', 'rb') as f:
    model = pickle.load(f)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['Post'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    features = [np.array(int_features)]
    prediction = model.predict(features)

    return render_template('index.html', prediction_text='Yay!Today is a good day to commute with {}'.
                           format(prediction))


if __name__ == "__main__":
    app.run()
