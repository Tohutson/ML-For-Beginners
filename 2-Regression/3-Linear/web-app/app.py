import numpy as np 
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

model = pickle.load(open("./3-model.pk1", "rb"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    int_features = [int(x)/100 for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0][0] * 100,2)

    return render_template(
        "index.html", prediction_text="Likely 3P%: {}%".format(output)
    )

if __name__ == "__main__":
    app.run(debug=True)