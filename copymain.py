from flask import Flask, render_template, request, redirect
from flask_bootstrap import Bootstrap
import webbrowser
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)

app.config["IMAGE_UPLOADS"]= "/Users/gabrieldumbrille/Desktop/COPYFMP/uploads/unknown_photo"

@app.route("/", methods=["GET", "POST"])

def upload_img():

    if request.method == "POST":

        if request.files:

            image = request.files["image"]

            image.save(os.path.join(app.config["IMAGE_UPLOADS"], image.filename))

            print("Image saved!")

            return redirect(request.url)

    return render_template("upload_img.html", methods=["GET", "POST"])

@app.route("/find_my_photo")

def find_my_photo():
    photo = '/Users/gabrieldumbrille/Desktop/COPYFMP/uploads'
    locations = pd.read_csv(r'/Users/gabrieldumbrille/Desktop/key.csv')
    model = load_model(r'/Users/gabrieldumbrille/Desktop/MobileNet_Model.h5')

    pred = ImageDataGenerator(rescale=1./255)

    pred = pred.flow_from_directory(
        photo,
        target_size=(160, 160),
        batch_size=1,
        shuffle=False
        )

    predictions = model.predict(pred, steps=1)

    preds = np.argmax(predictions,axis=1)

    answer = locations.loc[locations['name'].index == preds[0], 'coord']

    street_view =  (''.join(list(answer))).replace(' ', '')

    return webbrowser.open_new_tab(
        'http://maps.google.com/maps?q=&layer=c&cbll=' + street_view)

if __name__ == "__main__":
    app.run(debug=True)
