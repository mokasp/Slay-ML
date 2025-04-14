import os
import base64
import cv2
import numpy as np
from flask import Flask, request, render_template
from datetime import datetime as dt
import tensorflow as tf
import logging
from prepare_input import display_colors, model_input, display_prediction


app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)



@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """ route that gets and displays color predicition for lip color """

    # load in model
    model = tf.keras.models.load_model('model/test_model_00.keras')

    # get image taken from webcam
    if request.method == 'POST':
        data_url = request.form['image']  # base64 string

        if data_url.startswith('data:image'):
            header, encoded = data_url.split(",", 1)
            # get image from webcam as numpy array
            image_data = base64.b64decode(encoded)
            np_arr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if img is not None:
                # get image as base64 for viewing
                _, buffer = cv2.imencode('.jpg', img)
                img_base64 = base64.b64encode(buffer).decode('utf-8')
                logging.debug("/predict route was hit!")
                logging.debug(f"📸 Image data starts with: {request.form['image'][:30]}")

                # get lip color prediction
                prediction = model_input(img, model)
                logging.debug(f"prediction: {prediction}")

                # get lip color as image and convert to base64 for viewing
                output = display_prediction(prediction)
                _, buffer = cv2.imencode('.jpg', output)
                output64 = base64.b64encode(buffer).decode('utf-8')


                return f'<h2>Image received!</h2><img src="data:image/jpeg;base64,{img_base64}" width="300"><img src="data:image/jpeg;base64,{output64}" width="300">'
            else:
                return 'Error decoding image'
        else:
            return 'Invalid image data'
    return render_template('index.html')