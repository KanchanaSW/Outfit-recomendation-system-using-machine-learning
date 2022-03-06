from flask import Flask, render_template, request, redirect

import numpy as np
import cv2
from keras.models import load_model

app = Flask(__name__)

predictShape = load_model('modelTF.h5')    # load classifying model
validCheck = load_model('modelCheck.h5')    # load img validation model

labels = ['hourglass', 'pear', 'rectangle']  # class labels


@app.route('/', methods=['GET'])
def start():  # put application's code here
    return render_template('index.html')


@app.route('/', methods=['POST'])
def predict():
    image_file = request.files['image_file']
    image_path = "./images/" + image_file.filename
    image_file.save(image_path)

    img1 = cv2.imread(image_path)            # load the image as a metric
    img = cv2.resize(img1, (224, 224))       # resize the image
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)   # convert color mode image
    image1 = np.expand_dims(img, axis=0)     # expand dim (1,224,224,3)

    result = validCheck.predict(image1)     # checking image validity

    if result[0][0] == 1:
        print("valid!")     # Image is valid proceed
        prediction_img = predictShape.predict(image1)      # body shape classifier model
        prediction_img = np.argmax(prediction_img)      # retrieve the output shape
        print(prediction_img)
        class_label = labels[prediction_img]
        if class_label == 'hourglass':
            return redirect("https://www.trunkclub.com/womens-style/hourglass-figure")
        if class_label == 'pear':
            return redirect("https://www.trunkclub.com/womens-style/pear-shaped-body")
        if class_label == 'rectangle':
            return redirect("https://www.trunkclub.com/womens-style/rectangle-body-shape")

        return render_template('index.html', prediction=class_label)
    else:
        print("invalid!")
        class_label = "invalid image"
        return render_template('index.html', prediction=class_label)


if __name__ == '__main__':
    app.run()


#https://www.trunkclub.com/womens-style/rectangle-body-shape
#https://www.trunkclub.com/womens-style/pear-shaped-body
#https://www.trunkclub.com/womens-style/hourglass-figure