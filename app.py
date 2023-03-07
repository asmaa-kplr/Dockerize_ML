import os


import numpy as np
from keras.models import load_model
import keras.utils as image
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import cv2


app = Flask(__name__)


MODEL_PATH = './static/model/tounsiModel.h5'


model = load_model(MODEL_PATH)


def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(28, 28)) 
    img_array = np.asarray(img)
    x = cv2.cvtColor(img_array,cv2.COLOR_BGR2GRAY)
    result = int(img_array[0][0][0])
    print(result)
    if result > 128:
      img = cv2.bitwise_not(x)
    else:
      img = x
    img = img/255
    img = (np.expand_dims(img,0)) 

    preds =  model.predict(img)
    print(preds)
    return preds


@app.route('/', methods=['GET'])
def index():
   
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        
        f = request.files['file']

        
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, './static/upload', secure_filename(f.filename))
        f.save(file_path)

        
        preds = model_predict(file_path, model)
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        predicted_label = np.argmax(preds)
        result = class_names[predicted_label]
        return result
    return None


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
