import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf

from flask import Flask, render_template, request

app = Flask(__name__)

model = load_model("vgg16.h5")

text = ""

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict')
def upload():
    return render_template("predict.html")

@app.route('/output', methods=['POST'])
def predict():
    if request.method == 'POST':
        print("heyhey")
        imageFile = request.files["imagefile"]
        basepath = os.path.dirname(__file__)
        
        filepath = os.path.join(basepath, 'images', imageFile.filename)
        imageFile.save(filepath)
        
        img = image.load_img(filepath, target_size = (224,224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis = 0)
        result = model.predict(x)
        
        print(filepath)
        index = ['normal','ulcerative_colitis','polyps','esophagitis']
        
        text = str(index[np.argmax(result)])
        return render_template('output.html', text = text)
    return render_template('predict.html')
    

if __name__ == '__main__':
    app.run()
