from __future__ import division, print_function
# coding=utf-8
import os
import numpy as np
from flask import Flask, render_template, request

# Keras
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Function to check the file extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

model_path1 = 'ensemble.h5'  # Load the model
model = load_model(model_path1, compile=False)

def model_predict(image_path, model):
    print("Predicting...")
    image = load_img(image_path, target_size=(128, 128))
    image = img_to_array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)

    result = model.predict(image)
    predicted_class = np.argmax(result)
    
    print(f"Prediction result: {predicted_class}")
    
    if predicted_class == 0:
        return "NORMAL!", "result.html"        
    elif predicted_class == 1:
        return "PNEUMONIA!", "result.html"

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        print("Entered predict route.")
        
        file = request.files.get('file')
        
        if file and allowed_file(file.filename):
            filename = file.filename        
            print(f"Input posted: {filename}")
            
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)

            print("Predicting class...")
            pred, output_page = model_predict(file_path, model)
            return render_template(output_page, pred_output=pred, img_src=UPLOAD_FOLDER + filename)
        else:
            print("Invalid file format!")
            return render_template('error.html', message="Invalid file format. Please upload a PNG, JPG, or JPEG image.")
    return redirect('/index')

#if __name__ == '__main__':
    #app.run(debug=True)
