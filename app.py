from flask import Flask, request, jsonify
import os
from werkzeug.utils import secure_filename
import urllib.request
from classifier import Classifier

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.join(FILE_DIR, os.pardir) 
BASE_DIR = os.path.join(PARENT_DIR, 'app')

if os.path.exists(os.path.join(BASE_DIR, 'fruit-v1.pkl')) == False:
    urllib.request.urlretrieve('https://elasticbeanstalk-us-east-2-838649319005.s3.us-east-2.amazonaws.com/fruit-v1.pkl', 'fruit-v1.pkl')

UPLOAD_FOLDER = 'uploads'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return f'Server up and running {FILE_DIR} {PARENT_DIR} {BASE_DIR}'

@app.route('/predict/<model>', methods=['POST'])
def predict(model=None):
    if request.method == 'POST':
        file = request.files['upload']

        dest = os.path.join(BASE_DIR, app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
        file.save(dest)

        classifier = Classifier(BASE_DIR)
        prediction = classifier.get_prediction(dest, model)
        return jsonify(prediction[0])
