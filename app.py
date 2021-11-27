from flask import Flask, request, jsonify
import os
from werkzeug.utils import secure_filename
import urllib.request
from classifier import Classifier

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.join(FILE_DIR, os.pardir) 
base_dir = os.path.join(PARENT_DIR, 'src')

if os.path.exists(os.path.join(base_dir, 'fruit-v1.pkl')) == False:
    urllib.request.urlretrieve('https://elasticbeanstalk-us-east-2-838649319005.s3.us-east-2.amazonaws.com/fruit-v1.pkl', 'fruit-v1.pkl')

UPLOAD_FOLDER = './uploads'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return "Server up and running"

@app.route('/predict/<model>', methods=['POST'])
def predict(model=None):
    if request.method == 'POST':
        file = request.files['upload']

        dest = os.path.join(base_dir, app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
        file.save(dest)

        classifier = Classifier(base_dir)
        prediction = classifier.get_prediction(dest, model)
        return jsonify(prediction[0])
