from flask import Flask, request, jsonify
import uuid
from classifier import Classifier

app = Flask(__name__)

@app.route('/')
def index():
    return "Server up and running"

@app.route('/predict/<model>', methods=['POST'])
def predict(model=None):
    if request.method == 'POST':
        file = request.files['upload']
        fileid = uuid.uuid1()
        filepath = f'./uploads/{fileid}.jpg'
        file.save(filepath)
        classifier = Classifier()
        prediction = classifier.get_prediction(filepath, model)
        return jsonify(prediction[0])