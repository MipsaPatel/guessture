import os

from flask import render_template, jsonify, request
from werkzeug.utils import secure_filename

from model.predict import load_model, predict
from . import app

model = load_model()


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/guess', methods=['POST'])
def guess():
    file = request.files['video-file']
    path = os.path.join('uploads', secure_filename(file.filename))
    file.save(path)
    output = predict(model, path)
    return jsonify({'result': output})
