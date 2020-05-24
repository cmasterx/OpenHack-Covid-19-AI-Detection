from flask import Flask, jsonify, request, render_template, flash, request, redirect, url_for, send_from_directory
import os, io, json
import pandas as pd
import pickle
from flask_cors import CORS

from werkzeug.utils import secure_filename

UPLOAD_FOLDER = './downloads/'
FILES_DIRECTORY = './files'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

PUBLIC_DIR = './public'

DOMAIN_NAME = ''

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
CORS(app)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def default_index():
    return default('')

@app.route('/<path:path>', methods=['GET', 'POST'])
def default(path):
    if request.method == 'GET':

        print('Path :"{}"'.format(path))
        
        if not path:
            path = 'index.html'

        return send_from_directory(PUBLIC_DIR, path)
        

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=5000)