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

def printi(msg, indicator='-> '):
    """Prints string on to console with indicator prefix"""
    print(indicator + msg)

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
    
    print('\n----- Starting Breather Webserver -----\n')
    print('Github: https://github.com/cmasterx/OpenHack-Covid-19-AI-Detection')
    print('Author: Charlemagne Wong, Ben Costa')
    print('')

    printi('Initializing setup')
    # load configurations
    deployment = {}
    configurations = {}

    # load deployment file
    printi('Loading deployment.json')
    with open('./deployment.json', 'r') as file:
        deployment = json.load(file)
    printi('Deployment settings loaded')
    
    # check for config file and create if not exist
    printi('Checking configuration file')
    if not os.path.exists('./config.json'):
        printi('Configurations file not found. Creating...')
        with open('./config.json', 'w+') as file:
            json.dump(deployment['default-deployment'], file)
            printi('Success!')
    
    # load config file
    printi('Loading configuration file')
    with open('./config.json', 'r') as file:
        configurations = json.load(file)
        printi('Configurations file loaded')

    # load deployment settings
    deployment_type = configurations['deployment'] if 'deployment' in configurations else deployment['default-deployment']['deployment']
    if not deployment_type in deployment['deployment']:
        deployment_type = deployment['default-deployment']['deployment']
    
    deployment_settings = deployment['deployment'][deployment_type]
    printi('Using deployment type: < "{}" - {} >'.format(deployment_type, deployment_settings))

    printi('Starting Web Server')
    print('')
    app.run(host=deployment_settings['host'],port=deployment_settings['port'])