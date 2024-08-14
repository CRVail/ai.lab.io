#####################################################################
# ai.lab.io Requirements
#####################################################################
from flask import Flask, jsonify, request, send_file
import random
import json
#import torch
import sqlite3
import requests
from components import ai_observe_utilities
from components import ai_observe_models
from components import ai_observe_exporter

#####################################################################
# ai.lab.io Configurations
#####################################################################
SQLiteConnectionPath = ""
app = Flask(__name__) 
#####################################################################
# ai.lab.io Services
#####################################################################
@app.route('/api/dataViewer', methods = ['GET', 'POST'])
def dataViewer():
    if(request.method == 'GET'):
        table = request.args.get('dataset')
        volumeName = request.args.get('volumeName')
        user_address = request.remote_addr
        statement = "SELECT * FROM " + table
        rows = ai_observe_utilities.volumeRead(ai_observe_utilities.getVolConnString(volumeName), statement)
        return rows, 200
    if(request.method == 'POST'):
        volumeName = request.form.get("volumeName")
        user_address = request.remote_addr
        statement = request.form.get("statement")
        rows = ai_observe_utilities.volumeRead(volumeName, statement)
        return rows, 200

@app.route('/api/newNNVolume', methods = ['POST'])
def newNNVolume():
    if(request.method == 'POST'):
        volumeName = request.form.get("volumeName")
        volumeDesc = request.form.get("volumeDesc")
        user_address = request.remote_addr
        msg = "The Neural Network Volume Builder process completed! ai.lab.io attempted a volume build and initialization of Neural Network Volume: '" + volumeName + "'. Please review the 'init' key for a full summary." 
        init = ai_observe_utilities.initialize_project_volume(volumeName, volumeDesc, user_address)
        res = {
            "User": str(user_address),
            "System": msg,
            "init": init
        }
        return jsonify(res) , 200

@app.route('/api/describeVolumeTable', methods = ['GET'])
def describeVolumeTable():
    if(request.method == 'GET'):
        table = request.args.get('table')
        volumeName = request.args.get('volumeName')
        res = ai_observe_utilities.describeVolumeTable(ai_observe_utilities.getVolConnString(volumeName), table)
        return res, 200

@app.route('/api/describeVolume', methods = ['GET', 'POST'])
def describeVolume():
    if(request.method == 'GET'):
        volumeName = request.args.get('volumeName')
        res = ai_observe_utilities.describeVolume(ai_observe_utilities.getVolConnString(volumeName))
        return res, 200
    elif(request.method == 'POST'):
        volumeName = request.args.get('volumeName')
        res = ai_observe_utilities.describeVolume('data/' + volumeName)
        return res, 200

@app.route('/api/trainModel', methods = ['POST'])
def trainModel():
    if(request.method == 'POST'):
        user_address = request.remote_addr
        volumeName = request.form.get("volumeName")
        inputFile = request.form.get("inputFile")
        type = request.form.get("type")
        batch_size = request.form.get("batch_size")
        input_size = request.form.get("input_size")
        hidden_layers = request.form.get("hidden_layers")
        num_classes = request.form.get("num_classes")
        learning_rate = request.form.get("learning_rate")
        iterations = request.form.get("iterations")
        print(volumeName, inputFile, type, batch_size, input_size, hidden_layers, num_classes, learning_rate, iterations, user_address)
        res = ai_observe_models.train(volumeName, inputFile, type, eval(batch_size), eval(input_size), eval(hidden_layers), eval(num_classes), eval(learning_rate), eval(iterations), user_address)
        return res, 200

@app.route('/api/employModel', methods = ['POST'])
def employModel():
    if(request.method == 'POST'):
        user_address = request.remote_addr
        volumeName = request.form.get("volumeName")
        model = request.form.get("model")
        type = request.form.get("type")
        values = request.form.get("values")
        res = ai_observe_models.employModel(volumeName, type, model, values, user_address)
        return res, 200

@app.route('/api/upload', methods = ['POST'])
def upload():
    import os
    if(request.method == 'POST'): 
        res = []
        user_address = request.remote_addr
        volumeName = request.form.get("volumeName")
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        else:
            file = request.files['file']
            if file:
                path = 'data/neural.network.data.volumes/' + volumeName + '/input/'
                fileName = path + file.filename
                file.save(fileName)
                msg = file.filename + " uploaded successfully to " + volumeName
                res.append(msg)
                msg2 = ai_observe_utilities.logEvent(volumeName, "File Upload", msg, user_address)
                res.append(msg2)
                return jsonify(res), 200
        
    else:
        return "Failed!", 500
 
@app.route('/api/logMetric', methods = ['POST'])
def logMetric():
    if(request.method == 'POST'): 
        user_address = request.remote_addr
        volumeName = request.form.get("volumeName")
        datasetName = request.form.get("datasetName")
        input = request.form.get("input")
        output = request.form.get("output")
        msg = ai_observe_utilities.logMetric(volumeName, datasetName, input, output, user_address)
        return jsonify(msg), 200        
    else:
        return "Failed!", 500

@app.route('/api/removeFile', methods = ['POST'])
def removeFile():
    if(request.method == 'POST'): 
        user_address = request.remote_addr
        volumeName = request.form.get("volumeName")
        io = request.form.get("io")
        file = request.form.get("file")
        res = []
        msg = ai_observe_utilities.removeFile(volumeName, io, file)
        msg2 = ai_observe_utilities.logEvent(volumeName, "File Deleted", msg, user_address)
        res.append(msg)
        res.append(msg2)
        return jsonify(res), 200        
    else:
        return "Failed!", 500
  
@app.route('/api/initDataset', methods = ['POST'])
def initDataset():
    if(request.method == 'POST'): 
        user_address = request.remote_addr
        volumeName = request.form.get("volumeName")
        datasetName = request.form.get("datasetName")
        msg = ai_observe_utilities.initialize_project_dataset(volumeName, datasetName, user_address)
        return jsonify(msg), 200        
    else:
        return "Failed!", 500
  
@app.route('/api/initObserve', methods = ['POST'])
def initObserve():
    if(request.method == 'POST'): 
        key = request.form.get("key")  
        user_address = request.remote_addr
        if key == 'v4ven4life':
            msg = ai_observe_utilities.initialize_ai_observe_data_utilities(user_address)
            return jsonify(msg), 200
        else:
            return "Invalid Key!", 200
    else:
        return "Failed!", 500

@app.route('/api/dropTable', methods = ['POST'])
def dropTable():
    if(request.method == 'POST'):
        user_address = request.remote_addr
        key = request.form.get("key")
        volumeName = request.form.get("volumeName")
        useConnector = request.form.get("useConnector")
        table = request.form.get("table")
        if useConnector == "True":
            volumeName = ai_observe_utilities.getVolConnString(volumeName)
        else:
            volumeName = 'data/' + volumeName
        if key == 'v4ven4life':
            res = ai_observe_utilities.dropTable(volumeName, table)
            return res, 200
        else:
            return "Invalid Key!", 200
      
@app.route('/api/commitStatement', methods = ['POST'])
def commitStatement():
    if(request.method == 'POST'):
        user_address = request.remote_addr
        key = request.form.get("key")
        volumeName = request.form.get("volumeName")
        useConnector = request.form.get("useConnector")
        statement = request.form.get("statement")
        if useConnector == "True":
            volumeName = ai_observe_utilities.getVolConnString(volumeName)
        else:
            volumeName = 'data/' + volumeName
        if key == 'v4ven4life':
            res = ai_observe_utilities.volumeCommit(volumeName, statement)
            return res, 200
        else:
            return "Invalid Key!", 200

@app.route('/api/exportModel', methods = ['GET'])
def exportModel():
    if(request.method == 'GET'):
        volumeName = request.args.get('volumeName')
        modelName = request.args.get('modelName')
        export = ai_observe_exporter.export_model(volumeName, modelName)
        return send_file(export["downloadPath"], as_attachment=True, download_name=export["downloadName"]), 200
 
@app.route('/api/testAPI', methods = ['GET'])
def testAPI():
    if(request.method == 'GET'):
        #TEST CODE HERE
        return "Nothing being tested at this time", 200
    
if __name__ == '__main__': 
  
    app.run(debug = True) 
