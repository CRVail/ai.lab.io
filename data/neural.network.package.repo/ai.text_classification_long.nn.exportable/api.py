#####################################################################
# ai.lab.io Requirements
#####################################################################
from flask import Flask, jsonify, request, send_file
import random
import json
#import torch
import sqlite3
import requests
import model
#####################################################################
# ai.lab.io Configurations
#####################################################################
SQLiteConnectionPath = ""
app = Flask(__name__) 
#####################################################################
# ai.lab.io Services
#####################################################################
@app.route('/api/trainModel', methods = ['POST'])
def trainModel():
    if(request.method == 'POST'):
        input_size = request.form.get("input_size")
        hidden_layers = request.form.get("hidden_layers")
        num_classes = request.form.get("num_classes")
        learning_rate = request.form.get("learning_rate")
        iterations = request.form.get("iterations")
        print(input_size, hidden_layers, num_classes, learning_rate, iterations)
        res = model.train(eval(input_size), eval(hidden_layers), eval(num_classes), eval(learning_rate), eval(iterations))
        return res, 200

@app.route('/api/employModel', methods = ['POST'])
def employModel():
    if(request.method == 'POST'):
        values = request.form.get("values")
        res = model.employModel(values)
        return res, 200
   
if __name__ == '__main__': 
  
    app.run(debug = True) 
