#####################################################################
# ai.lab.io Requirements ClassificationByArrayDataset_Long
#####################################################################
from flask import Flask, jsonify, request, send_file
import model
#####################################################################
# ai.lab.io Configurations ClassificationByArrayDataset_Long
#####################################################################
SQLiteConnectionPath = ""
app = Flask(__name__) 
#####################################################################
# ai.lab.io Services ClassificationByArrayDataset_Long
#####################################################################
@app.route('/api/trainModel', methods = ['POST'])
def trainModel():
    if(request.method == 'POST'):
        batch_size = request.form.get("batch_size")
        input_size = request.form.get("input_size")
        hidden_layers = request.form.get("hidden_layers")
        num_classes = request.form.get("num_classes")
        learning_rate = request.form.get("learning_rate")
        iterations = request.form.get("iterations")
        res = model.train(eval(batch_size), eval(input_size), eval(hidden_layers), eval(num_classes), eval(learning_rate), eval(iterations))
        return res, 200
    
@app.route('/api/employModel', methods = ['POST'])
def employModel():
    if(request.method == 'POST'):
        values = request.form.get("values")
        res = model.employModel(values)
        return res, 200
    
if __name__ == '__main__': 
  
    app.run(debug = True) 
