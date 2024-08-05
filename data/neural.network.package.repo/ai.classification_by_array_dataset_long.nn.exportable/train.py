import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from components import ai_observe_utilities
from components import ai_observe_datasets
import sys
import json
import warnings
warnings.simplefilter("ignore", category=Warning)
import random
from model import NeuralNet

class ClassificationByArrayDataset_Long(Dataset):
    def __init__(self, json_file):
        with open(json_file, 'r') as f:
            data = json.load(f)
        self.inputs = [torch.tensor(d['input'], dtype=torch.float32) for d in data['data']]
        self.labels = [torch.tensor(d['label'], dtype=torch.long) for d in data['data']]
    def __len__(self):
        return len(self.inputs)
    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]

def dataLoader(path, batchSize):
        dataset = ClassificationByArrayDataset_Long(path)
        dataloader = DataLoader(dataset, batch_size=batchSize, shuffle=True)
        return dataloader

def train(volume, file, type, batchSize, inputSize, hiddenLayers, numClasses, learningRate, interations, IP):    
    path = 'data/neural.network.data.volumes/' + volume
    dataPath = path + '/input/' + file 
    savePath = path + '/output/' + file + '.model'
    finalLoss = 0
    #TRAINING SCRIPT: ClassificationByArrayDataset_Long
    dataloader = dataLoader(dataPath, type, batchSize)
    model = NeuralNet(inputSize, hiddenLayers, numClasses)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learningRate)
    for epoch in range(interations):
        for inputs, labels in dataloader:
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)        
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        print(f'Iteration [{epoch+1}/{interations}], Loss: {loss.item():.4f} \r')
    with open(dataPath, 'r') as f:
        data = json.load(f)
    if "definitions" in data:
        definitions = data["definitions"]
    else:
        definitions = "None"
    modelData = {
            "model_state": model.state_dict(),
            "input_size": inputSize,
            "hidden_layers": hiddenLayers,
            "num_classes": numClasses,
            "type": type,
            "inputs": inputs,
            "outputs": outputs,
            "definitions": definitions
    }
    torch.save(modelData, savePath)
    print("model saved")
    modelName = file + ".model"
    sMsg = "Training Complete! Model " + modelName + " has been created."
    logModelData = ai_observe_utilities.logModel(volume, modelName, type, dataPath)
    msg = ai_observe_utilities.logEvent(volume, "Model Created!", sMsg, IP)
    msgObj = []
    msgObj.append(logModelData)
    msgObj.append(sMsg)
    msgObj.append(msg)
    resObj = {
        "messages": msgObj,
        "modelName": str(file + ".model"),
        "finalLoss" : str(finalLoss),
        "definitions": definitions,
        "rawModel": str(model.state_dict())
    }
    return resObj