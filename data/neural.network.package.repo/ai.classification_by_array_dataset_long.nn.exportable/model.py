import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import sys
import json
import warnings
warnings.simplefilter("ignore", category=Warning)
import random

TrainingPath = 'trainingData.json'
ModelPath = 'classification_by_array_dataset_long.model'
ModelType = "ClassificationByArrayDataset_Long"

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNetwork, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size) 
        self.l2 = nn.Linear(hidden_size, hidden_size) 
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        return out

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

def train(batchSize, inputSize, hiddenLayers, numClasses, learningRate, interations):
    dataPath = TrainingPath 
    savePath = ModelPath #SQLite Model Data
    type = ModelType
    finalLoss = 0
    #TRAINING SCRIPT: ClassificationByArrayDataset_Long
    dataloader = dataLoader(dataPath, batchSize)
    model = NeuralNetwork(inputSize, hiddenLayers, numClasses)
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
    res = {
        "Status": "Complete",
        "Iterations": interations
    }
    return res

def employModel(values):
    data = torch.load(ModelPath)
    inputSize = data["input_size"]
    hiddenLayers = data["hidden_layers"]
    numClasses =  data["num_classes"]
    modelState = data["model_state"]
    model = NeuralNetwork(input_size=inputSize, hidden_size=hiddenLayers, num_classes=numClasses)
    model.load_state_dict(modelState)
    model.eval()
    val = eval(values)
    input_data = torch.tensor([val], dtype=torch.float32)
    with torch.no_grad():
        outputs = model(input_data)
        _, predicted = torch.max(outputs, 1)
    res = f'Classification Prediction: {predicted.item()}'
    return res #+ msg
    

