from sklearn.feature_extraction.text import CountVectorizer
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
vectorizer = CountVectorizer()

TrainingPath = 'trainingData.json'
ModelPath = 'text_classification_long.model'
ModelType = "ai.text_classification_long"

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

def TextClassification_Long(dataPath):
        inputs = []
        labels = []
        with open(dataPath, 'r') as f:
            data = json.load(f)
        for d in data["data"]:
            inputs.append(d['input'])
            labels.append(d['label'])
        res = {
            "inputs": inputs,
            "labels": labels
        }
        return res



def dataLoader(path):
    dataset = TextClassification_Long(path)
    return dataset
    
def train(inputSize, hiddenLayers, numClasses, learningRate, iterations):
    dataPath = TrainingPath
    savePath = ModelPath
    finalLoss = 0
    vectorizer = CountVectorizer()
    intentsData = dataLoader(dataPath)
    X = vectorizer.fit_transform(intentsData["inputs"]).toarray()
    y = torch.tensor(intentsData["labels"])
    num_epochs = iterations
    input_size = X.shape[1]
    inputSize = input_size
    learning_rate = learningRate
    model = NeuralNetwork(input_size, hiddenLayers, numClasses)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()  # Clear the gradients
        outputs = model(X_tensor)  # Forward pass
        loss = criterion(outputs, y)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights
        finalLoss = loss.item()
        print(f'Iteration [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
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
            "inputs": intentsData["inputs"],
            "outputs": outputs,
            "definitions": definitions
        }
        torch.save(modelData, savePath)
        print("model saved")
    resObj = {
        "messages": "Complete",
        "iterations": str(iterations)
    }
    return resObj

def employModel(values):
    modelPath = ModelPath
    data = torch.load(modelPath)
    definitions = data["definitions"]
    inputSize = data["input_size"]
    hiddenLayers = data["hidden_layers"]
    numClasses =  data["num_classes"]
    modelState = data["model_state"]
    inputs = data["inputs"]
    labels = data["outputs"]
    X = vectorizer.fit_transform(inputs).toarray()
    y = torch.tensor(labels)
    model = NeuralNetwork(inputSize, hiddenLayers, numClasses)
    model.load_state_dict(modelState)
    model.eval()
    testData = str(values)
    req_texts = eval(testData)
    new_X = vectorizer.transform(req_texts).toarray()
    new_X_tensor = torch.tensor(new_X, dtype=torch.float32)
    with torch.no_grad():
        outputs = model(new_X_tensor)
        _, predicted = torch.max(outputs, 1)
    res = []
    for text, pred in zip(req_texts, predicted):
        if definitions == "None":
            s = {
                "prompt": text,
                "answer" : str(pred.item())
            }
            res.append(s)
        else:
            for definedAs in definitions: 
                if pred.item() == definedAs["label"]:
                    s = {
                        "prompt": text,
                        "answer" : definedAs["definedAs"]
                    }
                    res.append(s)
    return res