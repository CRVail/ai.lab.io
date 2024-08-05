from sklearn.feature_extraction.text import CountVectorizer
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import ai_observe_utilities
import ai_observe_datasets

import sys
import json
import warnings
warnings.simplefilter("ignore", category=Warning)

# Sample text data
texts = ["hello world", "machine learning is fun", "text classification with neural networks", "hello machine learning"]

# Corresponding labels (assuming 4 classes)
labels = [0, 1, 2, 0]

# Convert text to numerical data
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts).toarray()  # Feature vectors
y = torch.tensor(labels)  # Labels


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
    
def train():
    # Define hyperparameters
    input_size = X.shape[1]  # Number of features
    hidden_size = 10
    num_classes = 3  # Number of classes (change according to your problem)
    num_epochs = 100
    learning_rate = 0.01

    # Initialize the model
    model = NeuralNetwork(input_size, hidden_size, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Convert features to tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()  # Clear the gradients
        outputs = model(X_tensor)  # Forward pass
        loss = criterion(outputs, y)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    path = '../data/neural.network.data.volumes/' + "sampleVolume"
    type = "TextClassifierDataset"
    savePath = path + '/' + 'textTest' + '.model'
    modelData = {
        "model_state": model.state_dict(),
        "input_size": input_size,
        "hidden_layers": hidden_size,
        "num_classes": num_classes,
        "type": type
    }
    torch.save(modelData, savePath)

def botPrediction():
    path = '../data/neural.network.data.volumes/' + "sampleVolume/textTest.model" 
    data = torch.load(path)
    inputSize = data["input_size"]
    hiddenLayers = data["hidden_layers"]
    numClasses =  data["num_classes"]
    modelState = data["model_state"]
    type = data["type"]
    model = NeuralNetwork(inputSize, hiddenLayers, numClasses)
    model.load_state_dict(modelState)
    model.eval()
    new_texts = ["machine learning is fun", "hello world", "text classification with neural networks"]
    new_X = vectorizer.transform(new_texts).toarray()
    new_X_tensor = torch.tensor(new_X, dtype=torch.float32)
    with torch.no_grad():
        outputs = model(new_X_tensor)
        _, predicted = torch.max(outputs, 1)
    for text, pred in zip(new_texts, predicted):
        print(f'Text: "{text}" is predicted as class {pred.item()}')



class TextClassificationBot:
    def __init__(self, model, vectorizer):
        self.model = model
        self.vectorizer = vectorizer

    def classify_text(self, text):
        self.model.eval()
        text_vector = self.vectorizer.transform([text]).toarray()
        text_tensor = torch.tensor(text_vector, dtype=torch.float32)
        
        with torch.no_grad():
            outputs = self.model(text_tensor)
            _, predicted = torch.max(outputs, 1)
            return predicted.item()
        
def RunBot(input):
    path = '../data/neural.network.data.volumes/' + "sampleVolume/textTest.model" 
    data = torch.load(path)
    inputSize = data["input_size"]
    hiddenLayers = data["hidden_layers"]
    numClasses =  data["num_classes"]
    modelState = data["model_state"]
    type = data["type"]
    model = NeuralNetwork(inputSize, hiddenLayers, numClasses)
    bot = TextClassificationBot(model, vectorizer)
    predicted_class = bot.classify_text(input)
    print(f'Predicted class for "{input}": {predicted_class}')

RunBot("text classification with")