from sklearn.feature_extraction.text import CountVectorizer
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import ai_observe_utilities
import ai_observe_datasets
from .nltk_utils import bag_of_words, tokenize, stem
import os
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



#with open('textClassTEST.json', 'r') as f:
#            data = json.load(f)
#T = vectorizer.fit_transform(d['input'] for d in data['data']).toarray()
#sd = T.shape[1]
#print(sd)
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


def testTrain(path):
    with open('../data/neural.network.data.volumes/sampleVolume/input/intents.json', 'r') as f:
        intents = json.load(f)
    all_words = []
    tags = []
    xy = []
    # loop through each sentence in our intents patterns
    for intent in intents['intents']:
        tag = intent['tag']
    # add to tag list
        tags.append(tag)
        for pattern in intent['patterns']:
        # tokenize each word in the sentence
            w = tokenize(pattern)
        # add to our words list
            all_words.extend(w)
        # add to xy pair
            xy.append((w, tag))

# stem and lower each word
    ignore_words = ['?', '.', '!']
    all_words = [stem(w) for w in all_words if w not in ignore_words]
# remove duplicates and sort
    all_words = sorted(set(all_words))
    tags = sorted(set(tags))

    print(len(xy), "patterns")
#print(len(tags), "tags:", tags)
#print(len(all_words), "unique stemmed words:", all_words)

# create training data
    X_train = []
    y_train = []
    for (pattern_sentence, tag) in xy:
    # X: bag of words for each pattern_sentence
        bag = bag_of_words(pattern_sentence, all_words)
        X_train.append(bag)
    # y: PyTorch CrossEntropyLoss needs only class labels, not one-hot
        label = tags.index(tag)
        y_train.append(label)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

# Hyper-parameters 
    num_epochs = 2200
    batch_size = 8
    learning_rate = 0.001
    input_size = len(X_train[0])
    hidden_size = 8
    output_size = len(tags)
    print(input_size, output_size)

    class ChatDataset(Dataset):

        def __init__(self):
            self.n_samples = len(X_train)
            self.x_data = X_train
            self.y_data = y_train

    # support indexing such that dataset[i] can be used to get i-th sample
        def __getitem__(self, index):
            return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
        def __len__(self):
            return self.n_samples

    dataset = ChatDataset()
    train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = NeuralNetwork(input_size, hidden_size, output_size).to(device)

# Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
    for epoch in range(num_epochs):
        for (words, labels) in train_loader:
            words = words.to(device)
            labels = labels.to(dtype=torch.long).to(device)
        
        # Forward pass
            outputs = model(words)
        # if y would be one-hot, we must apply
        # labels = torch.max(labels, 1)[1]
            loss = criterion(outputs, labels)
        
        # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if (epoch+1) % 100 == 0:
            print (f'Iterations [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


    print(f'final loss: {loss.item():.4f}')

    data = {
        "model_state": model.state_dict(),
        "input_size": input_size,
        "hidden_size": hidden_size,
        "output_size": output_size,
        "all_words": all_words,
        "tags": tags
    }

    FILE = "data.pth"
    torch.save(data, FILE)
    print(f'training complete. file saved to {FILE}')
    print("*music* And thats why this ai--- likes big butts and i cannot lie, cause brother, ai dont lie! 8) 8) 8) lol (8 (8 (8  *music* this ai runs on conda and i loves buns huns! *music* You can do side bend or sit ups, just please dont lose that butt. *music* baby got back!")
    
    
testPath = "C:\Repository\BIT_BUCKET\tasc.observe\..\data\neural.network.data.volumes\sampleVolume\intents.json"   
#testTrain(testPath)   
st = os.path.dirname(__file__) + '..\data\neural.network.data.volumes\sampleVolume\intents.json'   
#print(st)
#testTrain(st)    


path = '../../data/neural.network.data.volumes/sampleVolume/intents.json'
#if os.path.exists(testPath):
#    print("Path exists.")
#else:
#    print("Path does not exist.")#

#inpuaize = X.shape[1]
#print(inpuaize)

def train():
    # Define hyperparameters
      # Number of features
    hidden_size = 10
    num_classes = 3  # Number of classes (change according to your problem)
    num_epochs = 100
    input_size = X.shape[1]
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
    path = '../data/neural.network.data.volumes/' + "sampleVolume/textClassTEST.json.model" 
    paths = "textClassTEST.model"
    data = torch.load(path)
    inputSize = data["input_size"]
    hiddenLayers = data["hidden_layers"]
    numClasses =  data["num_classes"]
    modelState = data["model_state"]
    type = data["type"]
    model = NeuralNetwork(inputSize, hiddenLayers, numClasses)
    model.load_state_dict(modelState)
    model.eval()
    testData = str(["hi", "later", "really"])
    req_texts = eval(testData)
    new_X = vectorizer.transform(req_texts).toarray()
    new_X_tensor = torch.tensor(new_X, dtype=torch.float32)
    with torch.no_grad():
        outputs = model(new_X_tensor)
        _, predicted = torch.max(outputs, 1)
    for text, pred in zip(req_texts, predicted):
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

def testCommit():
    volume = ''
    statememt = {
        "model": "Andrew",
        "type": "Classifier"
    }
    ai_observe_utilities.volumeCommit(volume, statememt)
testCommit()
#RunBot("text classification with")
#botPrediction()
#testTrain("")