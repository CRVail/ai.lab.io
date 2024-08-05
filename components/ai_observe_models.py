from sklearn.feature_extraction.text import CountVectorizer
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from components import ai_observe_utilities
from components import ai_observe_datasets
from .nltk_utils import bag_of_words, tokenize, stem
import sys
import json
import warnings
warnings.simplefilter("ignore", category=Warning)
vectorizer = CountVectorizer()
import random
     
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

def dataLoader(path, type, batchSize):
    if type == "ClassificationByArrayDataset_Long":
        dataset = ai_observe_datasets.ClassificationByArrayDataset_Long(path)
        dataloader = DataLoader(dataset, batch_size=batchSize, shuffle=True)
        return dataloader
    elif type == "TextClassification_Long":
        dataset = ai_observe_datasets.TextClassification_Long(path)
        return dataset
    elif type == "AIFramework_Bot":
        dataset = ai_observe_datasets.FrameworkDataset_Bot(path)
        #dataLoader = DataLoader(dataset=dataset, batch_size=batchSize, shuffle=True, num_workers=0)
        return dataset
    else:
        return "Classification Type Error!"
    
def train(volume, file, type, batchSize, inputSize, hiddenLayers, numClasses, learningRate, interations, IP):    
    path = 'data/neural.network.data.volumes/' + volume
    dataPath = path + '/input/' + file 
    savePath = path + '/output/' + file + '.model'
    finalLoss = 0
    #TRAINING SCRIPT: ClassificationByArrayDataset_Long
    if type == "ClassificationByArrayDataset_Long":
        dataloader = dataLoader(dataPath, type, batchSize)
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
    #TRAINING SCRIPT: TextClassification_Long    
    elif type == "TextClassification_Long":
        vectorizer = CountVectorizer()
        intentsData = dataLoader(dataPath, "TextClassification_Long", "")
        X = vectorizer.fit_transform(intentsData["inputs"]).toarray()
        y = torch.tensor(intentsData["labels"])
        num_epochs = interations
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
    #TRAINING SCRIPT: BotBuilder
    elif type == "AIFramework_Bot":
        # Prep Data
        dataloader = dataLoader(dataPath, type, batchSize)
        X_train = dataloader['X'] 
        y_train = dataloader['y']     
        tags = dataloader["tags"]
        all_words = dataloader["all_words"]
        # Hyper-parameters 
        input_size = len(X_train[0])
        output_size = len(tags)
        print(input_size, output_size)
        dset = ai_observe_datasets.ChatDataset(X_train, y_train)
        train_loader = DataLoader(dataset=dset, batch_size=batchSize, shuffle=True, num_workers=0)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = NeuralNetwork(input_size, hiddenLayers, output_size).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)
        for interation in range(interations):
            for (words, labels) in train_loader:
                words = words.to(device)
                labels = labels.to(dtype=torch.long).to(device)
                outputs = model(words) # Forward pass 
                loss = criterion(outputs, labels)
                optimizer.zero_grad() # Backward and optimize
                loss.backward()
                optimizer.step()                        
            if (interation+1) % 100 == 0:
                print (f'Iterations [{interation+1}/{interations}], Loss: {loss.item():.4f}')
        print(f'final loss: {loss.item():.4f}')
        finalLoss = loss.item()
        definitions = "None"
        modelData = {
            "model_state": model.state_dict(),
            "input_size": input_size,
            "hidden_size": hiddenLayers,
            "output_size": output_size,
            "all_words": all_words,
            "tags": tags,
            "type": type
        }
        torch.save(modelData, savePath)
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

def employModel(volumeName, type, modelData, values, IP):
    modelPath = 'data/neural.network.data.volumes/' + volumeName + "/output/" + modelData
    data = torch.load(modelPath)
    type = data["type"]
    if "definitions" in data:
        definitions = data["definitions"]
    if type == "ClassificationByArrayDataset_Long":
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
        msg = ai_observe_utilities.logEvent(volumeName, "Model Employed!", "A model was employed to make an evaluation: " + res, IP)
        return res #+ msg
    elif type == "TextClassification_Long":
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
                    "answer" : pred.item()
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
        msg = ai_observe_utilities.logEvent(volumeName, "Model Employed!", "A model was employed to make an evaluation: " + str(res), IP)
        return res
    elif type == "AIFramework_Bot":
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        modelMeta = ai_observe_utilities.getModel(volumeName, modelData)
        intentsFile = modelMeta[0][2]
        with open(intentsFile, 'r') as json_data:
            intents = json.load(json_data)
        inputSize = data["input_size"]
        outputSize = data["output_size"]
        all_words = data["all_words"]
        tags = data["tags"]
        hiddenLayers = data["hidden_size"]
        modelState = data["model_state"]
        model = NeuralNetwork(inputSize, hiddenLayers, outputSize).to(device)
        model.load_state_dict(modelState)
        model.eval()
        
        user_prompt = values
        prompt = user_prompt
        user_prompt = tokenize(user_prompt)
        X = bag_of_words(user_prompt, all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(device)            
        output = model(X)
        _, predicted = torch.max(output, dim=1)
        tag = tags[predicted.item()]
        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]
        if prob.item() > 0.75:
                for intent in intents['intents']:
                    if tag == intent["tag"]:
                        sources = []
                        if "command" in intent:
                            sources.append(intent["command"]["url"])
                        else:
                            cmd = "None"
                        res = random.choice(intent['responses'])                        
                        prompt_response_dict = {
                                "Prompt": prompt,
                                "Answer": res,
                                "Sources": sources,
                                "System": cmd
                            }
                        return prompt_response_dict
                    else:
                        msg1 = "Failed! " + type + "/" + modelData + " Failure at Neural Network processing step."
                        msg2 = "Model needs more training." 
                        msg3 = "Not enough patterns for target tag in intents.json file"
                        probCause = []
                        probCause.append(msg2)
                        probCause.append(msg3)
                        failMsg = {
                            "Message": msg1,
                            "ProbableCauses": probCause
                        }
                        return failMsg
    else:
        retMsg = """
        Invalid AI Classification 'type'. \n
        This service requires a classification type to determine how the Neural Netowrk will process your request.\n
        To make this request, please include the 'type' parameter. The expected values are 'ClassificationByArrayDataset_Long' \n
        or TextClassification_Long or frameworks you can also specify AIFramework_Bot to if your model was trained using the \n
        AIFramework_Bot framework classifier.        
        """
        return retMsg

    