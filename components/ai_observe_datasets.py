from sklearn.feature_extraction.text import CountVectorizer
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from .nltk_utils import bag_of_words, tokenize, stem
import torch.optim as optim
import sys
import json
import warnings
warnings.simplefilter("ignore", category=Warning)

#FORMERLY TextClassificationBot
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

def FrameworkDataset_Bot(dataPath):
    try:
        with open(dataPath, 'r') as f:
            intents = json.load(f)
        all_words = []
        tags = []
        xy = []
        for intent in intents['intents']:
            tag = intent['tag']
            tags.append(tag)
            for pattern in intent['patterns']:
                w = tokenize(pattern) # tokenize each word in the sentence
                all_words.extend(w) # add to our words list
                xy.append((w, tag)) # add to xy pair
        ignore_words = ['?', '.', '!'] # stem and lower each word
        all_words = [stem(w) for w in all_words if w not in ignore_words]
        all_words = sorted(set(all_words)) # remove duplicates and sort
        tags = sorted(set(tags))
        print(len(xy), "patterns")
        X_train = []
        y_train = []
        for (pattern_sentence, tag) in xy:
            bag = bag_of_words(pattern_sentence, all_words) # X: bag of words for each pattern_sentence
            X_train.append(bag) # y: PyTorch CrossEntropyLoss needs only class labels, not one-hot
            label = tags.index(tag)
            y_train.append(label)
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        res = {
            "tags": tags,
            "all_words": all_words,
            "X": X_train,
            "y": y_train
        }
        return res
    except Exception as error:
        print("An ERROR OCCURRED!!")
        print(error)
        return error

class ChatDataset(Dataset):
            def __init__(self, X_train, y_train):
                self.n_samples = len(X_train)
                self.x_data = X_train
                self.y_data = y_train
            # support indexing such that dataset[i] can be used to get i-th sample
            def __getitem__(self, index):
                return self.x_data[index], self.y_data[index]
            # we can call len(dataset) to return the size
            def __len__(self):
                return self.n_samples        
   
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