from sklearn.feature_extraction.text import CountVectorizer
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import ai_observe_utilities

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