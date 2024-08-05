import os,shutil
import time
import torch
from components import ai_observe_utilities
import zipfile

def export_model(volumeName, model):
    repoPath = 'data/neural.network.package.repo/'
    volumePath = 'data/neural.network.data.volumes/' + volumeName
    modelPath = volumePath + "/output/" + model
    data = torch.load(modelPath)
    type = data["type"]
    modelMeta = ai_observe_utilities.getModel(volumeName, model)
    intentsFile = modelMeta[0][2]
    if type == "ClassificationByArrayDataset_Long":
        return "This export option has not been developed yet."
    elif type == "TextClassification_Long":
        return "This export option has not been developed yet."
    elif type == "AIFramework_Bot":
        source = repoPath + "ai.framework.bot.exportable.zip"
        file = "ai.framework.bot.exportable.zip"
        shutil.copy(os.path.join(repoPath, file), volumePath + "/output/")
        with zipfile.ZipFile(volumePath + "/output/" + file, "a", compression=zipfile.ZIP_DEFLATED) as zipf:
            #Add Model Data via SQLite to exportable zip 
            destination_modelData = 'ai.framework.bot.exportable/data.pth'
            zipf.write(modelPath, destination_modelData)
            #Add intents.json file 
            destination_Intents = 'ai.framework.bot.exportable/intents.json'
            zipf.write(intentsFile, destination_Intents)
            res = {
                "downloadName": file,
                "downloadPath": volumePath + "/output/" + file
            }
        return res
    else:
        msgTitle = "Invalid AI Classification 'type'"
        msgDetails = """
        This service requires a classification type to determine how the Neural Netowrk will process your request.\n
        To make this request, please include the 'type' parameter. The expected values are 'ClassificationByArrayDataset_Long' \n
        or TextClassification_Long or frameworks you can also specify AIFramework_Bot to if your model was trained using the \n
        AIFramework_Bot framework classifier.        
        """
        retMsg = []
        retMsg.append(msgTitle)
        retMsg.append(msgDetails)
        res = {
            "Messages": retMsg
        }    
        return res

def export_volume(volumeName):
    return ""

def export_NNDS(volumeName):
    return ""