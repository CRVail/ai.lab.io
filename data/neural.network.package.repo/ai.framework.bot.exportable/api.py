#####################################################################
# tasc.bot Requirements
#####################################################################
from flask import Flask, jsonify, request
import random
import json
import torch
import sqlite3
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
#####################################################################
# tasc.bot Model & NN CONFIGURATIONS
#####################################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)
FILE = "data.pth"
data = torch.load(FILE)
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()
bot_name = "ai.observe.bot"

app = Flask(__name__) 
#####################################################################
# tasc.bot COMMANDS AND FUNCTIONS
#####################################################################
#Performs GET request for tasc.bot
def runCMD(url):
    import requests
    response = requests.request("GET", url)
    return response.json() 

#Commits an interaction to the botMem.db (SQLite) 
def requestLogCommit(IP, Prompt, Answer, Sources, System):
    connection_obj = sqlite3.connect('botMem.db')
    strIP = str(IP).replace("'","")
    strPrompt = str(Prompt).replace("'","")
    strAnswer = str(Answer).replace("'","")
    strSources = str(Sources).replace("'","")
    strSystem = str(System).replace("'","")
    sql = "INSERT INTO REQUEST_LOG VALUES ('" + strIP + "','" + strPrompt + "','" + strAnswer  + "','" + strSources  + "','" + strSystem + "')"
    cursor_obj = connection_obj.cursor()
    cursor_obj.execute(sql)
    connection_obj.commit()
    return "Success!"

#Queries Log Commits by IP
def requestLogReference(IP):
    connection_obj = sqlite3.connect('botMem.db')
    sql = "SELECT * FROM REQUEST_LOG WHERE IP ='" + IP + "'"
    cursor_obj = connection_obj.cursor()
    cursor_obj.execute(sql)
    rows = cursor_obj.fetchall()
    return rows

#Commits a Long Term Memory 
# Such as a name or location) to botMem.db (SQLite)
def commitLongTermMemory(IP, Name, Ref):
    connection_obj = sqlite3.connect('botMem.db')
    strIP = str(IP).replace("'","")
    strName = str(Name).replace("'","")
    strRef = str(Ref).replace("'","")
    sql = "INSERT INTO LONG_TERM_MEMORY VALUES ('" + strIP + "','" + strName + "','" + strRef + "')"
    cursor_obj = connection_obj.cursor()
    cursor_obj.execute(sql)
    connection_obj.commit()
    return "Success!"

#Queries Long Term Memories by IP 
# Allows tasc.bot to recall names, paths, etc)
def requestLongTermMemory(IP):
    connection_obj = sqlite3.connect('botMem.db')
    sql = "SELECT * FROM LONG_TERM_MEMORY WHERE IP ='" + IP + "'"
    cursor_obj = connection_obj.cursor()
    cursor_obj.execute(sql)
    rows = cursor_obj.fetchall()
    return rows

#Commits a Short Term Memory 
# Such as a clients position in a form) to botMem.db (SQLite)
def commitShortTermMemory(IP, Step, Ref):
    connection_obj = sqlite3.connect('botMem.db')
    strIP = str(IP).replace("'","")
    strStep = str(Step).replace("'","")
    strRef = str(Ref).replace("'","")
    sql = "INSERT INTO SHORT_TERM_MEMORY VALUES ('" + strIP + "','" + strStep + "','" + strRef + "')"
    cursor_obj = connection_obj.cursor()
    cursor_obj.execute(sql)
    connection_obj.commit()
    return "Success!"

#Queries Short Term Memories by IP
# Allows tasc.bot to step users through forms and remember 
# specific information for that session.
def requestShortTermMemory(IP):
    connection_obj = sqlite3.connect('botMem.db')
    sql = "SELECT * FROM SHORT_TERM_MEMORY WHERE IP ='" + IP + "'"
    cursor_obj = connection_obj.cursor()
    cursor_obj.execute(sql)
    rows = cursor_obj.fetchall()
    return rows

#Memory Recollection
# Allows tasc.bot to recall memories during client interactions
# This is useful for personalization and step based data gathering.
def memoryReference(IP):
    LongTerm = requestLongTermMemory(IP)
    ShortTerm = requestShortTermMemory(IP)
    Name = ""
    Step = ""
    if LongTerm:
        Name = LongTerm["NAME"]
    else:
        Name = ""
    if ShortTerm:
        Step = ShortTerm["STEP"]
    else: 
        Step = ""
    obj = {
        "Name": Name,
        "Step": Step
    }
    return obj
    
#####################################################################
# tasc.bot Services
#####################################################################
#Chat with tasc.bot
@app.route('/api/prompt_route', methods = ['GET', 'POST']) 
def prompt_route(): 
    if(request.method == 'GET'): 
        user_prompt = request.form.get("user_prompt")
        user_address = request.remote_addr
        if user_prompt:
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
                            cmd = runCMD(intent["command"]["url"])
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
                        requestLogCommit(user_address, prompt, res, sources, cmd)
                        return jsonify(prompt_response_dict), 200
            else:
                res = "Sorry, I do not understand... tasc.bots are trained and specialized for specific directives, subjects or tasks. You can ask me 'What is your directive?' or 'What do you do?' to learn my directives or simply type 'help'."
                sources = []
                cmd = "None"
                prompt_response_dict = {
                                "Prompt": prompt,
                                "Answer": res,
                                "Sources": [],
                                "System": {}
                            }
                requestLogCommit(request.remote_addr, prompt, res, "", cmd)
                return jsonify(prompt_response_dict), 200
    else:
        return "No user prompt received", 200

#Returns a list of commands tasc.bot can perfom
@app.route('/api/command_route', methods = ['GET', 'POST'])
def command_route():
    import requests
    if(request.method == 'GET'):
        cmd_request = request.args.get('cmd')
        for intent in intents['intents']:
            if "command" in intent:
                if cmd_request in intent["tag"]:
                    url = intent["command"]["url"]
                    response = requests.request("GET", url)
                    res_dict = {
                        "CommandName": cmd_request,
                        "System": response.json()
                    }
                    return jsonify(res_dict), 200
                else: 
                    return "Command not found!", 200

#Returns a list of responses tasc.bot can return
@app.route('/api/response_route', methods = ['GET']) 
def response_route(): 
    if(request.method == 'GET'):
        response_dict = {
                         "BotName": bot_name,
                         "Message": "Here is a copy of my intents file. I use this file for training. It helps me understand user prompts as well as what commands and functions I serve or that I am responsible for. If you are looking for my short term memory log, please use route '/api/shortTermMemory'",
                         "System": intents['intents']
                        }
    
        return jsonify(response_dict), 200
    else:
        return "Bad Request", 500

#Returns a log of interactions with tasc.bot by IP
@app.route('/api/requestLogMemory', methods = ['GET'])
def shortTermMemory_route():
    if(request.method == 'GET'):
        res = requestLogReference(request.remote_addr)        
        res_dict = {
                        "CommandName": "requestLogMemory",
                        "System": res
                    }
        return jsonify(res_dict), 200
    else:
        return "Bad Request", 500
        
if __name__ == '__main__': 
  
    app.run(debug = True) 