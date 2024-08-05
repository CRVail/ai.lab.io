## ai.observe.bot Quick Setup

### Ananconda Environment (Python)
[CONDA](https://www.anaconda.com/download)
```
conda create -n ai.observe.bot python=3.10.0
conda activate ai.observe.bot
```
### Install Requirements
```
pip install -r requirements.txt
```
### Start API
```
python api.py
#ai.observe.bot will run on port 5000
```
### Start CLI Session
```
python chat.py
#This is useful for testing.
```
### Training ai.observe.bot
```
python train.py
#This starts a training session for ai.observe.bot. 
```
Iterations are configured within the file. It is not recommended to modifiy this file unless you really know what you are doing.

### intents.json
This json file contains all the responses and commands for ai.observe.bot. This file can be changed and modified, however, any changes to this file will require training sessions. Changes to this file without training sessions will result in errors. 

### ai.observe.bot Web Client
To interact with ai.observe.bot (outside of a web service capacity), please refer to and install the ai.observe.bot Web Client repository.

### Production Deployments
ai.observe.bot requires very minimal process capacity and storage. The solution, however, does utilize sqlite for deep memory and neural network tokenization storage/reference so some advance storage planning should be considered.
```
OS : Linux/Ubuntu/Windows
FrameWorks: Python 3+/Anaconda (or conda lite)/sqlite
OperatingPort: 5000
```

## References & Acknowledgments

This project was developed by Chris Vail. 