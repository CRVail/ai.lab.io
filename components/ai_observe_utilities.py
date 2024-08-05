import os
import sqlite3
import threading
from datetime import datetime, timedelta
operatingDIR = "data/neural.network.data.volumes/" 

def getVolConnString(volume):
    return operatingDIR + volume + '/' + volume + '.nnds'

def create_project_volume(newVolumeName):
    conn = None
    try:
        conn = sqlite3.connect(newVolumeName)
        print(sqlite3.sqlite_version)
    except sqlite3.Error as e:
        print(e)
    finally:
        if conn:
            conn.close()
            return "Success! New Neural Network Volume " + newVolumeName + " has been created!"
  
def describeVolumeTable(volumeName, table):
    try:
        connection_obj = sqlite3.connect(volumeName)
        a = connection_obj.execute("PRAGMA table_info('" + table + "')")
        res = []
        for i in a:
            print(i)
            res.append(i)
        return str(res)
    except:
        return "An error occured! Could not describe volume table. Does it exist?" 
            
def CreateTable(volumeName, tableScript):
    try:
        connection_obj = sqlite3.connect(volumeName)
        cursor_obj = connection_obj.cursor()
        table = tableScript
        cursor_obj.execute(table)
        connection_obj.close()
        print("table created!")
        return "A table was created for volume '" + volumeName + "' in NNDS (Neural Network Data Storage)"
    except:
        return "An error occured! Could not create table in Neural Network Volume '" + volumeName + "'!"
    
def dropTable(volumeName, table):
    connection_obj = sqlite3.connect(volumeName)
    cursor_obj = connection_obj.cursor()
    table = "DROP TABLE " + table
    cursor_obj.execute(table)
    connection_obj.close()
    print("table dropped!")
    return "Success! Table '" + table + "' was dropped from Neural Network Volume '" + volumeName + "'!"

def describeVolume(volumeName):
    connection_obj = sqlite3.connect(volumeName)
    cursor_obj = connection_obj.cursor()
    sqlTables ="""SELECT name FROM sqlite_master WHERE type='table';"""
    cursor_obj.execute(sqlTables)
    rows = cursor_obj.fetchall()
    return rows

def volumeCommit(volumeName, statement):
    try:
        connection_obj = sqlite3.connect(volumeName)
        sql = statement
        cursor_obj = connection_obj.cursor()
        cursor_obj.execute(sql)
        connection_obj.commit()
        return "Success! Statement was committed to Neural Network Volume '" + volumeName + "'!"
    except sqlite3.Error as error:
        return "An error occured! Failed to commit statement! \n\n" + str(error) + "\n\n" + statement

def volumeRead(volumeName, statement):
    connection_obj = sqlite3.connect(volumeName)
    cursor_obj = connection_obj.cursor()
    sqlTables = statement
    cursor_obj.execute(sqlTables)
    rows = cursor_obj.fetchall()
    return rows         

def logEvent(volumeName, eventName, description, IP):
    try:
        sql = "INSERT INTO VOLUME_LOG VALUES ('" + str(eventName,) + "','" + str(description) + "','" + str(IP) + "', '" + str(datetime.now()) + "')"
        a = volumeCommit(getVolConnString(volumeName), sql)
        return a
    except:
        return "Failed to log event to volume '" + volumeName + "'"

def logMetric(volumeName, datasetName, input, output, IP):
    try:
        sql = "INSERT INTO " + datasetName + " VALUES ('" + str(input) + "','" + str(output) + "','" + str(IP) + "')"
        a = volumeCommit(getVolConnString(volumeName), sql)
        return a
    except:
        return "Failed to log event to volume '" + volumeName + "'"

def logModel(volumeName, modelName, type, inputFile):
    try:
        sql = "INSERT INTO MODELS VALUES ('" + str(modelName) + "','" + str(type) + "','" + str(inputFile) + "', '" + str(datetime.now()) + "')"
        a = volumeCommit(getVolConnString(volumeName), sql)
        return a
    except:
        return " Failed to commit model meta data to volume. This will result in errors if you attempt to export the model."

def getModel(volumeName, modelName):
    statement = "SELECT * FROM MODELS WHERE MODEL_NAME='" + modelName + "'"
    res = volumeRead(getVolConnString(volumeName), statement)
    return res

def removeFile(volumeName, io, file):
    res =[]
    try:
        if io == "root":
            path = operatingDIR + volumeName + "/" + file
        else:
            path = operatingDIR + volumeName + "/" + io + "/" + file
        os.remove(path)
        res.append(str(file + " was successfully removed from volume " + volumeName))
    except:
        res.append(str("Attempted to delete " + file + " but an occured!"))
    return res

def initialize_ai_observe_data_utilities(IP):
    import os
    event = threading.Event()
    res = []
    #BUILD VOLUME DIRECTORY
    try:
        if os.path.isdir("data/neural.network.package.repo"):
            msg = "Attempted to build pathway to 'neural.network.package.repo' but it already existed."
            res.append(msg)
        else:
            msg = "'neural.network.package.repo' pathway was created."
            os.mkdir("data/neural.network.package.repo")
            res.append(msg)
        if os.path.isdir(operatingDIR):
            msg = "Attempted to build pathway to 'neural.network.data.volumes' but it already existed."
            res.append(msg)
        else:
            msg = "'neural.network.data.volumes' pathway was created."
            os.mkdir(operatingDIR)
            res.append(msg)
        event.set()
    except:
        msg = "Failed to build Neural Network Pathways! This is a critical failure that is likely due to permissions. Please make sure ai.observe has r/w access to its operating directory. Process aborted."
        event.set()
        return msg
    #BUILD VOLUME
    try:
        msg = create_project_volume('data/observe.data')
        res.append(msg)
        event.set()
    except:
        msg = "Failed to build observe.data volume! This is a critical failure. Please check the console for errors. Process aborted."
        event.set()
        return msg
    try:
        event.wait()
        msg = CreateTable('data/observe.data', "CREATE TABLE ACTIVITY_LOG (EVENT_NAME VARCHAR(255) NOT NULL, EVENT_VALUES VARCHAR(255) NOT NULL, IP VARCHAR(255) NOT NULL)")
        res.append(msg)
        event.set()
    except:
        msg = "Failed to CREATE TABLE ACTIVITY_LOG!"
        res.append(msg)
        event.set()
    try:
        event.wait()
        sql = "INSERT INTO ACTIVITY_LOG VALUES ('INIT','ai.observe was successfully initialized.','" + str(IP) + "')"
        msg = volumeCommit('data/observe.data', sql)
        res.append(msg)
        event.set()
    except:
        msg = "Failed to Commit to ACTIVITY_LOG!"
        res.append(msg)
        event.set()
    return res

def initialize_project_volume(volumeName, volumeDesc, IP):
    import os
    event = threading.Event()
    res = []
    #BUILD VOLUME DIRECTORY
    try:
        directory = operatingDIR + volumeName
        os.mkdir(directory)
        directory = operatingDIR + volumeName + "/input"
        os.mkdir(directory)
        directory = operatingDIR + volumeName + "/output"
        os.mkdir(directory)
        a = "A virtual 'Neural Network Data Store' was created for '" + volumeName + "'"
        res.append(a)
        event.set()
    except:
        a = "Failed to build Neural Network Volume!"
        event.set()
        return a
    #BUILD VOLUME
    try:
        a = create_project_volume(getVolConnString(volumeName))
        res.append(a)
        event.set()
    except:
        a = "Failed to build Neural Network Volume!"
        event.set()
        return a
    #ADD VOLUME DETAILS
    try:
        event.wait()
        a = CreateTable(getVolConnString(volumeName), "CREATE TABLE VOLUME_DETAILS (VOLUME_NAME VARCHAR(255) NOT NULL, VOLUME_DESCRIPTION VARCHAR(255) NOT NULL, IP VARCHAR(255) NOT NULL)")
        res.append(a)
        event.set()
    except:
        a = "Failed to CREATE TABLE VOLUME_DETAILS!"
        res.append(a)
        event.set()
    try:
        event.wait()
        a = CreateTable(getVolConnString(volumeName), "CREATE TABLE VOLUME_LOG (EVENT_NAME VARCHAR(255) NOT NULL, EVENT_DESCRIPTION VARCHAR(255) NOT NULL, IP VARCHAR(255) NOT NULL, TIMESTAMP VARCHAR(255) NOT NULL)")
        res.append(a)
        event.set()
    except:
        a = "Failed to CREATE TABLE VOLUME_LOG!"
        res.append(a)
        event.set()
    try:
        event.wait()
        sql = "INSERT INTO VOLUME_DETAILS VALUES ('" + str(volumeName) + "','" + str(volumeDesc) + "','" + str(IP) + "')"
        a = volumeCommit(getVolConnString(volumeName), sql)
        res.append(a)
        event.set()
    except:
        a = "Failed to Commit to VOLUME_DETAILS!"
        res.append(a)
        event.set()
    try:
        event.wait()
        a = CreateTable(getVolConnString(volumeName), "CREATE TABLE MODELS (MODEL_NAME VARCHAR(255) NOT NULL, MODEL_TYPE VARCHAR(255) NOT NULL, MODEL_INPUT_FILE VARCHAR(255) NOT NULL, TIMESTAMP VARCHAR(255) NOT NULL)")
        res.append(a)
        event.set()
    except:
        a = "Failed to CREATE TABLE VOLUME_LOG!"
        res.append(a)
        event.set()
    #
    
    return res

def initialize_project_dataset(volumeName, datasetName, IP):
    try:
        res = []
        sql = "CREATE TABLE " + datasetName + " (input VARCHAR(255) NOT NULL, label VARCHAR(255) NOT NULL, IP VARCHAR(255) NOT NULL)"
        sysMsg = CreateTable(getVolConnString(volumeName), sql)
        res.append(sysMsg)
        msg1 = "You can start collecting metrics for this dataset by posting data to this url: \r\n" 
        res.append(msg1)
        msg2 = "https://<ai.observe.url>/api/logMetric?volumeName=" + volumeName + "&dataSet=" + datasetName + "\r\n"
        res.append(msg2)
        sysMsg2 = logEvent(volumeName,"New Dataset Initialized", "A new dataset, " + datasetName + ", was initialized for metrics collections.", IP)
        res.append(sysMsg2)
        rstMsg = "Project Dataset Initialized! You can now start collecting metrics for Neural Network Processing!"
        res.append(rstMsg)
        return res
    except:
        return "Failed to initialize new project table in project volume!"