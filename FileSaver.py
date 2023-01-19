import os
import numpy as np

class AgentFileSaver:
    """
    Class dedicated to creating all the datapoints needed to train an ML model
    """
    def __init__(self,agentNumber:int,nelx:int,nely:int):
        self.number = agentNumber
        self.nelx = nelx
        self.nely = nely
        self.agentDirectory = os.path.join(os.getcwd(),"Agents")
        self.agentFolderPath = ""
        self.arraysToSave = []

        os.makedirs(self.agentDirectory,exist_ok=True)

    def setNumber(self,agentNumber):
        self.number = agentNumber

    def createAgentFile(self):
        """
        Creates a new directory to store the current agent in
        """
        #check to see if dimension folder exists
        dimesionFolder = os.path.join(self.agentDirectory,"{}_{}".format(self.nelx,self.nely))
        pathExists = os.path.exists(dimesionFolder)
        if( not pathExists):
            os.makedirs(dimesionFolder)
        
        #check to see if the current agent number already exists
        agentFolder = os.path.join(dimesionFolder,"Agent_{}".format(self.number))
        pathExists = os.path.exists(agentFolder)
        if(not pathExists):
            os.makedirs(agentFolder)
        else:
            # if the agent folder currently exist then create an Agent#_ folder
            foundOpenNumber = False
            currentNumber = 1
            while( not foundOpenNumber):
                currentAgentN = os.path.join(dimesionFolder,"Agent{}_{}".format(currentNumber,self.number))
                pathExists = os.path.exists(currentAgentN)
                if(pathExists):
                    currentNumber += 1
                else:
                    foundOpenNumber = True
            os.makedirs(currentAgentN)
            agentFolder = currentAgentN

        self.agentFolderPath = agentFolder

    def loadFileToSave(self,name:str,array:np.ndarray):
        """
        Puts a file to save into a buffer that can be saved at a later time

        Requres external function to actually save the data stored
        """
        self.arraysToSave.append([name,array])

    def saveStoredFiles(self):
        """
        Takes the files currently stored in the save array and saves them to a file
        """
        originalWorkingDirectory = os.getcwd()
        if(len(self.agentFolderPath) == 0):
            self.createAgentFile()

        os.chdir(self.agentFolderPath)

        for name,array in self.arraysToSave:
            fileNameToSaveAs = name + ".csv"
            try:
                np.savetxt(fileNameToSaveAs,array,delimiter=',',header=str(array.shape))
            except(ValueError):
                print("Something went wrong.")
                print("Tried to save: {}".format(fileNameToSaveAs))
                print("Array is shape: {}".format(array.shape))


        os.chdir(originalWorkingDirectory)

    def saveCompressedFiles(self, **args):
        """
        Takes the files currently stored in the save array and saves them to a file
        """
        originalWorkingDirectory = os.getcwd()
        if(len(self.agentFolderPath) == 0):
            self.createAgentFile()

        os.chdir(self.agentFolderPath)

        
        fileNameToSaveAs = "Agent{}".format(self.number) + ".csv"
        try:
            np.savez_compressed(fileNameToSaveAs,args)
        except:
            print("Something went wrong.")
            print("Tried to save: {}".format(fileNameToSaveAs))


        os.chdir(originalWorkingDirectory)


    def saveNumpyArray(self,name:str,array:np.ndarray):
        """
        Takes a file name as well as a numpy array and stores it inside the file for the agent.

        Name should only be the descriptor of the file and should not include the extension or path of the file

        saves the file as a csv with a header holding the shape of the array, just in case.
        """

        fileNameToSaveAs = name + ".csv"
        originalWorkingDirectory = os.getcwd()
        if(len(self.agentFolderPath) <= 2):
            self.createAgentFile()

        os.chdir(self.agentFolderPath)
        np.savetxt(fileNameToSaveAs,array,delimiter=',',header=str(array.shape))
        os.chdir(originalWorkingDirectory)