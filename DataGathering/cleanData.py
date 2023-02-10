"""
This small library is dedicated to cleaning out bad data from the dataset.
Given a folder of agents to work on this library will remove any agents with 0,Nan, or inf compliance and will attempt to remove any duplicate members as well
"""

import numpy as np
import os

def getAgent(path):
    """
    Given a path to a specific agent csv file will return the numpy arrays representing the data:

        - xPhys: a (nelx,nely) array of ones or zeros represneting the physical structure of the part
        - forces: a [(nelx+1)*(nely+1)*2,4] array of the forces to be applied on the agent
        - degreesOfFreedom: a (nelx+1)*(nely+1)*2 array of indexes that can be moved by the forces
        - shape: an array holding the shape of the part
        - compliance_max: the compliance that the agent was targeting
        - compliance: the compliance of the current agent
    """

    filesToPull = os.listdir(path)

    dataPoints = []

    for fileName in filesToPull:
        if(fileName.index('.csv.npz') < 0):
            print("Invalid File: {} is not a numpy array.".format(fileName))
        else:
            try:

                data = np.load(os.path.join(path,fileName),allow_pickle=True)

                xPhys = data['a']
                forces = data['b']
                degreesOfFreedom = data['c']
                formating_array = data['d']

                maxDof = forces.shape[0]
                toPad = maxDof - len(degreesOfFreedom)
                degreesOfFreedom = np.pad(degreesOfFreedom,[0,toPad])

                compliance = formating_array[0]
                compliance_max = formating_array[1]
                shape = formating_array[2:]

                #forces = np.reshape(forces,(shape[0],shape[1],4))

                dataPoints.append([xPhys,forces,compliance])

            except:
                print("Error in reading file couldn't get file: {}".format(fileName))
    
    return dataPoints

def getAgents(path):
    """
    Given a path to the agents folder (should look like "30_30" or "<nelx>_<nely>" of the agents) returns a specified number of agents under the max total agents in the file.

    Given the path:  
    'os.path.join(os.getcwd(),"Data\\30_30")'

    You should be able to extract all agents
    """

    dir_list = os.listdir(path)
    max_data_points = len(dir_list)
    numberToGrab = max_data_points
    print("Number of data points: {}".format(len(dir_list)))

    

    data = []
    for i in range(numberToGrab):
        agent = getAgent(os.path.join(path,dir_list[i]))
        for xPhys,forces,compliance in agent:
            data.append([xPhys,forces,compliance])
    

    return data

def cleanData(path):
