import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import os

def unpackLoadCondtions(loadConditions):
    forces = loadConditions['a']
    free = loadConditions['b']
    passive = loadConditions['c']
    formating_array = loadConditions['d']

    volfrac = formating_array[0]
    nelx = int(formating_array[1])
    nely = int(formating_array[2])
    penal = formating_array[3]
    rmin = formating_array[4]

    #print(volfrac,nelx,nely)

    return forces,free,passive,volfrac,nelx,nely,penal,rmin


def unpackIteration(iteration):
    x = iteration['a']
    xPhys = iteration['b']
    formating_array = iteration['c']

    compliance = formating_array[0]
    change = formating_array[1]
    mass = formating_array[2]

    #print(compliance,change,mass)
    return x,xPhys,compliance,change,mass

def getAgentData(agentPath):
    """
    Given a agent file path as a input, return all the unpacked and sorted data from that agent
    returns:
        forces,free,passive,volfrac,nelx,nely,penal,rmin,x_array,xPhys_array,compliance_array,change_array,mass_array
    """

    FilesToGrab = os.listdir(agentPath)
    numberOfIterations = len(FilesToGrab) - 1
    iterations = []

    loadConditionsExist = False


    for fileName in FilesToGrab:
        if('loadConditions' in fileName):
            loadConditions = np.load(os.path.join(agentPath,fileName))
            #print('loadCondtions Exist')
            loadConditionsExist = True
            
        elif('iteration_' in fileName):
            number_extension = fileName[len('iteration_'):]
            extesionIndex = number_extension.find('.')
            number = int(number_extension[:extesionIndex])
            #print(number)
            iterations.append([number,np.load(os.path.join(agentPath,fileName))])
        #print(fileName)
    
    if(not loadConditionsExist):
        raise Exception("File path {} does not hold propper data".format(agentPath))

    def sortKey(x):
        return x[0]

    iterations.sort(key=sortKey)
    #print(iterations)

    forces,free,passive,volfrac,nelx,nely,penal,rmin = unpackLoadCondtions(loadConditions)

    x_array = []
    xPhys_array = []
    compliance_array = []
    change_array = []
    mass_array = []

    for i in range(numberOfIterations):
        x,xPhys,compliance,change,mass = unpackIteration(iterations[i][1])
        x_array.append(x)
        xPhys_array.append(xPhys)
        compliance_array.append(compliance)
        change_array.append(change)
        mass_array.append(mass)

    return forces,free,passive,volfrac,nelx,nely,penal,rmin,x_array,xPhys_array,compliance_array,change_array,mass_array

def cleanData(path):
    """
    The data set was generatied via the compliance minmization TopOpt, this method was prone to some numerical instabilities thus some data points are marked as invalid.
    This function will remove them.
    """

    AgentsToGrab = os.listdir(path)
    print("There are {} files to explore.".format(len(AgentsToGrab)))

    invalidAgents = []

    for agent in AgentsToGrab:
        agentFiles = os.listdir(os.path.join(path,agent))
        for fileName in agentFiles:
            if('Invalid_' in fileName):
                invalidAgents.append(agent)
                break
        
    print("Of the {} files scanned {} were invalid.".format(len(AgentsToGrab),len(invalidAgents)))

    for agent in invalidAgents:
        agentFiles = os.listdir(os.path.join(path,agent))
        for fileName in agentFiles:
            fileToRemove = os.path.join(path,agent,fileName)
            print("removing: {}".format(fileToRemove))
            os.remove(os.path.join(path,agent,fileName))
        directoryToRemove = os.path.join(path,agent)
        print("removing: {}\n".format(directoryToRemove))
        os.rmdir(os.path.join(path,agent))

def formatIterativeModelDataSet(agentFilePath):
    """
    For the iterative model data.
    Takes the extensive output of the getAgentData function and returns the formated data needed for the iterateive model


    Forces, passive, and free must be rebuilt as propper images and not 1D arrays
    """
    forces,free,passive,_,nelx,nely,_,_,_,xPhys_array,_,_,_ = getAgentData(agentFilePath)
    # print("Forces shape",forces.shape)
    # print("free shape",free.shape)
    # print("passive shape",passive.shape)
    # print("xphys shape",xPhys_array[0].shape)


    finalShape = (int(nelx+1),int(nely+1))
    forces2 = forces.sum(1)
    forces2 = np.reshape(forces2,(finalShape[0],finalShape[1],2))

    d2 = np.ones(2*finalShape[0]*finalShape[1])
    for index in free:
        d2[index] = 0
    d3 = np.reshape(d2,(finalShape[0],finalShape[1],2))
    degreesOfFreedom2 = d3.sum(2)

    passive2 = np.zeros((nelx*nely))
    passive2 = np.where(passive > 0,1,0)
    passive2 = np.reshape(passive2,(nelx,nely))


    #reshape x_arrays as an np array
    numberOfIterations = len(xPhys_array)
    xPhys_np_array = np.zeros((nelx,nely,numberOfIterations))
    for i in range(numberOfIterations):
        xPhys_np_array[:,:,i] = np.reshape(xPhys_array[i],(nelx,nely))

    
    return forces2,degreesOfFreedom2,passive2,xPhys_np_array,numberOfIterations
         
        

def main():
    agentName = "Agent_344"
    agentFileToGet = os.path.join(os.getcwd(),"MachineLerning","Data","100_50",agentName)

    FilesToGrab = os.listdir(agentFileToGet)
    numberOfIterations = len(FilesToGrab) - 1
    iterations = []


    for fileName in FilesToGrab:
        if('loadConditions' in fileName):
            loadConditions = np.load(os.path.join(agentFileToGet,fileName))
            print('loadCondtions Exist')
            
        elif('iteration_' in fileName):
            number_extension = fileName[len('iteration_'):]
            extesionIndex = number_extension.find('.')
            number = int(number_extension[:extesionIndex])
            #print(number)
            iterations.append([number,np.load(os.path.join(agentFileToGet,fileName))])
        #print(fileName)
    
    def sortKey(x):
        return x[0]

    iterations.sort(key=sortKey)
    #print(iterations)

    forces,free,passive,volfrac,nelx,nely,penal,rmin = unpackLoadCondtions(loadConditions)

    x_array = []
    xPhys_array = []
    compliance_array = []
    change_array = []
    mass_array = []

    for i in range(numberOfIterations):
        x,xPhys,compliance,change,mass = unpackIteration(iterations[i][1])
        x_array.append(x)
        xPhys_array.append(xPhys)
        compliance_array.append(compliance)
        change_array.append(change)
        mass_array.append(mass)


    fig,ax = plt.subplots(2,2)
    im1 = ax[0,0].imshow(np.reshape(xPhys_array[0],(nelx,nely)).T, cmap='gray_r',norm=colors.Normalize(vmin=0,vmax=1))
    im2 = ax[0,1].imshow(np.reshape(xPhys_array[1],(nelx,nely)).T, cmap='gray_r',norm=colors.Normalize(vmin=0,vmax=1))
    
    im3 = ax[1,0].imshow(np.reshape(xPhys_array[-1],(nelx,nely)).T, cmap='gray_r',norm=colors.Normalize(vmin=0,vmax=1))
    im4 = ax[1,1].imshow(np.reshape(xPhys_array[-2],(nelx,nely)).T, cmap='gray_r',norm=colors.Normalize(vmin=0,vmax=1))

    plt.show()


    





if(__name__ == "__main__"):
    #AgentFolder = os.path.join(os.getcwd(),'MachineLerning','Data','100_50')
    #cleanData(AgentFolder)  

    main()



