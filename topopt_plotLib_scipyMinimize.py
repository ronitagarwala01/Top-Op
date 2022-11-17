# A 165 LINE TOPOLOGY OPTIMIZATION CODE BY NIELS AAGE AND VILLADS EGEDE JOHANSEN, JANUARY 2013
import numpy as np

from matplotlib import colors
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from topopt_scipyMinimize import topOpter

from time import time
import sys


def updateImageDropOff(imageArray,val):
    #remap image from [-1,0] to some other range
    rec = val/(val+1)
    #v = rec*np.ones(imageArray.shape)
    above0 = np.where(imageArray > val,0,(imageArray/(val+1)) - rec)

    return above0


def cantileverSetup(topopter :topOpter):
    """
    Setup the optimization enviroment for the cantilever beam.
    This is a bar that is anchored on the left side and has a force applied to it on the far left side

    Because this function changes variables within an object there is no need to return anything
    """
    nelx = topopter.nelx
    nely = topopter.nely
    anchorArray = np.zeros((nelx,nely))
    #fully anchor the far left side
    for y in range(nely):
        anchorArray[0,y] = 3
    topopter.updateFixed(anchorArray)

    # apply a force to the far richt side at the center
    topopter.updateForceVectors([[nelx-1,nely//2,0,1]])


# The real main driver    
if __name__ == "__main__":
    # Default input parameters
    nelx=20
    nely=10
    volfrac=0.4
    rmin=5.4
    penal=3.0
    ft=0 # ft==0 -> sens, ft==1 -> dens
    # The variables are in order: x position of cylinder, y position of cylinder, radius of the cylinder, the magnitude of the force,
    # and the counterclockwise angle of the force in degrees.
    circle_1 = [10,20,5,4,63]
    circle_2 = [15,20,7,5,115]
    circle_3 = [20,30,3,6,275]

    loads = [circle_1,circle_2,circle_3]

    if len(sys.argv)>1: nelx   =int(sys.argv[1])
    if len(sys.argv)>2: nely   =int(sys.argv[2])
    if len(sys.argv)>3: volfrac=float(sys.argv[3])
    if len(sys.argv)>4: rmin   =float(sys.argv[4])
    if len(sys.argv)>5: penal  =float(sys.argv[5])
    if len(sys.argv)>6: ft     =int(sys.argv[6])
    #print(loads)



    t = topOpter(nelx,nely,volfrac,penal,rmin,ft)
    cantileverSetup(t)
    """
    Durring testing the size of the part can change larger number of elements take increasinly more compuational power to optimize

    complianceMax is the maximum allowable movement allowed within the part. (10 is just arbitrary number that looks good)

    maxElementChange is the allowable movement of each element from its previous value to its new optimized value
     - is an interval between 0 and 1 
        - it can be higher than 1 but it is clamped so that will have no impact on it
        - If it is less than zero, you will break the system.
     - decreasing max element change (values closer to but not equal to 0) increases the number of required itterations to part comletion(increasing time) but also increases acuracy
     - increasing max element change (values closer to 1) decrease the number of required itterations but increases inacuracy. 
        -The optimizer will also run a bit longer per itteration but is still faster than a low change ammount

    minChange is the minimum change required before the optimizer is considered 'done'
     - the optimizer calculates change as the max change of any single element in the part. 
        Therefore if the max change of an element is lower than minchange the optimizer will think it is done. 
        This may lead to an unfinished part especially if maxElementChange is low
    """
    t.complianceMax = 10
    t.maxElementChange = 0.1
    t.minChange = 0.005
    
    # building the matplotlib plot
    itterateionArray = [] #stores the xvalues of the mass over time plot
    massOverItterationsArray=[] # stores the y-values of the mass over time plot
    plt.ion() # Ensure that redrawing is possible
    fig,ax = plt.subplots(2,1)
    im1 = ax[0].imshow(t.getPart().T, cmap='gray', interpolation='none',norm=colors.Normalize(vmin=-1,vmax=0))#setup the part model
    im2 = ax[1].scatter(itterateionArray,massOverItterationsArray,color='blue')#setup the mass over time graph
    #label the mass over time graph
    ax[1].set_ylabel("Mass")
    ax[1].set_xlabel("Itterations")
    fig.show()


    #begin actual optimzation loop
    optimizing = True
    timeStart = time()
    timerOn = True
    while(plt.fignum_exists(fig.number)): #while the user has the figure open

        #Clear the canvas
        fig.canvas.flush_events()
        
        #optimize the part
        optimizing = t.itterate()

        #update the outputs
        if(optimizing):
            #update the mass over time graph
            itterateionArray.append(t.loop)
            massOverItterationsArray.append(t.getMassWithPenalty(0))

            #update the graph with the current part model
            im1.set_array(-t.getPart().T)
            im2 = ax[1].scatter(itterateionArray,massOverItterationsArray,color='blue')

        elif(timerOn): #if finished optimizing then read out the time it took to complete the part
            timerOn = False
            timeEnd = time()
            print("Time elapsed: {:.2f}".format(timeEnd-timeStart))

        #redraw the canvas
        fig.canvas.draw()





