# A 165 LINE TOPOLOGY OPTIMIZATION CODE BY NIELS AAGE AND VILLADS EGEDE JOHANSEN, JANUARY 2013
import numpy as np

from matplotlib import colors
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from topopt import topOpter
from ProblemMapper import*

def randomCircleGenerator():
    x1 = np.random.random()
    y1 = np.random.random()
    r1 = np.random.random()/3
    if(r1 <= 0.05):
        r1 = 0.1
    f1 = np.random.random()*3
    a1 = np.random.random()*2*np.pi
    c1 = [x1,y1,r1,f1,a1]
    return c1

def forceEquilizer(circle_1,circle_2):
    #[x1,y1,r1,f1,a1]
    fx = circle_1[3]*np.cos(circle_1[4]) + circle_2[3]*np.cos(circle_2[4])
    fy = circle_1[3]*np.sin(circle_1[4]) + circle_2[3]*np.sin(circle_2[4])

    counterForce_x = -fx
    counterForce_y = -fy

    magnitude = np.linalg.norm([counterForce_x,counterForce_y],ord=2)
    angle = np.arctan2(counterForce_y,counterForce_x)

    x1 = np.random.random()
    y1 = np.random.random()
    r1 = np.random.random()/3
    if(r1 <= 0.05):
        r1 = 0.1

    c1 = [x1,y1,r1,magnitude,angle]
    return c1



def generateRandomProblemStatement(nelx,nely):
    setupTries = 100
    canSetUp = False
    for i in range(setupTries):
        try:                                                                            
            circle_1 = randomCircleGenerator()
            circle_2 = randomCircleGenerator()
            circle_3 = forceEquilizer(circle_1,circle_2)
            filledArea,supportArea,forceVector,minViableArea = mapProblemStatement2D(nelx,nely,circle_2,circle_1,circle_3,"y")
            #filledArea2,supportArea2,forceVector2,minViableArea = mapProblemStatement2D(nelx,nely,circle_2,circle_1,circle_3,"y",correctError=False)
        except Exception as e:
            #print(e)
            canSetUp = False
        else:
            canSetUp = True
            return filledArea,supportArea,forceVector#,filledArea2,supportArea2,forceVector2
    
    if(canSetUp == False):
        # The variables are in order: x position of cylinder, y position of cylinder, radius of the cylinder, the magnitude of the force,
        # and the counterclockwise angle of the force in degrees.

        circle_1 = [.15,.15,.1,1,(3/2)*np.pi]
        circle_2 = [.5,.5,.2,1,(1/2)*np.pi]
        circle_3 = [.85,.85,.1,1,(3/2)*np.pi]
        filledArea,supportArea,forceVector,minViableArea = mapProblemStatement2D(nelx,nely,circle_2,circle_1,circle_3,"y")
        return filledArea,supportArea,forceVector


def updateImageDropOff(imageArray,val):
    #remap image from [-1,0] to some other range
    rec = val/(val+1)
    #v = rec*np.ones(imageArray.shape)
    above0 = np.where(imageArray > val,0,(imageArray/(val+1)) - rec)

    return above0

def runTopOpt():
    # Default input parameters
    nelx=100
    nely=50
    volfrac=0.4
    rmin=5.4
    penal=3.0
    ft=0 # ft==0 -> sens, ft==1 -> dens

    # import sys
    # if len(sys.argv)>1: nelx    =int(sys.argv[1])
    # if len(sys.argv)>2: nely    =int(sys.argv[2])
    # if len(sys.argv)>3: volfrac =float(sys.argv[3])
    # if len(sys.argv)>4: rmin    =float(sys.argv[4])
    # if len(sys.argv)>5: penal   =float(sys.argv[5])
    # if len(sys.argv)>6: ft      =int(sys.argv[6])
    #print(loads)



    t = topOpter(nelx,nely,volfrac,penal,rmin,ft,saveFile=False)

    filledArea,supportArea,forceVector = generateRandomProblemStatement(nelx,nely)
    t.ApplyProblem(filledArea,supportArea,forceVector)
    t.saveLoadConditions()
    

    plt.ion() # Ensure that redrawing is possible
    fig,ax = plt.subplots(2,1)
    im1 = ax[0].imshow(t.getPart().T, cmap='gray', interpolation='none',norm=colors.Normalize(vmin=-1,vmax=0))
    fig.show()
    sliderAxis = fig.add_axes([0.1, 0.25, 0.0225, 0.63])
    slider_for_material = Slider(
        ax=sliderAxis,
        label="p",
        valmin=-.999,
        valmax=0,
        valinit=0,
        orientation="vertical"
    )
    prevSliderVal = slider_for_material.val
    im2 = ax[1].imshow(t.getDerivetiveOfSensitivity().T, cmap='plasma_r', interpolation='none',norm=colors.Normalize(vmin=-1,vmax=0))
    done = True
    while(plt.fignum_exists(fig.number)):
        im1.set_array(updateImageDropOff(-t.getPart().T,slider_for_material.val))
        im2.set_array(t.getDerivetiveOfSensitivity().T)
        fig.canvas.draw()
        fig.canvas.flush_events()
        if(done):
            #print("save itt")
            t.saveIteration()
        done = t.itterate()

def testProblemMap():
    nelx = 160 
    nely = 80
    filledArea,supportArea,forceVector = generateRandomProblemStatement(nelx,nely)

    print(filledArea.shape)
    print(supportArea.shape)
    print(forceVector.shape)

    #format the forces
    fTotal = forceVector.sum(1)


    fx = fTotal[0:len(fTotal):2]
    fy = fTotal[1:len(fTotal):2]

    fx = np.reshape(fx,(nelx+1,nely+1))
    fy = np.reshape(fy,(nelx+1,nely+1))

    print(fx.shape)
    print(fy.shape)

    if(True):
        fig,ax = plt.subplots(2,2)

        im1 = ax[0,0].imshow(filledArea.T, cmap='gray_r')
        im2 = ax[0,1].imshow(supportArea.T, cmap='gray_r')


        im3 = ax[1,0].imshow(fx.T, cmap='plasma')
        im4 = ax[1,1].imshow(fy.T, cmap='plasma')
        plt.show()

def runTopOpt_SaveData():
    # Default input parameters
    nelx=100
    nely=50
    volfrac= np.random.random()*.6 + 0.3 #value between .3 and .9
    rmin=5.4
    penal=3.0
    ft=0 # ft==0 -> sens, ft==1 -> dens

    # import sys
    # if len(sys.argv)>1: nelx    =int(sys.argv[1])
    # if len(sys.argv)>2: nely    =int(sys.argv[2])
    # if len(sys.argv)>3: volfrac =float(sys.argv[3])
    # if len(sys.argv)>4: rmin    =float(sys.argv[4])
    # if len(sys.argv)>5: penal   =float(sys.argv[5])
    # if len(sys.argv)>6: ft      =int(sys.argv[6])
    #print(loads)

    

    t = topOpter(nelx,nely,volfrac,penal,rmin,ft,saveFile=True)

    filledArea,supportArea,forceVector = generateRandomProblemStatement(nelx,nely)
    t.ApplyProblem(filledArea,supportArea,forceVector)
    t.saveLoadConditions()
    
 
    done = True
    while(done):
        t.saveIteration()
        done = t.itterate()


# The real main driver    
if __name__ == "__main__":
    for i in range(10000):
        print("\tCurrent Problem = {}".format(i))
        runTopOpt_SaveData()
    #testProblemMap()


