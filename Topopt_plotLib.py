# A 165 LINE TOPOLOGY OPTIMIZATION CODE BY NIELS AAGE AND VILLADS EGEDE JOHANSEN, JANUARY 2013
import numpy as np

from matplotlib import colors
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from topopt import topOpter
from ProblemMapper import*



def updateImageDropOff(imageArray,val):
    #remap image from [-1,0] to some other range
    rec = val/(val+1)
    #v = rec*np.ones(imageArray.shape)
    above0 = np.where(imageArray > val,0,(imageArray/(val+1)) - rec)

    return above0




# The real main driver    
if __name__ == "__main__":
    # Default input parameters
    nelx=5
    nely=5
    volfrac=0.4
    rmin=5.4
    penal=3.0
    ft=0 # ft==0 -> sens, ft==1 -> dens
    # The variables are in order: x position of cylinder, y position of cylinder, radius of the cylinder, the magnitude of the force,
    # and the counterclockwise angle of the force in degrees.
    circle_1 = [.2,.3,.1,1,(3/2)*np.pi]
    circle_2 = [.5,.5,.2,1,(1/2)*np.pi]
    circle_3 = [.8,.7,.1,1,(3/2)*np.pi]

    filledArea,supportArea,forceVector = mapProblemStatement2D(nelx,nely,circle_2,circle_1,circle_3,"y")

    loads = [circle_1,circle_2,circle_3]
    import sys
    if len(sys.argv)>1: nelx    =int(sys.argv[1])
    if len(sys.argv)>2: nely    =int(sys.argv[2])
    if len(sys.argv)>3: volfrac =float(sys.argv[3])
    if len(sys.argv)>4: rmin    =float(sys.argv[4])
    if len(sys.argv)>5: penal   =float(sys.argv[5])
    if len(sys.argv)>6: ft      =int(sys.argv[6])
    #print(loads)



    t = topOpter(nelx,nely,volfrac,penal,rmin,ft)
    #t.updateLoads(loads)
    anchorArray = np.zeros((nelx,nely))
    anchorArray[1,2] = 3
    #t.updateFixed(anchorArray)
    #t.updateForceVectors([[150,10,0,1]])
    t.passive = (2*filledArea).reshape(nelx*nely)
    t.updateFixed(supportArea*3)
    t.f = forceVector
    t.numberOfForces = 4
    t.u=np.zeros((2*(nelx+1)*(nely+1),4))
    
    
    


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
    done = False
    agentNumber = 0
    while(plt.fignum_exists(fig.number)):
        im1.set_array(updateImageDropOff(-t.getPart().T,slider_for_material.val))
        im2.set_array(t.getDerivetiveOfSensitivity().T)
        fig.canvas.draw()
        fig.canvas.flush_events()
        done = t.itterate()


