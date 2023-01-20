import numpy as np

from GA import *
from GA_score_compliance import topOpter
from ProblemMapper import *
from FileSaver import AgentFileSaver
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.widgets import Slider
from random import randint


def dispayMember(figure,figImage,member):
    if(plt.fignum_exists(figure.number)):
        figImage.set_array(member)
        figure.canvas.draw()
        figure.canvas.flush_events()

def main():
    #start by defining the topopt problem we will solve
    nelx=30
    nely=30
    volfrac=0.4
    rmin=5.4
    penal=3.0
    ft=0
    maxCompliance = 30

    DisplayFlag = True
    # The variables are in order: x position of cylinder, y position of cylinder, radius of the cylinder, the magnitude of the force,
    # and the counterclockwise angle of the force in degrees.
    circle_1 = [.15,.15,.1,1,(3/2)*np.pi]
    circle_2 = [.5,.5,.2,1,(1/2)*np.pi]
    circle_3 = [.85,.85,.1,1,(3/2)*np.pi]

    filledArea,supportArea,forceVector,minViableArea = mapProblemStatement2D(nelx,nely,circle_2,circle_1,circle_3,"y")

    fig,ax = plt.subplots(1,1)
    im1 = ax.imshow(minViableArea, cmap='gray_r', interpolation='none',norm=colors.Normalize(vmin=0,vmax=1))
    plt.show()







if __name__ == "__main__":
    main()