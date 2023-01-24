from ProblemMapper import *

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np



def randomCircleGenerator():
    x1 = np.random.random()
    y1 = np.random.random()
    r1 = np.random.random()/3
    f1 = np.random.random()*3
    a1 = np.random.random()*2*np.pi
    c1 = [x1,y1,r1,f1,a1]
    return c1


def main():
    nelx = 30
    nely = 30

    m = np.ones((nelx,nely))



    circle_1 = randomCircleGenerator()
    circle_2 = randomCircleGenerator()
    circle_3 = randomCircleGenerator()

    print(circle_1[:3])
    print(circle_2[:3])
    print(circle_3[:3])
    print()

    filledArea,supportArea,forceVector,minViableArea = mapProblemStatement2D(nelx,nely,circle_2,circle_1,circle_3,"y")


    fig,ax = plt.subplots(1)
    im1 = ax.imshow(filledArea, cmap='gray_r', interpolation='none',norm=colors.Normalize(vmin=0,vmax=1))
    #im2 = ax[1].imshow(supportArea, cmap='gray_r', interpolation='none',norm=colors.Normalize(vmin=0,vmax=1))
    #im4 = ax[2].imshow(minViableArea, cmap='gray_r', interpolation='none',norm=colors.Normalize(vmin=0,vmax=1))
    plt.show()



if __name__ == "__main__":
    main()
