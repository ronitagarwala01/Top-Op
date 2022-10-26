# A 165 LINE TOPOLOGY OPTIMIZATION CODE BY NIELS AAGE AND VILLADS EGEDE JOHANSEN, JANUARY 2013
import numpy as np

from matplotlib import colors
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from topopt import topOpter






# The real main driver    
if __name__ == "__main__":
    # Default input parameters
    nelx=160
    nely=80
    volfrac=0.4
    rmin=5.4
    penal=3.0
    ft=0 # ft==0 -> sens, ft==1 -> dens
    # The variables are in order: x position of cylinder, y position of cylinder, radius of the cylinder, the magnitude of the force,
	# and the counterclockwise angle of the force in degrees.
    circle_1 = [50,60,20,40,63]
    circle_2 = [35,20,15,50,115]
    circle_3 = [20,60,10,60,275]

    loads = [circle_1,circle_2,circle_3]
    import sys
    if len(sys.argv)>1: nelx   =int(sys.argv[1])
    if len(sys.argv)>2: nely   =int(sys.argv[2])
    if len(sys.argv)>3: volfrac=float(sys.argv[3])
    if len(sys.argv)>4: rmin   =float(sys.argv[4])
    if len(sys.argv)>5: penal  =float(sys.argv[5])
    if len(sys.argv)>6: ft     =int(sys.argv[6])
    print(loads)



    t = topOpter(nelx,nely,volfrac,penal,rmin,ft)
    t.updateLoads(loads)
    t.updateForceVectors([[1,1,0,1]])

    done = False
    while(not done):
        done = t.itterate()


