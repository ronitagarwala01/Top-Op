# A 165 LINE TOPOLOGY OPTIMIZATION CODE BY NIELS AAGE AND VILLADS EGEDE JOHANSEN, JANUARY 2013
import numpy as np

from matplotlib import colors
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

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
            filledArea,supportArea,forceVector, _ = mapProblemStatement2D(nelx,nely,circle_2,circle_1,circle_3,"y")
        except Exception as e:
            canSetUp = False
        else:
            canSetUp = True
            return filledArea,supportArea,forceVector
    
    if(canSetUp == False):
        # The variables are in order: x position of cylinder, y position of cylinder, radius of the cylinder, the magnitude of the force,
        # and the counterclockwise angle of the force in degrees.

        circle_1 = [.15,.15,.1,1,(3/2)*np.pi]
        circle_2 = [.5,.5,.2,1,(1/2)*np.pi]
        circle_3 = [.85,.85,.1,1,(3/2)*np.pi]
        filledArea,supportArea,forceVector, _ = mapProblemStatement2D(nelx,nely,circle_2,circle_1,circle_3,"y")
        return filledArea,supportArea,forceVector
