import numpy as np
import matplotlib.pyplot as plt
from problemStatementGenerator import *
from ProblemMapper import *
from massopt_5 import *

def generateProblem(nelx=100, nely=50, C_max=20.0, S_max=3.0e+6, Y=2.0e+11):

    xDim, yDim = calcRatio(nelx, nely) # Length, Width

    c1, c2, c3, forces = FenicsCircleAndForceGenerator(xDim, yDim)

    initial_conditions = [c1, c2, c3, forces, nelx, nely, Y, C_max, S_max]

    def unpackConditions(conditions):
        unpackedConditions = []

        circles = initial_conditions[:3]

        for x in range(3):
            for variable in circles[x]:
                unpackedConditions.append(variable)

            unpackedConditions.append(forces[0, x])
            unpackedConditions.append(forces[1, x])

        for x in range(4, len(conditions)):
            unpackedConditions.append(conditions[x])

        return unpackedConditions
    
    
    def formatForFenics(conditions):
        cx, cy = [], []
        formattedCircles, radii = [], []

        circles = initial_conditions[:3]

        for circle in circles:
            cx.append(circle[0])
            cy.append(circle[1])
            radii.append(circle[2])

        formattedCircles.append(np.array(cx))
        formattedCircles.append(np.array(cy))

        formattedCircles = np.array(formattedCircles)
        radii = np.array(radii)

        formattedConditions = [formattedCircles, radii]


        for x in range(3, len(conditions)):
            formattedConditions.append(conditions[x])

        return formattedConditions
    

    unpackedConditions = np.array(unpackConditions(initial_conditions))
    formattedConditions = np.array(formatForFenics(initial_conditions), dtype=object)



    return unpackedConditions, formattedConditions



def generateData(numPoints):
    
    for x in range(numPoints):
        print("\n\n")
        unpacked, formatted = generateProblem()

        print("\n\nData Point:", x)
        print("Pass to fenics")
        fenicsOptimizer(formatted)
        print("After fenics")
        
    return

generateData(5)

