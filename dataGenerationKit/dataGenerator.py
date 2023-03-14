import numpy as np
import matplotlib.pyplot as plt
from problemStatementGenerator import *
from massopt_fenics import *
from DataSaver import *

from time import perf_counter


# Number of data points are these two values multiplied.
# Run the file after setting them, everything else is set up.
# Honestly, running in batches of 5 might be smart. 
# Though it really doesn't matter, since it saves after each optimization regardless
numberOfProblemOrientations = 10          # Circle locations, forces, etc.
numberOfConditionsChanges =  5         # Young's Modulus, C_max, S_max


def generateProblemOrientation(nelx=100, nely=50, C_max=2.0e-3, S_max=3.0e+7, Y=3.5e+11):

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


def generateProblemConditions(formatted):
    # formatted = [circles, radii, forces, nelx, nely, Y, C_max, S_max]

    YoungsModulusMax = 5.0e+11
    YoungsModulusMin = 5.0e+10
    
    CmaxRatio = 50.0
    CminRatio = 2.0
    # ComplianceMinVal = 0

    SmaxRatio = 500.0
    SminRatio = 100.0
    # StressMinVal = 0

    y,c,s = createConstraints(YoungsModulusMin,YoungsModulusMax,CmaxRatio,CminRatio,SmaxRatio,SminRatio)

    formatted[5] = y
    formatted[6] = c
    formatted[7] = s

    return formatted
    
def testBS():
    for i in range(10):
        print("\n", i)    
        _, formatted = generateProblemOrientation()
        formatted = generateProblemConditions(formatted)

        # print(formatted)

        circles = formatted[0]
        radii = formatted[1]
        forces = formatted[2]
        nelx, nely = formatted[3], formatted[4]
        Y, C_max, S_max = formatted[5], formatted[6], formatted[7]

        np.set_printoptions(suppress=False, precision=1000)
        print("Y",np.array([Y]))
        print("C", np.array([C_max]))
        print("S", np.array([S_max]))


def generateData(numOr, numCon):
    
    for x in range(numOr):
        print("\n\n")
        _, formatted = generateProblemOrientation(nelx=100, nely=50)

        for y in range(numCon):
            formatted = generateProblemConditions(formatted)

            print("\n\nData Point:", x)
            print("Pass to fenics")

            solutions_list, objective_list, derivative_list, C_max, S_max, converged = fenicsOptimizer(formatted)
            formatted[6] = C_max
            formatted[7] = S_max
            print("After fenics")

            saveData(formatted, solutions_list, objective_list, derivative_list, converged)            
            
    return


# 
# Utility Functions
# 
def extractData():
    conditions, x, der, obj = getData('Agents/40_20/Agent_370494')

    # print(x)
    # lastIteration = x[-1]
    # lastIteration = np.reshape(lastIteration, newshape=(100,50))

    # plt.imshow(lastIteration)

    sol, obj, der, cm, sm, vm, c = fenicsOptimizer(conditions)

# extractData()

def testIterLength():
    conditions, x, der = getData('Agents/40_20/Agent_225240')

    for iter in x:
        print(iter.shape)

    return

# testIterLength()


generateData(numberOfProblemOrientations, numberOfConditionsChanges)