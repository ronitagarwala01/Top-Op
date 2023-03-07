import numpy as np
import matplotlib.pyplot as plt
from problemStatementGenerator import *
from ProblemMapper import *
from massopt_fenics import *

from time import perf_counter


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

    YoungsModulusMax = 1e+12
    YoungsModulusMin = 1e+9
    
    CmaxRatio = 1e-5
    CminRatio = 1e-3
    ComplianceMinVal = 0

    SmaxRatio = 1e+8
    SminRatio = 1e+6
    StressMinVal = 0

    y,c,s = createConstraints(YoungsModulusMin,YoungsModulusMax,CmaxRatio,CminRatio,ComplianceMinVal,SmaxRatio,SminRatio,StressMinVal)

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



def generateData(numOr, numCon, numIter):
    
    start = perf_counter()
    for x in range(numOr):
        print("\n\n")
        _, formatted = generateProblemOrientation()

        for y in range(numCon):
            # formatted = generateProblemConditions(formatted)

            print("\n\nData Point:", x)
            print("Pass to fenics")

            iterations = fenicsOptimizer(formatted, numIter)

            # saveData(formatted, iterations)

        print("After fenics")
    end = perf_counter()

    time = end - start
    print(time)
        
    return

generateData(1, 1, 20)
# testBS()



