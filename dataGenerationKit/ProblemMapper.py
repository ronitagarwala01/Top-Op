"""
A small python library for transforming our problem statement into numpy arrays that can be accepted in the topopt program
"""

import numpy as np

def correctCircleOverlap(x:int,y:int,circlesArray,radiusScallingAxes:str="x"):
    """
    Reformats the circles arrays to account for the possible errors that may occur
    Shifts circles so that they do not overlap with the outer boundary and eachother

    Raises an error if the overlap cannot be fixed.

    Returns:
        - Circles Array(list): list of the circles that has been corrected for errors
        - RadusScallingFactor(float): The new radius scalling factor that can be used.
    """

    if(type(radiusScallingAxes) == int or type(radiusScallingAxes) == float):
        radiusScallingFactor = radiusScallingAxes
    elif(radiusScallingAxes == "x"):
        radiusScallingFactor = x
    else:
        radiusScallingFactor = y

    
    #check if circles go offscreen
    for i in range(len(circlesArray)):
        #Circles are formated as [x_coord,y_coord,radius,forceMagnitude,ForceAngle]
        c1_x = circlesArray[i][0] * x
        c1_y = circlesArray[i][1] * y
        c1_r = circlesArray[i][2] * radiusScallingFactor

        xCorrectionMade = False
        yCorrectionMade = False

        if(c1_x + c1_r > x):
            circlesArray[i][0] =  (x-c1_r)/x
            #print("Shift {} left to {}:{},{}".format(i,circlesArray[i][0],circlesArray[i][0] * x + c1_r,x))
            xCorrectionMade = True
        
        if(c1_x - c1_r < 0):
            if(xCorrectionMade):
                raise Exception("Error in generating Circles. Circle {} is out of bounds on x-axis and cannot be fixed.".format(i))
            else:
                circlesArray[i][0] =  c1_r/x
                #print("Shift {} right to {}:{},{}".format(i,circlesArray[i][0],circlesArray[i][0] * x - c1_r,0))
        
        if(c1_y + c1_r > y):
            circlesArray[i][1] =  (y-c1_r)/y
            #print("Shift {} down to {}:{},{}".format(i,circlesArray[i][1],circlesArray[i][1] * y + c1_r,y))
            yCorrectionMade = True
        
        if(c1_y - c1_r < 0):
            if(yCorrectionMade):
                raise Exception("Error in generating Circles. Circle {} is out of bounds on x-axis and cannot be fixed.".format(i))
            else:
                circlesArray[i][1] =  c1_r/y
            #print("Shift {} up to {}:{},{}".format(i,circlesArray[i][1],circlesArray[i][1] * y - c1_r,0))



    #check if circles overlap
    # correct radius scalling Factor
    for i,c1 in enumerate(circlesArray):
        for j,c2 in enumerate(circlesArray):
            if(i==j):
                continue
            else:
                #Circles are formated as [x_coord,y_coord,radius,forceMagnitude,ForceAngle]
                c1_x = c1[0] * x
                c1_y = c1[1] * y
                c1_r = c1[2]

                c2_x = c2[0] * x
                c2_y = c2[1] * y
                c2_r = c2[2]

                #check distance between circles
                distBetweenCirclesCenters = np.sqrt((c1_x-c2_x)**2 + (c1_y-c2_y)**2)
                #if the distance between midpoints is less than the radius of the circles then decrease the radius scalling factor
                if(np.floor(distBetweenCirclesCenters - radiusScallingFactor*(c1_r+c2_r)) <= 0):

                    radiusScallingFactor = np.floor(distBetweenCirclesCenters/(c1_r+c2_r))
                    if(radiusScallingFactor <= 0):
                        raise Exception("Error in generating Circles. Circle {} and Circle {} are too close together".format(i,j))
    return circlesArray, float(radiusScallingFactor)

def generateCircles(x:int,y:int,circlesArray,radiusScallingAxes:str="x"):
    """
    Creates a 2D boolean array where each circle is represented by a true value

    Inputs:
        - The x dimension of the grid to return
        - The y dimension of the grid to return
        - An array of circles defined as a tuple of [x_coord, y_coord, radius]
            - All values of the circels x,y coordinates as well as the radius should be in values in the interval [0,1] 
        - The axis by which the radius is scaled by in the event that the x and y sizes are different.radiusScallingAxes = "x" | "y".  Default "x"
        - If correctErrors is True then artifacts such as circles overlapping with each other and circles crossing outside the boundary will be fixed.
            - Fixing the artifacts may lead to unexpected outcomes.

    Returns:
        - A numpy boolean arrays of shape (x,y)
    """
    #print("Radius Scalling Axes = {}({})".format(radiusScallingAxes,type(radiusScallingAxes)))
    if(type(radiusScallingAxes) == int or type(radiusScallingAxes) == float):
        #print("radusi Scalling factor is Float: {}".format(radiusScallingAxes))
        radiusScallingFactor = radiusScallingAxes
    elif(radiusScallingAxes == "x"):
        radiusScallingFactor = x
        #print("radius Scalling factor is axis X: {}".format(x))
    else:
        radiusScallingFactor = y
        #print("radius Scalling factor is axis Y: {}".format(y))
        

    
    grid = np.zeros((x,y))
    #print(circlesArray)
    for c_x,c_y,c_r in circlesArray:
        #print(c_x,c_r,c_r)
        c_x *= x
        c_y *= y
        c_r *= radiusScallingFactor
        grid[int(c_x),int(c_y)] = 1
        for x1 in range(np.maximum(np.floor(c_x - c_r),0).astype("int32"),np.minimum(np.ceil(c_x + c_r) + 1,x).astype("int32")):
            for y1 in range(np.maximum(np.floor(c_y - c_r),0).astype("int32"),np.minimum(np.ceil(c_y + c_r) + 1,y).astype("int32")):
                if(np.sqrt((x1-c_x)**2 + (y1-c_y)**2 ) <= c_r):
                    grid[x1,y1] = 1
    booleanGrid = grid >= 1
    return booleanGrid

def generateCylinders(x:int,y:int,z:int,circlesArray,radiusScallingAxes:str="x"):
    """
    Creates a 3D boolean array where each Cylinder is represented by a true value

    Inputs:
        - The x dimension of the grid to return
        - The y dimension of the grid to return
        - An array of cylinders defined as a tuple of [x_coord, y_coord, radius]
            - All values of the cylinders x,y coordinates as well as the radius should be in values in the interval [0,1] 
            - Cylinders will exted throught hte entierty of the part so the cylinder hight is not needed.
        - The axis by which the radius is scaled by in the event that the x and y sizes are different.radiusScallingAxes = "x"|"y".  Default "x"
            - Radius cannot be scalled by the z axis
        - If correctErrors is True then artifacts such as Cylinders overlapping with each other and Cylinders crossing outside the boundary will be fixed.
            - Fixing the artifacts may lead to unexpected outcomes.

    Returns:
        - A numpy boolean arrays of shape (x,y,z)
    """
    twoDimensionalGrid = generateCircles(x,y,circlesArray,radiusScallingAxes)
    threeDimensionalGrid = np.repeat(twoDimensionalGrid[:,:,np.newaxis],z,axis=2)
    return threeDimensionalGrid

def generateForces2D(x:int,y:int,ForcesArray):
    """
    Creates a 1D array of forces, forces are indexed by their direction in each element
    
    Parameters:
        - The x dimension of the grid
        - The y dimension of the grid
        - The array containing the forces as tuples of [x_coord,y_coord,forceMagnitude,forceAngle]
            - the x and y coord of the forces should be in the interval [0-1]
            - The force magnitude will not be scalled
            - Force Angle is the angle the force is pointing in radians
        
    Returns:
        - A 1D arrays of size 2*(x+1)*(y+1) representing the degree of freedom where the force is applied
    """
    ndof = 2*(x+1)*(y+1)
    forceVector = np.zeros((ndof,2*len(ForcesArray)))
    i = 0

    for x1,y1,mag,angle in ForcesArray:
        x1 = int(x1*x)
        y1 = int(y1*y)
        fx = mag*np.cos(angle)
        fy = mag*np.sin(angle)

        index = 2*((y+1)*x1 + y1)

        forceVector[index,i] = fx
        forceVector[index+1,i+1] = fy
        i += 2
    

    return forceVector

def generateForces3D(x:int,y:int,z:int,ForcesArray):
    """
    Creates a 1D Sparce maxtirx of forces, forces are indexed by their direction in each element
    Returns the indexing method needed to construct such a sparce matrix
    
    Parameters:
        - The x dimension of the grid
        - The y dimension of the grid
        - The z dimension of the grid
        - The array containing the forces as tuples of [x_coord,y_coord,forceMagnitude,forceAngleTheta,forceAnglePhi]
            - the x and y coord of the forces should be in the interval [0-1], z coord is not needed
            - The force magnitude will not be scalled
            - Force Angle is the angle the force is pointing in radians
                - Theta represents the angle along the x-y axis(azimuth angle) in the interval [0,2*pi]
                - phi represent the angle offset from the z axis (polar angle) in the interval [0,pi] where pi/2 is along the x-y axis
        
    Returns:
        
        - fv: ForceValues An array containing the values to fill a sparce matrix
        - fi: ForceIndexes An array containing the indexes for the coresponding values to fill the sparce matrix
            - vf and vi can be used to build an array of size 3*(x+1)*(y+1)*(z+1) representing the degree of freedom where the force is applied
            - build as an array of zeros where array[fi[n]] = fv[n] for some n in range(len(fi))
    """
    ndof = 3*(x+1)*(y+1)*(z+1)
    fv_temp = []
    fi_temp = []

    for x1,y1,mag,theta,phi in ForcesArray:
        x1 = int(x1*x)
        y1 = int(y1*y)

        #since the force is distributed across the cylinder the magnitude is divided accordingly
        mag /= z

        fx = mag*np.cos(theta)*np.sin(phi)
        fy = mag*np.sin(theta)*np.sin(phi)
        fz = mag*np.cos(phi)

        for z1 in range(z):

            NID1 = z1*(x+1)*(y+1) + x1*(y+1) + (y+1 - y1)
            index_x = 3*NID1 - 2
            index_y = 3*NID1 - 1
            index_z = 3*NID1

            fv_temp.append(fx)
            fi_temp.append(index_x)

            fv_temp.append(fy)
            fi_temp.append(index_y)

            fv_temp.append(fz)
            fi_temp.append(index_z)
            
    fv = np.array(fv)
    fi = np.array(fi).astype("int32")
        
    

    return fv,fi

def mapProblemStatement2D(x:int,y:int,circle1Data,circle2Data,circle3Data,radiusScallingAxes:str="x",correctError:bool=True):
    """
    creates a list of properties needed to format the original problem statement into one that can be used by the topopt program

    Parameters:
        - The x dimension of the grid to return
        - The y dimension of the grid to return
        - A list contininf the data for the three circles that will be used
            - Each should be a list containing [x_coord,y_coord,radius,forceMagnitude,ForceAngle]
            - All values of the circles x,y coordinates as well as the radius should be in values in the interval [0,1] 
            - The force magnitude will not be scalled
            - Force Angle is the angle the force is pointing in radians
        - The axis by which the radius is scaled by in the event that the x and y sizes are different.radiusScallingAxes = "x" | "y".  Default "x"
        - If correctErrors is True then artifacts such as circles overlapping with each other and circles crossing outside the boundary will be fixed.
            - Fixing the artifacts may lead to unexpected outcomes.

    Returns:
        - filled area that will have solid material allways
        - support area that will be fixed in place(not move)
        - force vector 
        - minimum viable area that can be filled to link all three circles together
    """
    #print(circle1Data)
    #print(circle2Data)
    #print(circle3Data)
    if(correctError):
        circlesArray, radiusScallingFactor = correctCircleOverlap(x,y,[circle1Data,circle2Data,circle3Data],radiusScallingAxes)
    else:
        circlesArray = [circle1Data,circle2Data,circle3Data]
        radiusScallingFactor = radiusScallingAxes
    #print(circlesArray[0])
    #print(circlesArray[1])
    #print(circlesArray[2])

    c1_circleData = [circlesArray[0][0],circlesArray[0][1],circlesArray[0][2]]
    c2_circleData = [circlesArray[1][0],circlesArray[1][1],circlesArray[1][2]]
    c3_circleData = [circlesArray[2][0],circlesArray[2][1],circlesArray[2][2]]

    c1_forceData = [circlesArray[0][0],circlesArray[0][1],circlesArray[0][3],circlesArray[0][4]]
    c2_forceData = [circlesArray[1][0],circlesArray[1][1],circlesArray[1][3],circlesArray[1][4]]
    c3_forceData = [circlesArray[2][0],circlesArray[2][1],circlesArray[2][3],circlesArray[2][4]]

    filledArea = generateCircles(x,y,[c1_circleData,c2_circleData,c3_circleData],radiusScallingFactor)

    supportArea = generateCircles(x,y,[c1_circleData],radiusScallingFactor)

    def polarAddition(magnitude1,theta1,magnitude2,theta2):
        x1 = magnitude1*np.cos(theta1)
        y1 = magnitude1*np.sin(theta1)

        x2 = magnitude2*np.cos(theta2)
        y2 = magnitude2*np.sin(theta2)

        magnitude3 = np.sqrt((x1+x2)**2 + (y1+y2)**2)
        theta3 = np.arctan2(y1+y2,x1+x2)

        return magnitude3,theta3
    
    force2_newMagnitude,force2_newAngle = polarAddition(-c1_forceData[2]/2,c1_forceData[3],c2_forceData[2],c2_forceData[3])
    force3_newMagnitude,force3_newAngle = polarAddition(-c1_forceData[2]/2,c1_forceData[3],c3_forceData[2],c3_forceData[3])

    newForce2 = [c2_forceData[0],c2_forceData[1],force2_newMagnitude,force2_newAngle]
    newForce3 = [c3_forceData[0],c3_forceData[1],force3_newMagnitude,force3_newAngle]

    forceVector = generateForces2D(x,y,[newForce2,newForce3])
    minViableArea = generateMinimumMemberAccordinToProblemStatement2D(x,y,[c1_circleData,c2_circleData,c3_circleData],radiusScallingFactor)

    return filledArea,supportArea,forceVector,minViableArea

def generateMinimumMemberAccordinToProblemStatement2D(x_size:int,y_size:int,circlesArray,radiusScallingFactor):         
    grid = np.zeros((x_size,y_size))
    for i,c1 in enumerate(circlesArray):
        for j,c2 in enumerate(circlesArray):
            if(i==j):
                continue
            else:
                c1_x = int(c1[0] * x_size)
                c1_y = int(c1[1] * y_size)
                c1_r = c1[2] * radiusScallingFactor

                c2_x = int(c2[0] * x_size)
                c2_y = int(c2[1] * y_size)
                c2_r = c2[2] * radiusScallingFactor

                yOffset = abs(c1_y - c2_y)
                xOffset = abs(c1_x - c2_x)

                if(xOffset >= yOffset):
                    if(c1_x < c2_x):
                        start_x = c1_x
                        start_y = c1_y
                        startRadius = c1_r
                        end_x = c2_x
                        end_y = c2_y
                        endRadius = c2_r
                    else:
                        start_x = c2_x
                        start_y = c2_y
                        startRadius = c2_r
                        end_x = c1_x
                        end_y = c1_y
                        endRadius = c1_r
                    #print(i,c1,j,c2)
                    #print("({}-{})/({}-{})".format(end_y,start_y,end_x,start_x))
                    lineSlope = (end_y-start_y)/(end_x-start_x)
                    #printThing(start_x,start_y,startRadius,end_x,end_y,endRadius)
                    for x in range(start_x,end_x+1):
                        midPoint_y = lineSlope * (x-start_x) + start_y
                        interpolate = 1-(x-start_x)/(end_x-start_x)
                        currentRadius = startRadius * interpolate + (1-interpolate)*endRadius
                        for y in range(int(np.floor(midPoint_y - currentRadius)),int(np.ceil(midPoint_y + currentRadius))):
                            grid[max(0,min(x,x_size-1)),max(0,min(y,y_size-1))] = 1
                else:
                    if(c1_y < c2_y):
                        start_x = c1_x
                        start_y = c1_y
                        startRadius = c1_r
                        end_x = c2_x
                        end_y = c2_y
                        endRadius = c2_r
                    else:
                        start_x = c2_x
                        start_y = c2_y
                        startRadius = c2_r
                        end_x = c1_x
                        end_y = c1_y
                        endRadius = c1_r
                    #print("({}-{})/({}-{})".format(end_y,start_y,end_x,start_x))
                    lineSlope = (end_x-start_x)/(end_y-start_y)
                    for y in range(start_y,end_y+1):
                        midPoint_x = lineSlope * (y-start_y) + start_x
                        interpolate = 1-(y-start_y)/(end_y-start_y)
                        currentRadius = startRadius * interpolate + (1-interpolate)*endRadius
                        for x in range(int(np.floor(midPoint_x - currentRadius)),int(np.ceil(midPoint_x + currentRadius))):
                            grid[max(0,min(x,x_size-1)),max(0,min(y,y_size-1))] = 1


    
    for c_x,c_y,c_r in circlesArray:
        c_x *= x_size
        c_y *= y_size
        c_r *= radiusScallingFactor
        for x1 in range(int(np.maximum(np.floor(c_x - c_r),0)),int(np.minimum(np.ceil(c_x + c_r) + 1,x_size))):
            for y1 in range(np.maximum(np.floor(c_y - c_r),0).astype("int32"),np.minimum(np.ceil(c_y + c_r) + 1,y_size).astype("int32")):
                if(np.sqrt((x1-c_x)**2 + (y1-c_y)**2 ) <= c_r):
                    grid[x1,y1] = 1
    booleanGrid = grid >= 1
    return booleanGrid

def printThing(x1,y1,r1,x2,y2,r2):
    print(x1,y1,r1)
    print(x2,y2,r2)
    print()









