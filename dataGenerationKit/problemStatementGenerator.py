import numpy as np
import matplotlib.pyplot as plt

def calcRatio(a, b):
    """
    Finds the ratio between two numbers. Used to prevent FEniCS from freaking out.
    Aka, in xDim, yDim, and L, W within massopt_n.py
    """
    gcd = np.gcd(a, b)

    aReduced = a / gcd
    bReduced = b / gcd
    
    return aReduced, bReduced

def correctCircleOverlap2(x:float,y:float,circlesArray):
    """
    Reformats the circles arrays to account for the possible errors that may occur
    Shifts circles so that they do not overlap with the outer boundary and eachother

    Raises an error if the overlap cannot be fixed.

    parameters:
        x(float): ratio of x to y dimension
        y(float): ratio of x to y dimension
            - for an outut that is twice as wide as it is tall x=2 and y=1
            - for an output that is tall and skinny x=1 y=3


    Returns:
        - Circles Array(list): list of the circles that has been corrected for errors
    """
    
    #check if circles go offscreen
    for i in range(len(circlesArray)):
        #Circles are formated as [x_coord,y_coord,radius,forceMagnitude,ForceAngle]
        c1_x = circlesArray[i][0]
        c1_y = circlesArray[i][1]
        c1_r = circlesArray[i][2]

        xCorrectionMade = False
        yCorrectionMade = False

        #check the circle for the right wall colision
        if(c1_x + c1_r > x):
            circlesArray[i][0] =  (x-c1_r)
            print("Shift {} left to {}:{},{}".format(i,circlesArray[i][0],circlesArray[i][0] * x + c1_r,x))
            xCorrectionMade = True
        
        #check the circle for the left wall colision
        if(c1_x - c1_r < 0):
            if(xCorrectionMade):
                raise Exception("Error in generating Circles. Circle {} is out of bounds on x-axis and cannot be fixed.".format(i))
            else:
                circlesArray[i][0] =  c1_r
                print("Shift {} right to {}:{},{}".format(i,circlesArray[i][0],circlesArray[i][0] * x - c1_r,0))
        
        #check the circle for the floor colision
        if(c1_y + c1_r > y):
            circlesArray[i][1] =  (y-c1_r)
            print("Shift {} down to {}:{},{}".format(i,circlesArray[i][1],circlesArray[i][1] * y + c1_r,y))
            yCorrectionMade = True
        
        #check the circle for the ceiling colision
        if(c1_y - c1_r < 0):
            if(yCorrectionMade):
                raise Exception("Error in generating Circles. Circle {} is out of bounds on x-axis and cannot be fixed.".format(i))
            else:
                circlesArray[i][1] =  c1_r
            print("Shift {} up to {}:{},{}".format(i,circlesArray[i][1],circlesArray[i][1] * y - c1_r,0))


    """
    Radius scaling factor will shrink to fit the circle radiuses so they do not overlap
    This number will then be multiplied to the circles after it has been fully minimized

    When the first checks are run this number should be 1 but will get smaller if an overlap occurs
    """
    minRadius = 0.01
    radiusScallingFactor = 1
    #check if circles overlap
    # correct radius scalling Factor
    for i,c1 in enumerate(circlesArray):
        for j,c2 in enumerate(circlesArray):
            if(i==j):
                continue
            else:
                #Circles are formated as [x_coord,y_coord,radius,forceMagnitude,ForceAngle]
                c1_x = c1[0]
                c1_y = c1[1]
                c1_r = c1[2] * radiusScallingFactor

                c2_x = c2[0]
                c2_y = c2[1]
                c2_r = c2[2] * radiusScallingFactor

                #check distance between circles
                distBetweenCirclesCenters = np.sqrt((c1_x-c2_x)**2 + (c1_y-c2_y)**2)
                #if the distance between midpoints is less than the radius of the circles then decrease the radius scalling factor
                print(f'Comparing {i} to {j}: dist: {distBetweenCirclesCenters} - {radiusScallingFactor*(c1_r+c2_r)} = {(distBetweenCirclesCenters - radiusScallingFactor*(c1_r+c2_r))}')
                if((distBetweenCirclesCenters - (radiusScallingFactor*(c1_r+c2_r))) <= 0):

                    radiusScallingFactor = (distBetweenCirclesCenters/(radiusScallingFactor*(c1_r+c2_r))) * 0.95 #this .95 multiplier ensures that the circles will not touch.
                    print(f"\tNew Radius scalling factor is: {radiusScallingFactor}.")
                    if(radiusScallingFactor*c2_r <= minRadius):
                        raise Exception("Error in generating Circles. Circle {} and Circle {} are too close together".format(i,j))
    for i in range(len(circlesArray)):
        #Circles are formated as [x_coord,y_coord,radius,forceMagnitude,ForceAngle]
        
        circlesArray[i][2] = circlesArray[i][2] * radiusScallingFactor
    
    return circlesArray

def randomCircleGenerator(x,y,maxForce:float,minForce:float=.1):
    x1 = np.random.random()*x
    y1 = np.random.random()*y
    r1 = np.random.random()/2
    if(r1 <= 0.05):
        r1 = 0.1
    f1 = (np.random.random()*(maxForce-minForce)) + minForce
    a1 = np.random.random()*2*np.pi
    c1 = [x1,y1,r1,f1,a1]
    return c1

def forceEquilizer(x,y,circle_1,circle_2):
    #[x1,y1,r1,f1,a1]
    fx = circle_1[3]*np.cos(circle_1[4]) + circle_2[3]*np.cos(circle_2[4])
    fy = circle_1[3]*np.sin(circle_1[4]) + circle_2[3]*np.sin(circle_2[4])

    counterForce_x = -fx
    counterForce_y = -fy

    magnitude = np.linalg.norm([counterForce_x,counterForce_y],ord=2)
    angle = np.arctan2(counterForce_y,counterForce_x)

    x1 = np.random.random()*x
    y1 = np.random.random()*y
    r1 = np.random.random()/3
    if(r1 <= 0.05):
        r1 = 0.1

    c1 = [x1,y1,r1,magnitude,angle]
    return c1

def generateRandomProblemStatement_2D(nelx,nely):
    """
    Given a grid dimension create a 2D problem statement with three circels and three forces that all balence out.

    Does this by generateing two random circles as well as a third circle to balence out the forces.
    Then corrects the possible overlapping of the circles on a continious grid
    """
    setupTries = 1000
    canSetUp = False
    for i in range(setupTries):
        try:                                                                            
            circle_1 = randomCircleGenerator(nelx,nely,3)
            circle_2 = randomCircleGenerator(nelx,nely,3)
            circle_3 = forceEquilizer(nelx,nely,circle_1,circle_2)

            circle_1,circle_2,circle_3 = correctCircleOverlap2(nelx,nely,[circle_1,circle_2,circle_3])

        except Exception as e:
            canSetUp = False
            print('\nNewProblem\n')
        else:
            canSetUp = True
            return circle_1,circle_2,circle_3
    
    if(canSetUp == False):
        # The variables are in order: x position of cylinder, y position of cylinder, radius of the cylinder, the magnitude of the force,
        # and the counterclockwise angle of the force in degrees.

        circle_1 = [.2,.15,.1,1,(3/2)*np.pi]
        circle_2 = [.75,.5,.2,1,(1/2)*np.pi]
        circle_3 = [1.3,.85,.1,1,(3/2)*np.pi]
        return circle_1,circle_2,circle_3

def polarToCartesian(angle,magnitude):
    x = magnitude * np.cos(angle)
    y = magnitude * np.sin(angle)
    return x,y

def FenicsCircleAndForceGenerator(xDim,yDim):
    """
    Builds a set of three circles and forces inside each that sum to zero
    Formats the three circles as:
        - [x_Location,y_Location,radius]
    Formats the forces as a 2D array columns indicate circle number, and rows indicate force direction

    Will prevent circles from overlaping themselves or edges, but somtimes only spawns in three circles

    Returns:
        - circle1,circle2,circle3,forces
    """

    c1,c2,c3 = generateRandomProblemStatement_2D(xDim,yDim)

    circle1 = [c1[0],c1[1],c1[2]]
    circle2 = [c2[0],c2[1],c2[2]]
    circle3 = [c3[0],c3[1],c3[2]]

    forces = np.zeros((2,3))
    x,y = polarToCartesian(c1[4],c1[3])
    forces[0,0] = x
    forces[1,0] = y

    x,y = polarToCartesian(c2[4],c2[3])
    forces[0,1] = x
    forces[1,1] = y

    x,y = polarToCartesian(c3[4],c3[3])
    forces[0,2] = x
    forces[1,2] = y

    return circle1,circle2,circle3,forces
    

def test():
    c1,c2,c3 = generateRandomProblemStatement_2D(2,1)
    print(c1[:2],c1[2])
    print(c2[:2],c2[2])
    print(c3[:2],c3[2])

    fig, ax = plt.subplots(1,1)
    res = 500
    x= np.linspace(0,2,res)
    y = np.linspace(0,1,res//2)
    X,Y = np.meshgrid(x,y)
    l1 = 1
    def dist(circle_def):#[x1,y1,r1,f1,a1
        return np.sqrt((circle_def[0]-X)**2 + (circle_def[1]-Y)**2) - circle_def[2]
    Z1 = np.minimum(dist(c1),np.minimum(dist(c2),dist(c3)))
    #Z1 = dist(c1)

    Z1 = np.where(Z1>0,1,Z1)
    Z1 = np.where(Z1<=0,-1,Z1)

    print(Z1.shape)
    ax.imshow(Z1,cmap='gray')


    plt.show()


if(__name__ == "__main__"):
    test()

