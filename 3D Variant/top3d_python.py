"""
This code needs to be converted from matlab to python

You will need to convert the matlab code into python code.
most of this will require changing the array indexing
This is an attempt at a 1:1 translation from MATLAB to python, but is more like 1:3



Notes:
    - Still requires a way to input different load and constraaint conditions

    

"""
# A 169 LINE 3D TOPOLOGY OPTIMIZATION CODE BY LIU AND TOVAR (JUL 2013)
# https://link.springer.com/article/10.1007/s00158-014-1107-x#Tab2
import numpy as np
from numpy.matlib import repmat
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from matplotlib import colors
import matplotlib.pyplot as plt
from math import floor

class TopOpt3D:
    def __init__(self,nelx,nely,nelz,volfrac,penal,rmin):
        self.nelx = nelx
        self.nely = nely
        self.nelz = nelz
        self.volfrac = volfrac
        self.penal = penal
        self.rmin = rmin
        self.maxloop = 200    # Maximum number of iterations
        self.tolx = 0.01      # Terminarion criterion
        displayflag = 0  # Display structure flag

        # USER-DEFINED MATERIAL PROPERTIES
        self.E0 = 1           # Young's modulus of solid material
        self.Emin = 1e-9      # Young's modulus of void-like material
        self.nu = 0.3         # Poisson's ratio
        
        # PREPARE FINITE ELEMENT ANALYSIS
        self.nele = nelx*nely*nelz
        self.ndof = 3*(nelx+1)*(nely+1)*(nelz+1)

        
        

        # USER-DEFINED SUPPORT FIXED DOFs
        iif,jf,kf = np.meshgrid(0,np.arange(nely),np.arange(nelz))  # Coordinates
        fixednid = kf*(nelx+1)*(nely+1)+iif*(nely+1)+(nely+1-jf)    # Node IDs
        self.fixeddof = np.array([[3*fixednid[:]], [3*fixednid[:]-1], [3*fixednid[:]-2]])  # DOFs

    
        # # USER-DEFINED LOAD DOFs
        #apply Force along the top x values
        il,jl,kl = np.meshgrid(nelx, 0, np.arange(nelz))       # Coordinates
        loadnid = kl*(nelx+1)*(nely+1)+il*(nely+1)+(nely+1-jl) # Node IDs
        loaddof = 3*loadnid[:] - 1                             # DOFs
        
        value = -np.ones(nelx)
        vi = np.reshape(loaddof,nelx)
        self.F = self.ApplyForcesAsSparce(value,vi)
        
        #U = np.zeros(self.ndof)
        self.freedofs = np.setdiff1d(np.arange(self.ndof),self.fixeddof)
        self.KE = self.lk_H8()
        nodegrd = np.reshape(np.arange((nely+1)*(nelx+1)),(nely+1,nelx+1)) #node grid
        nodeids = np.reshape(nodegrd[0:-1,0:-1],(nely*nelx,1)) #node ids 


        #nodeidz = (0:(nely+1)*(nelx+1):(nelz-1)*(nely+1)*(nelx+1))
        nodeidz = np.arange(0,(nelz-1)*(nely+1)*(nelx+1),step=(nely+1)*(nelx+1))
        #print(nodeidz.shape,nodeids.shape)
        nodeids = nodeids* np.ones((len(nodeids),len(nodeidz))) + nodeidz*np.ones((len(nodeids),len(nodeidz)))

        edofVec = 3*nodeids[:]+1

        #print(edofVec)
        self.edofMat = repmat(edofVec,1,24) 
        #print("\nElement degrees of freedome vector:")
        #print(edofVec.shape)

        self.edofMat = self.generateElementDegreesOfFreedomMatrix()


        # PREPARE FILTER
        """
        Here, our filter is represented by a 2D sparse matrix  build witht he following conditions:
            - iH and jH are arrays holding the indexes to where sH will go in our matrix
                - This means that for and index n the filter matrix H[iH[n],jH[n]] = sH[n]
            - iH is build from e1 which is the element number we are working with
                - thus each row of our filter matrix will represent an elements filter values
            - jH is build from e2 which is the secondary element we are using to filter the first element
            - sH is a filter weight build as a gaussian sphere where the filter is strongest at the center and progressivly moves to zero when the element is outisde of rmin
        """
        iH = np.ones(self.nele*((2*floor(rmin))**3))
        jH = np.ones(len(iH))
        sH = np.zeros(len(iH))
        k = 0
        for k1 in range(nelz):
            for i1 in range(nelx):
                for j1 in range(nely):
                    e1 = int(k1*nelx*nely + i1*nely + j1)
                    #changed all the int(np.ceil(rmin))-1 to floor(rmin)
                    for k2 in range(            max(k1-floor(rmin),0),   min(k1+floor(rmin),nelz)):
                        for i2 in range(        max(i1-floor(rmin),0),   min(i1+floor(rmin),nelx)):
                            for j2 in range(    max(j1-floor(rmin),0),   min(j1+floor(rmin),nely)):
                                e2 = int(k2*nelx*nely + i2*nely + j2)
                                iH[k] = e1
                                jH[k] = e2
                                sH[k] = max(0,rmin-np.sqrt((i1-i2)**2+(j1-j2)**2+(k1-k2)**2))
                                k+=1
        #print("k:",k)
        #print("len(iH):",len(iH))
    


        #H = sparse(iH,jH,sH)
        self.H = coo_matrix((sH, (iH,jH)),shape = (self.nele,self.nele)).tocsc() 
        self.Hs = self.H.sum(1)

        #print("Filter matrix H:",self.H.shape)
        #print()
        #print("H sum", Hs)

        self.iK = np.reshape(np.kron(self.edofMat,np.ones((24,1))).T, (24*24*self.nele)).astype('int32')
        self.jK = np.reshape(np.kron(self.edofMat,np.ones((1,24))).T, (24*24*self.nele)).astype('int32')

        # INITIALIZE ITERATIOn
        self.x = volfrac * np.ones((nely,nelx,nelz))
        self.xPhys = self.x.copy() 
        self.loop = 0 
        self.change = 1

    def itterationStep(self):
        # START ITERATION
        if (self.change > self.tolx and self.loop < self.maxloop):
            self.loop += 1

            # FE-ANALYSIS
            K = self.buildCurrentStiffnessMatrix(self.xPhys)
            # Remove constrained dofs from matrix
            K = K[self.freedofs,:][:,self.freedofs]

            #Solve for displacement vector
            U = np.zeros(self.ndof)
            U[self.freedofs] = spsolve(K,self.F[self.freedofs])

            # OBJECTIVE FUNCTION AND SENSITIVITY ANALYSIS
            c,dc = self.sensitivityAnalysis(U)
            dv = np.ones((self.nely,self.nelx,self.nelz))

            # FILTERING AND MODIFICATION OF SENSITIVITIES
            dc = self.filterAcrossElements(dc)
            dv = self.filterAcrossElements(dv)

            # OPTIMALITY CRITERIA UPDATE
            xnew,xPhys = self.OptimalityCriterion(dc,dv,self.x)

            self.change = np.linalg.norm(np.reshape(xnew,(self.nele,1))-np.reshape(self.x,(self.nele,1)), ord = np.inf)
            self.x = xnew
            self.xPhys = xPhys

            # PRINT RESULTS
            print("it.: {0} , obj.: {1:.3f} Vol.: {2:.3f}, ch.: {3:.3f}".format(self.loop,c,np.mean(self.xPhys[:]),self.change))
            return True
        else:
            self.saveToCSV("output3Dshape.csv")
            return False

    def buildCurrentStiffnessMatrix(self,x):
        """
        Build a the element stiffeness matrix based off the given element values

        takes the x as a 3D matrix of nelx,nely,nelz
        returns a matrix K which is Symetric of size ndof,ndof where ndof is 3*(nelx+1)*(nely+1)*(nelz+1)
        """
        YoungsModulusForElementsWithPenalty = np.reshape(self.Emin+(np.power(x.T,self.penal)) * (self.E0-self.Emin),(self.nele,1))
        KE_flattened = np.reshape(self.KE,(24*24,1))
        toReshape = KE_flattened @ YoungsModulusForElementsWithPenalty.T
        sK = np.reshape(toReshape,(24*24*self.nele))

        K = coo_matrix((sK,(self.iK,self.jK)),shape=(self.ndof,self.ndof)).tocsc() 
        K = (K+K.T)/2
        return K

    def sensitivityAnalysis(self,U):
        U_times_KE = (U[self.edofMat]@self.KE)
        U_KE_U = U_times_KE*U[self.edofMat]
        ce = np.reshape(np.sum(U_KE_U,axis=1),[self.nely,self.nelx,self.nelz])
        c_toSum = (self.Emin+(self.xPhys**self.penal)*(self.E0-self.Emin))*ce

        c = np.sum(np.sum(np.sum(c_toSum)))
        dc = -self.penal*(self.E0-self.Emin)*(self.xPhys**(self.penal-1))*ce
        return c,dc

    def filterAcrossElements(self,x):
        """
        Performs a weighted average of the given x for all the values within the radius rMin of the element

        Effectivly creates a blur filter over the given x
        Used to reduce checker boarding by somothing out differences in compliance

        It works by making the elements a 1D array(column vector), then when you matrix multiply them you can multiply each element by some coresponding row vector of weights
        This dot product between each element and a weight value is then summed and put as the value of the element
        We then need to divide by theweight values to get the propper average but this is actually done first.
        """
        filter1 = np.reshape(x,(self.nele,1))/self.Hs # reshape x and divide It by Hs(H sum) This creates a pre weighted averageing sum
        filter2 = np.array(self.H@filter1) # Matrix multiply the filter scheme over our elements(H is nele by nele)
        return np.reshape(filter2,[self.nelx,self.nely,self.nelz]) # return the reshaped and filtered 

    def OptimalityCriterion(self,dc,dv,x):
        """
        A bisection algroithm that performs the important step of minimizing compliance

        returns:
            - xnew a 3D matrix of all elements
            - xPhys a filtered matrix of xnew
        """
        #print("Optimality Criterion")
        l1 = 0
        l2 = 1e9
        move = 0.2
        while ((l2-l1)/(l1+l2) > 1e-3):
            lmid = 0.5*(l2+l1)
            B_e = -(dc/dv)/lmid
            """
            The following line of code is confusing be it ultimately boils down to the following peicewise function:

            x_min = max(0,x_e-move) # the new min value of x_e must be either 0 or it's alllocated move distance in the negative direction
            x_max = min(1,x_e+move) # the new max value of x_e must be either 1 or it's alllocated move distance in the positve direction

            if(x_e * sqrt(B_e) <= x_min):
                x_new = x_min

            elif(x_e * sqrt(B_e) >= x_max):
                x_new = x_max

            else:
                x_new = x_e * sqrt(B_e)

            This locks off the values of xnew to only values between 0 and 1
            """
            xnew = np.maximum(0,np.maximum(x-move,np.minimum(1,np.minimum(x+move,x*np.sqrt(B_e)))))

            xPhys_flat = np.array((self.H@np.reshape(xnew,(self.nele,1)))/self.Hs)# filter values of xnew


            # actual bisection algorihm 
            if (np.sum(xPhys_flat) > volfrac*self.nele):
                l1 = lmid 
            else: 
                l2 = lmid


        xPhys = np.reshape(xPhys_flat,(nelx,nely,nelz))
        return xnew,xPhys

    def lk_H8(self):
        """GENERATE ELEMENT STIFFNESS MATRIX"""
        #print("\nBuilding KE")
        A = np.array([[32, 6, -8, 6, -6, 4, 3, -6, -10, 3, -3, -3, -4, -8],
            [-48, 0, 0, -24, 24, 0, 0, 0, 12, -12, 0, 12, 12, 12]])
        #print("A.T.shape:",A.T.shape,"\nnu.shape:",np.array([[1], [self.nu]]).shape)
        k = 1/144*A.T@np.array([[1], [self.nu]])
        #print("k.shape:",k.shape)

        K1 = np.reshape([[k[0], k[1], k[1], k[2], k[4], k[4]],
            [k[1], k[0], k[1], k[3], k[5], k[6]],
            [k[1], k[1], k[0], k[3], k[6], k[5]],
            [k[2], k[3], k[3], k[0], k[7], k[7]],
            [k[4], k[5], k[6], k[7], k[0], k[1]],
            [k[4], k[6], k[5], k[7], k[1], k[0]]],(6,6))
        K2 = np.reshape([[k[8],  k[7],  k[11], k[5],  k[3],  k[6]],
            [k[7],  k[8],  k[11], k[4],  k[2],  k[4]],
            [k[9], k[9], k[12], k[6],  k[3],  k[5]],
            [k[5],  k[4],  k[10], k[8],  k[1],  k[9]],
            [k[3],  k[2],  k[4],  k[1],  k[8],  k[11]],
            [k[10], k[3],  k[5],  k[11], k[9], k[12]]],(6,6))
        K3 = np.reshape([[k[5],  k[6],  k[3],  k[8],  k[11], k[7]],
            [k[6],  k[5],  k[3],  k[9], k[12], k[9]],
            [k[4],  k[4],  k[2],  k[7],  k[11], k[8]],
            [k[8],  k[9], k[1],  k[5],  k[10], k[4]],
            [k[11], k[12], k[9], k[10], k[5],  k[3]],
            [k[1],  k[11], k[8],  k[3],  k[4],  k[2]]],(6,6))
        K4 = np.reshape([[k[13], k[10], k[10], k[12], k[9], k[9]],
            [k[10], k[13], k[10], k[11], k[8],  k[7]],
            [k[10], k[10], k[13], k[11], k[7],  k[8]],
            [k[12], k[11], k[11], k[13], k[6],  k[6]],
            [k[9], k[8],  k[7],  k[6],  k[13], k[10]],
            [k[9], k[7],  k[8],  k[6],  k[10], k[13]]],(6,6))
        K5 = np.reshape([[k[0], k[1],  k[7],  k[2], k[4],  k[3]],
            [k[1], k[0],  k[7],  k[3], k[5],  k[10]],
            [k[7], k[7],  k[0],  k[4], k[10], k[5]],
            [k[2], k[3],  k[4],  k[0], k[7],  k[1]],
            [k[4], k[5],  k[10], k[7], k[0],  k[7]],
            [k[3], k[10], k[5],  k[1], k[7],  k[0]]],(6,6))
        K6 = np.reshape([[k[13], k[10], k[6],  k[12], k[9], k[11]],
            [k[10], k[13], k[6],  k[11], k[8],  k[1]],
            [k[6],  k[6],  k[13], k[9], k[1],  k[8]],
            [k[12], k[11], k[9], k[13], k[6],  k[10]],
            [k[9], k[8],  k[1],  k[6],  k[13], k[6]],
            [k[11], k[1],  k[8],  k[10], k[6],  k[13]]],(6,6))
        #print("K1.shape:",K1.shape)
        #print("K2.shape:",K2.shape)
        #print("K3.shape:",K3.shape)
        #print("K4.shape:",K4.shape)
        #print("K4.shape:",K5.shape)
        #print("K6.shape:",K6.shape)
        
        KE = 1/((self.nu+1)*(1-2*self.nu))* np.block([  [K1,    K2,     K3,     K4],
                                                        [K2.T,  K5,     K6,     K3.T],
                                                        [K3.T,  K6,     K5.T,   K2.T],
                                                        [K4,    K3,     K2,     K1.T]])
        return KE
    
    def getPlotPoints(self,cutoff= 0.5):
        """
        Uses the marching cubes algorithm to create a list of triangle verticies in 3D space that can be used to generate a 3D mesh of the current part

        Parameters:
            - cutoff measures at what density level the part is constricted to(any density above cutoff is included in the part)

        Returns:
            - three lists of points for the x,y,z locations of the verticies of the triangles. This can be inputed directly into the matplotlib plot_trisurf function.

        The marching cubes algorithm generates triangles but doesn't order them like by contecting verticies so the matplotlib part will look weird.
        There exists a pygame representation of this that can be used ot se the full part or you can just look at the csv output.
        """

        squareSize = max(self.nelx,self.nely,self.nelz) + 1
        equationMap = np.pad(self.xPhys,[(1,squareSize-self.nelx),(1,squareSize-self.nely),(1,squareSize-self.nelz)],mode='constant',constant_values=0)

        """
        from: http://paulbourke.net/geometry/polygonise/

        a cube is defined as a list of verticies and edges

        for each cube(set of 8 verticies) in our equation map create a cupe index by setting a bit for each vertex inside the polygon
        once each vetex has been identified find out what edges need to have a point on them(by lookup table)

        then for each set of three edges from the lookup table go and create a triangle from the midpoints of each line
        """
        TRI_TABLE = [[],
                [[0,8,3]],
                [[0,1,9]],
                [[1,8,3],[9,8,1]],
                [[1,2,10]],
                [[0,8,3],[1,2,10]],
                [[9,2,10],[0,2,9]],
                [[2,8,3],[2,10,8],[10,9,8]],
                [[3,11,2]],
                [[0,11,2],[8,11,0]],
                [[1,9,0],[2,3,11]],
                [[1,11,2],[1,9,11],[9,8,11]],
                [[3,10,1],[11,10,3]],
                [[0,10,1],[0,8,10],[8,11,10]],
                [[3,9,0],[3,11,9],[11,10,9]],
                [[9,8,10],[10,8,11]],
                [[4,7,8]],
                [[4,3,0],[7,3,4]],
                [[0,1,9],[8,4,7]],
                [[4,1,9],[4,7,1],[7,3,1]],
                [[1,2,10],[8,4,7]],
                [[3,4,7],[3,0,4],[1,2,10]],
                [[9,2,10],[9,0,2],[8,4,7]],
                [[2,10,9],[2,9,7],[2,7,3],[7,9,4]],
                [[8,4,7],[3,11,2]],
                [[11,4,7],[11,2,4],[2,0,4]],
                [[9,0,1],[8,4,7],[2,3,11]],
                [[4,7,11],[9,4,11],[9,11,2],[9,2,1]],
                [[3,10,1],[3,11,10],[7,8,4]],
                [[1,11,10],[1,4,11],[1,0,4],[7,11,4]],
                [[4,7,8],[9,0,11],[9,11,10],[11,0,3]],
                [[4,7,11],[4,11,9],[9,11,10]],
                [[9,5,4]],
                [[9,5,4],[0,8,3]],
                [[0,5,4],[1,5,0]],
                [[8,5,4],[8,3,5],[3,1,5]],
                [[1,2,10],[9,5,4]],
                [[3,0,8],[1,2,10],[4,9,5]],
                [[5,2,10],[5,4,2],[4,0,2]],
                [[2,10,5],[3,2,5],[3,5,4],[3,4,8]],
                [[9,5,4],[2,3,11]],
                [[0,11,2],[0,8,11],[4,9,5]],
                [[0,5,4],[0,1,5],[2,3,11]],
                [[2,1,5],[2,5,8],[2,8,11],[4,8,5]],
                [[10,3,11],[10,1,3],[9,5,4]],
                [[4,9,5],[0,8,1],[8,10,1],[8,11,10]],
                [[5,4,0],[5,0,11],[5,11,10],[11,0,3]],
                [[5,4,8],[5,8,10],[10,8,11]],
                [[9,7,8],[5,7,9]],
                [[9,3,0],[9,5,3],[5,7,3]],
                [[0,7,8],[0,1,7],[1,5,7]],
                [[1,5,3],[3,5,7]],
                [[9,7,8],[9,5,7],[10,1,2]],
                [[10,1,2],[9,5,0],[5,3,0],[5,7,3]],
                [[8,0,2],[8,2,5],[8,5,7],[10,5,2]],
                [[2,10,5],[2,5,3],[3,5,7]],
                [[7,9,5],[7,8,9],[3,11,2]],
                [[9,5,7],[9,7,2],[9,2,0],[2,7,11]],
                [[2,3,11],[0,1,8],[1,7,8],[1,5,7]],
                [[11,2,1],[11,1,7],[7,1,5]],
                [[9,5,8],[8,5,7],[10,1,3],[10,3,11]],
                [[5,7,0],[5,0,9],[7,11,0],[1,0,10],[11,10,0]],
                [[11,10,0],[11,0,3],[10,5,0],[8,0,7],[5,7,0]],
                [[11,10,5],[7,11,5]],
                [[10,6,5]],
                [[0,8,3],[5,10,6]],
                [[9,0,1],[5,10,6]],
                [[1,8,3],[1,9,8],[5,10,6]],
                [[1,6,5],[2,6,1]],
                [[1,6,5],[1,2,6],[3,0,8]],
                [[9,6,5],[9,0,6],[0,2,6]],
                [[5,9,8],[5,8,2],[5,2,6],[3,2,8]],
                [[2,3,11],[10,6,5]],
                [[11,0,8],[11,2,0],[10,6,5]],
                [[0,1,9],[2,3,11],[5,10,6]],
                [[5,10,6],[1,9,2],[9,11,2],[9,8,11]],
                [[6,3,11],[6,5,3],[5,1,3]],
                [[0,8,11],[0,11,5],[0,5,1],[5,11,6]],
                [[3,11,6],[0,3,6],[0,6,5],[0,5,9]],
                [[6,5,9],[6,9,11],[11,9,8]],
                [[5,10,6],[4,7,8]],
                [[4,3,0],[4,7,3],[6,5,10]],
                [[1,9,0],[5,10,6],[8,4,7]],
                [[10,6,5],[1,9,7],[1,7,3],[7,9,4]],
                [[6,1,2],[6,5,1],[4,7,8]],
                [[1,2,5],[5,2,6],[3,0,4],[3,4,7]],
                [[8,4,7],[9,0,5],[0,6,5],[0,2,6]],
                [[7,3,9],[7,9,4],[3,2,9],[5,9,6],[2,6,9]],
                [[3,11,2],[7,8,4],[10,6,5]],
                [[5,10,6],[4,7,2],[4,2,0],[2,7,11]],
                [[0,1,9],[4,7,8],[2,3,11],[5,10,6]],
                [[9,2,1],[9,11,2],[9,4,11],[7,11,4],[5,10,6]],
                [[8,4,7],[3,11,5],[3,5,1],[5,11,6]],
                [[5,1,11],[5,11,6],[1,0,11],[7,11,4],[0,4,11]],
                [[0,5,9],[0,6,5],[0,3,6],[11,6,3],[8,4,7]],
                [[6,5,9],[6,9,11],[4,7,9],[7,11,9]],
                [[10,4,9],[6,4,10]],
                [[4,10,6],[4,9,10],[0,8,3]],
                [[10,0,1],[10,6,0],[6,4,0]],
                [[8,3,1],[8,1,6],[8,6,4],[6,1,10]],
                [[1,4,9],[1,2,4],[2,6,4]],
                [[3,0,8],[1,2,9],[2,4,9],[2,6,4]],
                [[0,2,4],[4,2,6]],
                [[8,3,2],[8,2,4],[4,2,6]],
                [[10,4,9],[10,6,4],[11,2,3]],
                [[0,8,2],[2,8,11],[4,9,10],[4,10,6]],
                [[3,11,2],[0,1,6],[0,6,4],[6,1,10]],
                [[6,4,1],[6,1,10],[4,8,1],[2,1,11],[8,11,1]],
                [[9,6,4],[9,3,6],[9,1,3],[11,6,3]],
                [[8,11,1],[8,1,0],[11,6,1],[9,1,4],[6,4,1]],
                [[3,11,6],[3,6,0],[0,6,4]],
                [[6,4,8],[11,6,8]],
                [[7,10,6],[7,8,10],[8,9,10]],
                [[0,7,3],[0,10,7],[0,9,10],[6,7,10]],
                [[10,6,7],[1,10,7],[1,7,8],[1,8,0]],
                [[10,6,7],[10,7,1],[1,7,3]],
                [[1,2,6],[1,6,8],[1,8,9],[8,6,7]],
                [[2,6,9],[2,9,1],[6,7,9],[0,9,3],[7,3,9]],
                [[7,8,0],[7,0,6],[6,0,2]],
                [[7,3,2],[6,7,2]],
                [[2,3,11],[10,6,8],[10,8,9],[8,6,7]],
                [[2,0,7],[2,7,11],[0,9,7],[6,7,10],[9,10,7]],
                [[1,8,0],[1,7,8],[1,10,7],[6,7,10],[2,3,11]],
                [[11,2,1],[11,1,7],[10,6,1],[6,7,1]],
                [[8,9,6],[8,6,7],[9,1,6],[11,6,3],[1,3,6]],
                [[0,9,1],[11,6,7]],
                [[7,8,0],[7,0,6],[3,11,0],[11,6,0]],
                [[7,11,6]],
                [[7,6,11]],
                [[3,0,8],[11,7,6]],
                [[0,1,9],[11,7,6]],
                [[8,1,9],[8,3,1],[11,7,6]],
                [[10,1,2],[6,11,7]],
                [[1,2,10],[3,0,8],[6,11,7]],
                [[2,9,0],[2,10,9],[6,11,7]],
                [[6,11,7],[2,10,3],[10,8,3],[10,9,8]],
                [[7,2,3],[6,2,7]],
                [[7,0,8],[7,6,0],[6,2,0]],
                [[2,7,6],[2,3,7],[0,1,9]],
                [[1,6,2],[1,8,6],[1,9,8],[8,7,6]],
                [[10,7,6],[10,1,7],[1,3,7]],
                [[10,7,6],[1,7,10],[1,8,7],[1,0,8]],
                [[0,3,7],[0,7,10],[0,10,9],[6,10,7]],
                [[7,6,10],[7,10,8],[8,10,9]],
                [[6,8,4],[11,8,6]],
                [[3,6,11],[3,0,6],[0,4,6]],
                [[8,6,11],[8,4,6],[9,0,1]],
                [[9,4,6],[9,6,3],[9,3,1],[11,3,6]],
                [[6,8,4],[6,11,8],[2,10,1]],
                [[1,2,10],[3,0,11],[0,6,11],[0,4,6]],
                [[4,11,8],[4,6,11],[0,2,9],[2,10,9]],
                [[10,9,3],[10,3,2],[9,4,3],[11,3,6],[4,6,3]],
                [[8,2,3],[8,4,2],[4,6,2]],
                [[0,4,2],[4,6,2]],
                [[1,9,0],[2,3,4],[2,4,6],[4,3,8]],
                [[1,9,4],[1,4,2],[2,4,6]],
                [[8,1,3],[8,6,1],[8,4,6],[6,10,1]],
                [[10,1,0],[10,0,6],[6,0,4]],
                [[4,6,3],[4,3,8],[6,10,3],[0,3,9],[10,9,3]],
                [[10,9,4],[6,10,4]],
                [[4,9,5],[7,6,11]],
                [[0,8,3],[4,9,5],[11,7,6]],
                [[5,0,1],[5,4,0],[7,6,11]],
                [[11,7,6],[8,3,4],[3,5,4],[3,1,5]],
                [[9,5,4],[10,1,2],[7,6,11]],
                [[6,11,7],[1,2,10],[0,8,3],[4,9,5]],
                [[7,6,11],[5,4,10],[4,2,10],[4,0,2]],
                [[3,4,8],[3,5,4],[3,2,5],[10,5,2],[11,7,6]],
                [[7,2,3],[7,6,2],[5,4,9]],
                [[9,5,4],[0,8,6],[0,6,2],[6,8,7]],
                [[3,6,2],[3,7,6],[1,5,0],[5,4,0]],
                [[6,2,8],[6,8,7],[2,1,8],[4,8,5],[1,5,8]],
                [[9,5,4],[10,1,6],[1,7,6],[1,3,7]],
                [[1,6,10],[1,7,6],[1,0,7],[8,7,0],[9,5,4]],
                [[4,0,10],[4,10,5],[0,3,10],[6,10,7],[3,7,10]],
                [[7,6,10],[7,10,8],[5,4,10],[4,8,10]],
                [[6,9,5],[6,11,9],[11,8,9]],
                [[3,6,11],[0,6,3],[0,5,6],[0,9,5]],
                [[0,11,8],[0,5,11],[0,1,5],[5,6,11]],
                [[6,11,3],[6,3,5],[5,3,1]],
                [[1,2,10],[9,5,11],[9,11,8],[11,5,6]],
                [[0,11,3],[0,6,11],[0,9,6],[5,6,9],[1,2,10]],
                [[11,8,5],[11,5,6],[8,0,5],[10,5,2],[0,2,5]],
                [[6,11,3],[6,3,5],[2,10,3],[10,5,3]],
                [[5,8,9],[5,2,8],[5,6,2],[3,8,2]],
                [[9,5,6],[9,6,0],[0,6,2]],
                [[1,5,8],[1,8,0],[5,6,8],[3,8,2],[6,2,8]],
                [[1,5,6],[2,1,6]],
                [[1,3,6],[1,6,10],[3,8,6],[5,6,9],[8,9,6]],
                [[10,1,0],[10,0,6],[9,5,0],[5,6,0]],
                [[0,3,8],[5,6,10]],
                [[10,5,6]],
                [[11,5,10],[7,5,11]],
                [[11,5,10],[11,7,5],[8,3,0]],
                [[5,11,7],[5,10,11],[1,9,0]],
                [[10,7,5],[10,11,7],[9,8,1],[8,3,1]],
                [[11,1,2],[11,7,1],[7,5,1]],
                [[0,8,3],[1,2,7],[1,7,5],[7,2,11]],
                [[9,7,5],[9,2,7],[9,0,2],[2,11,7]],
                [[7,5,2],[7,2,11],[5,9,2],[3,2,8],[9,8,2]],
                [[2,5,10],[2,3,5],[3,7,5]],
                [[8,2,0],[8,5,2],[8,7,5],[10,2,5]],
                [[9,0,1],[5,10,3],[5,3,7],[3,10,2]],
                [[9,8,2],[9,2,1],[8,7,2],[10,2,5],[7,5,2]],
                [[1,3,5],[3,7,5]],
                [[0,8,7],[0,7,1],[1,7,5]],
                [[9,0,3],[9,3,5],[5,3,7]],
                [[9,8,7],[5,9,7]],
                [[5,8,4],[5,10,8],[10,11,8]],
                [[5,0,4],[5,11,0],[5,10,11],[11,3,0]],
                [[0,1,9],[8,4,10],[8,10,11],[10,4,5]],
                [[10,11,4],[10,4,5],[11,3,4],[9,4,1],[3,1,4]],
                [[2,5,1],[2,8,5],[2,11,8],[4,5,8]],
                [[0,4,11],[0,11,3],[4,5,11],[2,11,1],[5,1,11]],
                [[0,2,5],[0,5,9],[2,11,5],[4,5,8],[11,8,5]],
                [[9,4,5],[2,11,3]],
                [[2,5,10],[3,5,2],[3,4,5],[3,8,4]],
                [[5,10,2],[5,2,4],[4,2,0]],
                [[3,10,2],[3,5,10],[3,8,5],[4,5,8],[0,1,9]],
                [[5,10,2],[5,2,4],[1,9,2],[9,4,2]],
                [[8,4,5],[8,5,3],[3,5,1]],
                [[0,4,5],[1,0,5]],
                [[8,4,5],[8,5,3],[9,0,5],[0,3,5]],
                [[9,4,5]],
                [[4,11,7],[4,9,11],[9,10,11]],
                [[0,8,3],[4,9,7],[9,11,7],[9,10,11]],
                [[1,10,11],[1,11,4],[1,4,0],[7,4,11]],
                [[3,1,4],[3,4,8],[1,10,4],[7,4,11],[10,11,4]],
                [[4,11,7],[9,11,4],[9,2,11],[9,1,2]],
                [[9,7,4],[9,11,7],[9,1,11],[2,11,1],[0,8,3]],
                [[11,7,4],[11,4,2],[2,4,0]],
                [[11,7,4],[11,4,2],[8,3,4],[3,2,4]],
                [[2,9,10],[2,7,9],[2,3,7],[7,4,9]],
                [[9,10,7],[9,7,4],[10,2,7],[8,7,0],[2,0,7]],
                [[3,7,10],[3,10,2],[7,4,10],[1,10,0],[4,0,10]],
                [[1,10,2],[8,7,4]],
                [[4,9,1],[4,1,7],[7,1,3]],
                [[4,9,1],[4,1,7],[0,8,1],[8,7,1]],
                [[4,0,3],[7,4,3]],
                [[4,8,7]],
                [[9,10,8],[10,11,8]],
                [[3,0,9],[3,9,11],[11,9,10]],
                [[0,1,10],[0,10,8],[8,10,11]],
                [[3,1,10],[11,3,10]],
                [[1,2,11],[1,11,9],[9,11,8]],
                [[3,0,9],[3,9,11],[1,2,9],[2,11,9]],
                [[0,2,11],[8,0,11]],
                [[3,2,11]],
                [[2,3,8],[2,8,10],[10,8,9]],
                [[9,10,2],[0,9,2]],
                [[2,3,8],[2,8,10],[0,1,8],[1,10,8]],
                [[1,10,2]],
                [[1,3,8],[9,1,8]],
                [[0,9,1]],
                [[0,3,8]],
                []]

        EDGE_LOOKUP = [
            [0,1,0],
            [1,2,1],
            [2,3,0],
            [3,0,1],
            [4,5,0],
            [5,6,1],
            [6,7,0],
            [7,4,1],
            [0,4,2],
            [1,5,2],
            [2,6,2],
            [3,7,2],
        ]

        def marchingCubes_getPointOnEdge(edge,coordsOfPoints,isoValues,interpolate = True):
            """
            calculates the coordinate of the point for the given edge
            returns a point between the vertieces of the given edge
            """

            verticies = EDGE_LOOKUP[edge]
            p1 = coordsOfPoints[verticies[0]]
            p2 = coordsOfPoints[verticies[1]]
            isoCoord = verticies[2]

            #print(p1,p2)

            #average of the two points
            coord =[p1[0], p1[1], p1[2]]
            mu = (cutoff - isoValues[verticies[0]]) / (isoValues[verticies[1]] - isoValues[verticies[0]])
            if(interpolate):
                coord[isoCoord] = p1[isoCoord] + mu * (p2[isoCoord] - p1[isoCoord])

            
            return coord

        x_Points = []
        y_Points = []
        z_Points = []
        for x in range(squareSize):
            for y in range(squareSize):
                for z in range(squareSize):
                    # a cube is 8 points
                    gridval = [ equationMap[x,y,z],equationMap[x+1,y,z],equationMap[x+1,y+1,z],equationMap[x,y+1,z],
                                equationMap[x,y,z+1],equationMap[x+1,y,z+1],equationMap[x+1,y+1,z+1],equationMap[x,y+1,z+1]]
                    #get the cube index for use in the lookup table
                    cubeIndex = 0
                    #print("[{},{},{}]:\n\t{}".format(x,y,z,gridval))

                    if (gridval[0] > cutoff):
                        cubeIndex |= 1
                    if (gridval[1] > cutoff):
                        cubeIndex |= 2
                    if (gridval[2] > cutoff):
                        cubeIndex |= 4
                    if (gridval[3] > cutoff):
                        cubeIndex |= 8
                    if (gridval[4] > cutoff):
                        cubeIndex |= 16
                    if (gridval[5] > cutoff):
                        cubeIndex |= 32
                    if (gridval[6] > cutoff):
                        cubeIndex |= 64
                    if (gridval[7] > cutoff):
                        cubeIndex |= 128
                    #print(cubeIndex)
                    #get the coords of the points in the cube
                    if(len(TRI_TABLE[cubeIndex]) > 0):
                        x1 = x
                        y1 = y
                        z1 = z
                        x2 = x+1
                        y2 = y+1
                        z2 = z+1
                        coordsList = [  [x1,y1,z1],[x2,y1,z1],[x2,y2,z1],[x1,y2,z1],
                                        [x1,y1,z2],[x2,y1,z2],[x2,y2,z2],[x1,y2,z2]]
                        #print(coordsList)
                        for edgeList in TRI_TABLE[cubeIndex]:
                            p1 = marchingCubes_getPointOnEdge(edgeList[0],coordsList,gridval)
                            p2 = marchingCubes_getPointOnEdge(edgeList[1],coordsList,gridval)
                            p3 = marchingCubes_getPointOnEdge(edgeList[2],coordsList,gridval)

                            x_Points.append(p1[0])
                            x_Points.append(p2[0])
                            x_Points.append(p3[0])

                            y_Points.append(p1[1])
                            y_Points.append(p2[1])
                            y_Points.append(p3[1])

                            z_Points.append(p1[2])
                            z_Points.append(p2[2])
                            z_Points.append(p3[2])

        return x_Points,y_Points,z_Points

    def generateElementDegreesOfFreedomMatrix(self):
        """
        Generates the Element degrees of freedom conectivity matrix as described in the paper

        Due to lack of study material there is a posibility that the for loop for the y elements is out of order. The order may not matter, however that for loop could go before the z loop(unlikely) or before the x loop(more likely)
        I have the setup as y x z becuse that is how the filterning matrix is built and I think this should be the same way
        If errors occur check the order of the loops(should be fine)
        """
        

        def generateNodeIds(x1,y1,z1):
            """Get the degrees of freedom for an element based off it's xyz coordinate"""
            nodeIDs = np.zeros(24)
            NIDz = (self.nelx+1)*(self.nely+1)
            NID1 = z1*(self.nelx+1)*(self.nely+1) + x1*(self.nely+1) + (self.nely+1 - y1)
            NID2 = NID1 + (self.nely+1)
            NID3 = NID1 + self.nely
            NID4 = NID1-1
            NID5 = NID1 + NIDz
            NID6 = NID2 + NIDz
            NID7 = NID3 + NIDz
            NID8 = NID4 + NIDz
            listOfNodeIDs = [NID1,NID2,NID3,NID4,NID5,NID6,NID7,NID8]
            for i,nid in enumerate(listOfNodeIDs):
                nodeIDs[3*i] = 3*nid-2 -1
                nodeIDs[3*i+1] = 3*nid-1 -1
                nodeIDs[3*i+2] = 3*nid -1
            return nodeIDs


        edofmat = np.zeros((self.nele,24))
        currentNode = 0
        for z1 in range(nelz): # z loop
            for x1 in range(nelx):# x loop
                for y1 in range(nely): # y loop (may be misplaced)
                    edofmat[currentNode,:] = generateNodeIds(x1,y1,z1)
                    currentNode += 1

        #print("\nEdofmat Min and Max values:",np.max(edofmat),np.min(edofmat))

        return edofmat.astype("int32")

    def saveToCSV(self,filePath):
        np.savetxt(filePath, np.reshape(self.xPhys,(self.nele,1)), delimiter=",")
    
    def getFilteredDensityGrid(self):
        """
        Returns the current xPhys which is a density grid of the part.
        Values close to 1 represent full density or full material.
        Values close to 0 represent no density or air.

        This has been filtered to avoid the checkerboarding problem
        """
        return self.xPhys

    def getRawDensityGrid(self):
        """
        Returns the current x which is a density grid of the part.
        Values close to 1 represent full density or full material.
        Values close to 0 represent no density or air.

        This has NOT been filtered
        """
        return self.x

    def ApplyForcesAsSparce(self,values,i_coord):
        """
        Creates the force vector for the part
        Builds the vector as a 1D sparce matrix of size ndof = 3*(nelx+1)*(nely+1)*(nelz+1)

        Takes an array containing the values of the forces as well as and array of the index of the values
        len(values) must equal len(i_coord)

            - values is the array of force magnitudes applied to the part
            - i_coord is the array of the indexes for each coresponding index in values

        """
        F = coo_matrix((values,(i_coord,np.zeros(len(i_coord)))),shape=(self.ndof,1)).tocsc()#reshape must be changed
        return F
# =========================================================================
# === This code was written by K Liu and A Tovar, Dept. of Mechanical   ===
# === Engineering, Indiana University-Purdue University Indianapolis,   ===
# === Indiana, United States of America                                 ===
# === ----------------------------------------------------------------- ===
# === Please send your suggestions and comments to: kailiu@iupui.edu    ===
# === ----------------------------------------------------------------- ===
# === The code is intended for educational purposes, and the details    ===
# === and extensions can be found in the paper:                         ===
# === K. Liu and A. Tovar, "An efficient 3D topology optimization code  ===
# === written in Matlab", Struct Multidisc Optim, 50(6): 1175-1196, 2014, =
# === doi:10.1007/s00158-014-1107-x                                     ===
# === ----------------------------------------------------------------- ===
# === The code as well as an uncorrected version of the paper can be    ===
# === downloaded from the website: http://www.top3dapp.com/             ===
# === ----------------------------------------------------------------- ===
# === Disclaimer:                                                       ===
# === The authors reserves all rights for the program.                  ===
# === The code may be distributed and used for educational purposes.    ===
# === The authors do not guarantee that the code is free from errors, a

# === This code has be rewritten into python


# The real main driver    
if __name__ == "__main__":
    nelx=10
    nely=10
    nelz = 10
    volfrac=0.4
    rmin=5.4
    penal=3.0

    t=TopOpt3D(nelx,nely,nelz,volfrac,penal,rmin)

    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    tri = ax.plot_trisurf([0,0.1,0.1], [0,0.1,0], [0,0,0.1], cmap='viridis', linewidths=0.2)
    loop = 0
    output = False # control on outputting a csv file

    still_itterating = True
    while(plt.fignum_exists(fig.number)):
        if(still_itterating):
            still_itterating = t.itterationStep()
        
        x_plot,y_plot,z_plot = t.getPlotPoints()
        if(len(x_plot) >= 3):
            tri.remove()
            tri = ax.plot_trisurf(x_plot, y_plot, z_plot, cmap='viridis', linewidths=0.2)
            #Despite my attempts, if you exit the plot while it is still working you will get an error
            if(plt.fignum_exists(fig.number)):
                fig.canvas.flush_events()
                fig.canvas.draw()

        if(output):
            loop += 1
            if(loop %2 == 0):
                if(loop%4==0):
                    t.saveToCSV(r".\3D Variant\output3D2.csv")
                else:
                    t.saveToCSV(r".\3D Variant\output3D1.csv")
    
    #t.saveToCSV(r".\3D Variant\output3D.csv")
    print("Done.")