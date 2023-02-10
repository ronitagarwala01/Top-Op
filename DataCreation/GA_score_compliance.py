import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from scipy.optimize import Bounds,LinearConstraint,NonlinearConstraint
from scipy.optimize import linprog

from tensorflow.keras.layers import AveragePooling2D


class topOpter:
    def __init__(self,nelx,nely,volfrac,penal,rmin,ft,complianceConstraint):
        self.nelx =nelx
        self.nely = nely
        self.volfrac = volfrac
        self.penal = penal
        self.rmin = rmin
        self.ft = ft
        self.massPenaltyFactor = 0
        self.complianceMax = complianceConstraint
        self.minChange = 0.01
        self.maxElementChange = .1
        

        print("Minimum compliance problem with OC")
        print("ndes: " + str(nelx) + " x " + str(nely))
        print("volfrac: " + str(volfrac) + ", rmin: " + str(rmin) + ", penal: " + str(penal))
        print("Filter method: " + ["Sensitivity based","Density based"][ft])
        
        # Max and min stiffness
        self.Emin=1e-9
        self.Emax=1.0
        # dofs:
        self.ndof = 2*(nelx+1)*(nely+1)

        # Allocate design variables (as array), initialize and allocate sens.
        self.x=np.ones(nely*nelx,dtype=float)
        self.xold=self.x.copy()
        self.xPhys=self.x.copy()

        self.g=0 # must be initialized to use the NGuyen/Paulino OC approach
        self.dc=np.zeros((nely,nelx), dtype=float)

        # FE: Build the index vectors for the for coo matrix format.
        self.KE=self.lk()
        self.edofMat=np.zeros((nelx*nely,8),dtype=int)
        for elx in range(nelx):
            for ely in range(nely):
                el = ely+elx*nely
                n1=(nely+1)*elx+ely
                n2=(nely+1)*(elx+1)+ely
                self.edofMat[el,:]=np.array([2*n1+2, 2*n1+3, 2*n2+2, 2*n2+3,2*n2, 2*n2+1, 2*n1, 2*n1+1])

        # Construct the index pointers for the coo format
        self.iK = np.kron(self.edofMat,np.ones((8,1))).flatten()
        self.jK = np.kron(self.edofMat,np.ones((1,8))).flatten()    

        # Filter: Build (and assemble) the index+data vectors for the coo matrix format
        nfilter=int(nelx*nely*((2*(np.ceil(rmin)-1)+1)**2))
        self.iH = np.zeros(nfilter)
        self.jH = np.zeros(nfilter)
        self.sH = np.zeros(nfilter)
        cc=0
        for i in range(nelx):
            for j in range(nely):
                row=i*nely+j
                kk1=int(np.maximum(i-(np.ceil(rmin)-1),0))
                kk2=int(np.minimum(i+np.ceil(rmin),nelx))
                ll1=int(np.maximum(j-(np.ceil(rmin)-1),0))
                ll2=int(np.minimum(j+np.ceil(rmin),nely))
                for k in range(kk1,kk2):
                    for l in range(ll1,ll2):
                        col=k*nely+l
                        fac=rmin-np.sqrt(((i-k)*(i-k)+(j-l)*(j-l)))
                        self.iH[cc]=row
                        self.jH[cc]=col
                        self.sH[cc]=np.maximum(0.0,fac)
                        cc=cc+1
        # Finalize assembly and convert to csc format
        self.H=coo_matrix((self.sH,(self.iH,self.jH)),shape=(nelx*nely,nelx*nely)).tocsc()    
        self.Hs=self.H.sum(1)
        def IX(x,y):
            return y*nely + x
        # BC's and support
        dofs=np.arange(2*(nelx+1)*(nely+1))
        #fixed=np.union1d(dofs[0:2*(5):2],np.array([2*(nelx+1)*(nely+1)-1]))
        #fixed = np.array([0,1,3,4])
        fixed = dofs[0:2*(nely-1):1]
        self.free=np.setdiff1d(dofs,fixed)
        self.numberOfFixedPoints = len(fixed)

        # Solution and RHS vectors
        # for n number of forces each of these must be and n-dimesional column vector
        self.numberOfForces = 2
        self.f=np.zeros((self.ndof,self.numberOfForces))
        self.u=np.zeros((self.ndof,self.numberOfForces))

        # Set load
        #self.f[0,0] = -1
        

    

        #passive elements
        self.passive = np.zeros((nely) * (nelx))


        # Set loop counter and gradient vectors 
        self.loop=0
        self.change=1
        self.objective_mass = 0
        self.constraint_compliance = 0
        self.dv = np.ones(nely*nelx)
        self.dc = np.ones(nely*nelx)
        self.ce = np.ones(nely*nelx)

    def sensitivityAnalysis(self,x):
        #reshape x into a 1D array to perform calculations
        x = np.reshape(x,(self.nelx*self.nely))
        x = np.maximum(0.001,np.minimum(x,1))
        #Apply the force fill and force free areas
        x = np.where(self.passive == 1,0,x)
        x = np.where(self.passive == 2,1,x)

        # Setup and solve FE problem
        sK=((self.KE.flatten()[np.newaxis]).T*(self.Emin+(x)**self.penal*(self.Emax-self.Emin))).flatten(order='F')
        K_unconstrained = coo_matrix((sK,(self.iK,self.jK)),shape=(self.ndof,self.ndof)).tocsc()

        # Remove constrained dofs from matrix
        K = K_unconstrained[self.free,:][:,self.free]

        # Solve system 
        u=np.zeros((self.ndof,self.numberOfForces))
        f=self.f[self.free,:]
        sps = spsolve(K,f)
        u[self.free,:] = sps
         
        ce = np.ones(self.nely*self.nelx)
        dc = np.zeros(self.nely*self.nelx)
        compliance = 0
        for i in range(self.numberOfForces):
            Ui = u[:,i]
            ce = (np.dot(Ui[self.edofMat].reshape(self.nelx*self.nely,8),self.KE) * Ui[self.edofMat].reshape(self.nelx*self.nely,8) ).sum(1)
            compliance += ((self.Emin+x**self.penal*(self.Emax-self.Emin))*ce).sum()
            if(i == 0):
                dc[:]=(-self.penal*x**(self.penal-1)*(self.Emax-self.Emin))*ce
            else:
                dc[:]= dc[:] + (-self.penal*x**(self.penal-1)*(self.Emax-self.Emin))*ce

        # Sensitivity filtering:
        self.dc[:] = np.asarray((self.H*(self.x*self.dc))[np.newaxis].T/self.Hs)[:,0] / np.maximum(0.001,self.x)
        
        return compliance,dc,K_unconstrained,K,sps,u[self.free]

    def getCompliance(self,x):
        return self.sensitivityAnalysis(x)[0]

    def Jacobian_Compliance(self,x):
        """
        Takes in a numpy array representing the elements in the current part
        returns the jacobian of the complinace function as a numpy array where each index coresponds to the gradient of the compliance function with resepect to that index of x
        
        Work for the function is done in the senstvity analysis but only the derivative of the compliance is passed outside the function
        """
        return self.sensitivityAnalysis(x)[1]

    def lk(self):
        #element stiffness matrix
        E=1
        nu=0.3
        k=np.array([1/2-nu/6,1/8+nu/8,-1/4-nu/12,-1/8+3*nu/8,-1/4+nu/12,-1/8-nu/8,nu/6,1/8-3*nu/8])
        KE = E/(1-nu**2)*np.array([ [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
        [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
        [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
        [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
        [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
        [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
        [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
        [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]] ])
        return (KE)

    def updatePassives(self,passiveArray):
        """
        the passive array is a nelx by nely 2D array with either 0,1,2 denoting free, force empty, or force fill

        the self.passive array is 1D of size nelx * nely with the same constraints
        """

        for x in range(self.nelx):
            for y in range(self.nely):
                self.passive[x*self.nely + y] = passiveArray[x][y]
        
        self.change = 1 
        self.loop = 0

    def updateLoads(self,loads):

        self.passive = np.zeros((self.nely) * (self.nelx))
        def IX(x,y):
            y*self.nely + x

        for i in range(self.nelx):
            for j in range(self.nely):
                if np.sqrt((j-loads[0][0])**2+(i-loads[0][1])**2) < loads[0][2]:
                    self.passive[IX(i,j)] = 1
                if np.sqrt((j-loads[1][0])**2+(i-loads[0][1])**2) < loads[0][2]:
                    self.passive[IX(i,j)] = 1
                if np.sqrt((j-loads[2][0])**2+(i-loads[0][1])**2) < loads[0][2]:
                    self.passive[IX(i,j)] = 1
        
    def updateFixed(self,fixedArray):
        """
        the fixedArray is a nelx by nely 2D array with either 0,1,2,3 denoting free, fixed horizontal, fixed vertical, or fixed both
        the self.fixed array is 1D of size 2*(nelx+1) * (nely+1)

        required:
            method to transloat element number into its 4 corners
                - in a 4x3 part input 0 should return 0,1,2,3,8,9,10,11
                - in a 4x3 part input 7 should return 18,19,20,21,26,27,28,29
            method to translate the 4 corners into their x and y components
                - horizontal is even 
                - vertical is odd
        """
        def getCorners(x,y):
            elementNum = (self.nely+1)*x + y
            cornerTL = [2*elementNum,2*elementNum+1]
            cornerBL = [2*elementNum+2,2*elementNum+3]
            element1over = (self.nely+1)*(x+1) + y
            cornerTR = [2*element1over,2*element1over+1]
            cornerBR = [2*element1over+2,2*element1over+3]

            return cornerTL,cornerTR,cornerBL,cornerBR

        
        dofs=np.arange(2*(self.nelx+1)*(self.nely+1))
        fixed = []


        for x in range(self.nelx):
            for y in range(self.nely):
                cornerTL,cornerTR,cornerBL,cornerBR = getCorners(x,y)
                if(fixedArray[x][y] == 1):
                    fixed.append(cornerTL[0])
                    fixed.append(cornerBL[0])
                    fixed.append(cornerTR[0])
                    fixed.append(cornerBR[0])
                elif(fixedArray[x][y] == 2):
                    fixed.append(cornerTL[1])
                    fixed.append(cornerBL[1])
                    fixed.append(cornerTR[1])
                    fixed.append(cornerBR[1])
                elif(fixedArray[x][y] == 3):
                    fixed.append(cornerTL[0])
                    fixed.append(cornerBL[0])
                    fixed.append(cornerTR[0])
                    fixed.append(cornerBR[0])
                    fixed.append(cornerTL[1])
                    fixed.append(cornerBL[1])
                    fixed.append(cornerTR[1])
                    fixed.append(cornerBR[1])

        self.numberOfFixedPoints = len(fixed)
        if(self.numberOfFixedPoints > 0):
            self.free=np.setdiff1d(dofs,np.array(fixed))
        else:
            self.free = dofs
        

        self.change = 1 
        self.loop = 0

    def updateForceVectors(self,vectorArray):

        def getCorners(x,y):
            elementNum = (self.nely+1)*x + y
            cornerTL = [2*elementNum,2*elementNum+1]

            return cornerTL

        forces =[]
        for x,y,vx,vy in vectorArray:
            if(vx == 0):
                #force in y direction
                cornerTL = getCorners(x,y)
                force = vy/4
                forces.append([cornerTL[1],force])
            elif(vy == 0):
                #force in x direction
                cornerTL = getCorners(x,y)
                force = vx/4
                forces.append([cornerTL[0],force])
        
        self.numberOfForces = len(forces)
        if(self.numberOfForces > 1):
            self.f=np.zeros((self.ndof,self.numberOfForces))
            self.u=np.zeros((self.ndof,self.numberOfForces))
        if(self.numberOfForces == 1):
            self.f=np.zeros((self.ndof,2))
            self.u=np.zeros((self.ndof,2))

        for i,vec in enumerate(forces):
            elementNumber = vec[0]
            force = vec[1]
            self.f[elementNumber,i] = force


        self.change = 1 
        self.loop = 0

    def ApplyProblem(self, filledArea,supportArea,forceVector):
        self.passive = (2*filledArea).reshape(self.nelx*self.nely)
        self.updateFixed(supportArea*3)
        self.f = forceVector
        self.numberOfForces = 4
        self.u=np.zeros((2*(self.nelx+1)*(self.nely+1),4))

    def applyConstraints(self,x):
        #reshape x into a 1D array to perform calculations
        x = np.reshape(x,(self.nelx*self.nely))
        x = np.maximum(0,np.minimum(x,1))
        #Apply the force fill and force free areas
        x = np.where(self.passive == 1,0,x)
        x = np.where(self.passive == 2,1,x)
        x = np.reshape(x,(self.nelx,self.nely))
        return x

    def applyCantileiverSetup(self):
        anchorArray = np.zeros((self.nelx,self.nely))
        #fully anchor the far left side
        for y in range(self.nely):
            anchorArray[0,y] = 3
        self.updateFixed(anchorArray)

        # apply a force to the far richt side at the center
        self.updateForceVectors([[self.nelx-1,self.nely//2,0,1],[self.nelx-1,self.nely//2,0,0]])

    def blurFilterCutoff(self,x,cutoff):
        """
        Applyies a blur filter over the element and then removes any material below the cuttoff value
        good for removing checker boarding
        """
        
        xnew = self.H@np.reshape(x,(self.nelx*self.nely,1))/self.Hs
        #print(np.max(xnew),np.min(xnew))

        return np.reshape(np.where(xnew < cutoff,0,1),(self.nelx,self.nely))

    def OptimizeChunk(self,xPhys:np.ndarray,location:int,chunkSize:int=10,cutOff:float = 0.5):
        """
        This function minimizes the mass of a specified chunk of the given part
        Uses the scipy.optimize.minmize function

        The function takes:
        - x: the part it will optimize
        - location: an integer > 0 representing the random location where the part will be optimized
        - chunkSize: an integer between 1 and nelx/nely. Represents how large the square 'chunk' of elements will be optimized
        - cutOff: A float [0,1] representing at what point material will be considered in the part


        The function returns:
            xNew as the new updated values of x

        """
        #create a new copy of our elements so as to not acidentaly edit them while otimizing
        xnew=np.reshape(xPhys,(self.nelx*self.nely))
        move = 1

        #set up the chunk to be minimized
        sizeOfSubdivision = chunkSize
        numberOfSubdivision = self.nelx//sizeOfSubdivision

        #check the area of the part with the most material
        pool2D = AveragePooling2D(pool_size=(chunkSize,chunkSize),strides=(chunkSize,chunkSize),padding='same')

        activityMap = pool2D(np.reshape(xPhys,(1,self.nelx,self.nely,1)).astype('float32'))
        newShape = activityMap.shape[1]*activityMap.shape[2]

        activeValues = np.reshape(np.asarray(activityMap),(newShape))
        #remove all ones from the map as that indicates a completely full(support) area
        activeValues = np.where(activeValues == 1,0,activeValues)

        #find the number of zeros
        numZeros = len(activeValues) - len(activeValues[activeValues != 0])
        #arg sort the array to get the indexes of the sorted values then drop the fir
        indexes = np.argsort(activeValues)[numZeros:]
        if(len(indexes) <= 1):
            return xPhys,False

        #build a binary choise array filled with values starting at 0.5 and decreaseing by half each index
        binaryChoice = np.zeros((len(indexes)),dtype='float32')
        n = len(binaryChoice)
        for i in range(1,n+1):
            binaryChoice[n-i] = 1/(2**(i))
        
        #add the remainder value to the last index to ensure the sum of the array is one
        binaryChoice[0] += 1/(2**(n))

        l1 = np.random.choice(indexes,p=binaryChoice)

        x_coordinate = l1%numberOfSubdivision
        y_coordinate = l1//numberOfSubdivision

        move = np.zeros((self.nelx,self.nely))

        for x in range(x_coordinate*sizeOfSubdivision,(x_coordinate+1)*sizeOfSubdivision):
            for y in range(y_coordinate*sizeOfSubdivision,(y_coordinate+1)*sizeOfSubdivision):
                if(x < self.nelx and y < self.nely):
                    move[x,y] = self.maxElementChange
        move = np.reshape(move,(self.nelx*self.nely))


        ce,dc,_,_,_,_ = self.sensitivityAnalysis(xPhys)
        offset = (dc*xnew).sum()
        numberOfElements = len(xnew)

        b = np.array([self.complianceMax - ce + offset])
        c = np.ones(numberOfElements)

        A = np.zeros((1,len(dc)))
        A[0,:] = dc#work dc into a technically 2d array
        
        #define bouds of x to be within move distance
        x_lowerBound = np.maximum(0,xnew - move*np.ones(len(xnew)))
        x_lowerBound = np.where(self.passive == 1, 1, x_lowerBound)
        x_upperBound = np.minimum(1,xnew + move*np.ones(len(xnew)))
        x_upperBound = np.where(self.passive == 1, 1, x_upperBound)
        bounds = (np.array((x_lowerBound,x_upperBound))).T

        #create the minimization problem
        res = linprog(
            c = c,
            A_ub = A,
            b_ub = b,
            bounds = bounds,
            x0 = xnew)

        if(not res.success):
            #self.printResults(res)
            #print(np.array([x_lowerBound,x_upperBound]).T)
            return xPhys,False

        xnew = res.x

        #do the passives
        xnew = np.where(self.passive == 1, 0, xnew)
        xnew = np.where(self.passive == 2, 1, xnew)
            
        return self.blurFilterCutoff(xnew,cutOff),True






