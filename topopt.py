import os
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from matplotlib import pyplot as plt


class topOpter:
    def __init__(self,nelx,nely,volfrac,penal,rmin,ft,saveFile:bool=False):
        self.nelx =nelx
        self.nely = nely
        self.volfrac = volfrac
        self.penal = penal
        self.rmin = rmin
        self.ft = ft

        self.SaveAsFile = saveFile
        self.isError = False
        self.errorMarked = False

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
        self.x=volfrac * np.ones(nely*nelx,dtype=float)
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
        self.f[2*IX(nelx-10,nely-10)-1,0] = -1
        #self.f[2*IX(10,10),0] = 0.5
        

    

        #passive elements
        self.passive = np.zeros((nely) * (nelx))
        # for i in range(nelx):
        #     for j in range(nely):
        #         if np.sqrt((j-nely/2)**2+(i-nelx/3)**2) < nely/3:
        #             self.passive[IX(i,j)] = 1
        

        # Set loop counter and gradient vectors 
        self.loop=0
        self.change=1
        self.dv = np.ones(nely*nelx)
        self.dc = np.ones(nely*nelx)
        self.ce = np.ones(nely*nelx)

        self.obj = 0

        #setup the saving fileSystem
        if(self.SaveAsFile):
            workingDirectory = r"E:\TopoptGAfileSaves\ComplianceMinimization"#os.getcwd()
            agentDirectory = os.path.join(workingDirectory,"Agents")
            dimesionFolder = os.path.join(agentDirectory,"{}_{}".format(nelx,nely))
            pathExists = os.path.exists(dimesionFolder)
            if( not pathExists):
                os.makedirs(dimesionFolder)
            num = np.random.randint(1,999999)
            agentFolder = os.path.join(dimesionFolder,"Agent_{}".format(num))
            pathExists = os.path.exists(agentFolder)
            if(not pathExists):
                os.makedirs(agentFolder)
            else:
                # if the agent folder currently exist then create an Agent#_ folder
                foundOpenNumber = False
                currentNumber = 1
                while( not foundOpenNumber):
                    currentAgentN = os.path.join(dimesionFolder,"Agent{}_{}".format(currentNumber,num))
                    pathExists = os.path.exists(currentAgentN)
                    if(pathExists):
                        currentNumber += 1
                    else:
                        foundOpenNumber = True
                os.makedirs(currentAgentN)
                agentFolder = currentAgentN

            self.folderToSaveTo = agentFolder
    
    def saveLoadConditions(self):
        if(self.SaveAsFile):
            originalWorkingDirectory = os.getcwd()
            os.chdir(self.folderToSaveTo)
            fileNameToSaveAs = "loadConditions.csv"
            formating_array = np.array([self.volfrac,self.nelx,self.nely,self.penal,self.rmin])
            
            try:
                np.savez_compressed(fileNameToSaveAs,a=self.f,b=self.free,c=self.passive,d=formating_array)
            except:
                print("Something went wrong.")
                print("Tried to save: {}".format(fileNameToSaveAs))
            os.chdir(originalWorkingDirectory)
        else:
            return
    
    def saveIteration(self):
        if(self.SaveAsFile and not self.isError):
            originalWorkingDirectory = os.getcwd()
            os.chdir(self.folderToSaveTo)
            fileNameToSaveAs = f"iteration_{self.loop}" + ".csv"
            formating_array = np.array([self.obj,self.change,self.xPhys.sum()])
            
            try:
                np.savez_compressed(fileNameToSaveAs,a=self.x,b=self.xPhys,c=formating_array)
            except:
                print("Something went wrong.")
                print("Tried to save: {}".format(fileNameToSaveAs))
            os.chdir(originalWorkingDirectory)
        elif(self.isError and not self.errorMarked):
            #Mark that the solution is invalid thus should not be used
            originalWorkingDirectory = os.getcwd()
            os.chdir(self.folderToSaveTo)

            filesInDirectory = os.listdir()

            for file in filesInDirectory:
                os.rename(file,str("Invalid_" + file))



            os.chdir(originalWorkingDirectory)
            self.errorMarked = True

            

    def itterate(self):
        canCompute = (self.numberOfForces > 0) and (self.numberOfFixedPoints > 3 )
        if( self.change>0.01 and self.loop<2000 and canCompute):
            self.loop += 1

            # Setup and solve FE problem
            sK=((self.KE.flatten()[np.newaxis]).T*(self.Emin+(self.xPhys)**self.penal*(self.Emax-self.Emin))).flatten(order='F')
            K = coo_matrix((sK,(self.iK,self.jK)),shape=(self.ndof,self.ndof)).tocsc()

            # Remove constrained dofs from matrix
            K = K[self.free,:][:,self.free]

            # Solve system 
            self.u[self.free,:]=spsolve(K,self.f[self.free,:])    

            # Objective and sensitivity
            obj = 0
            for i in range(self.numberOfForces):
                Ui = self.u[:,i]
                #seperate to own funtion
                self.ce[:] = (np.dot(Ui[self.edofMat].reshape(self.nelx*self.nely,8),self.KE) * Ui[self.edofMat].reshape(self.nelx*self.nely,8) ).sum(1)
                obj += ((self.Emin+self.xPhys**self.penal*(self.Emax-self.Emin))*self.ce).sum()
                if(i == 0):
                    self.dc[:]=(-self.penal*self.xPhys**(self.penal-1)*(self.Emax-self.Emin))*self.ce
                else:
                    self.dc[:]= self.dc[:] + (-self.penal*self.xPhys**(self.penal-1)*(self.Emax-self.Emin))*self.ce
                self.dv[:] = np.ones(self.nely*self.nelx)

            # Sensitivity filtering:
            if self.ft==0:
                self.dc[:] = np.asarray((self.H*(self.x*self.dc))[np.newaxis].T/self.Hs)[:,0] / np.maximum(0.001,self.x)
            elif self.ft==1:
                self.dc[:] = np.asarray(self.H*(self.dc[np.newaxis].T/self.Hs))[:,0]
                self.dv[:] = np.asarray(self.H*(self.dv[np.newaxis].T/self.Hs))[:,0]

            # Optimality criteria
            self.xold[:]=self.x
            (self.x[:],self.g)= self.oc()

            # Filter design variables
            if self.ft==0:   self.xPhys[:]=self.x
            elif self.ft==1:    self.xPhys[:]=np.asarray(self.H*self.x[np.newaxis].T/self.Hs)[:,0]

            # Compute the change by the inf. norm
            self.change=np.linalg.norm(self.x.reshape(self.nelx*self.nely,1)-self.xold.reshape(self.nelx*self.nely,1),np.inf)

            self.obj = obj

            # Write iteration history to screen (req. Python 2.6 or newer)
            print("it.: {0} , obj.: {1:.3f} Vol.: {2:.3f}, ch.: {3:.3f}".format(self.loop,obj,(self.g+self.volfrac*self.nelx*self.nely)/(self.nelx*self.nely),self.change))
            return True
        else:
            return False
    
    def getPart(self):
        return self.xPhys.reshape((self.nelx,self.nely))
    
    def getDerivetiveOfSensitivity(self):
        return self.dc.reshape((self.nelx,self.nely))

    def clearPart(self):
        self.x=self.volfrac * np.ones(self.nely*self.nelx,dtype=float)
        self.xold=self.x.copy()
        self.xPhys=self.x.copy()
        self.change = 1 
        self.loop = 0
        self.dv = np.ones(self.nely*self.nelx)
        self.dc = np.ones(self.nely*self.nelx)
        self.ce = np.ones(self.nely*self.nelx)
        self.g=0
        
        #self.u=np.zeros((self.ndof,2))

    #element stiffness matrix
    def lk(self):
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

    # Optimality criterion
    def oc(self):
        l1=0
        l2=1e9
        move=0.2
        # reshape to perform vector operations
        xnew=np.zeros(self.nelx*self.nely)
        while ((l2-l1)/(l1+l2))>1e-3:
            lmid=0.5*(l2+l1)
            B_e = -self.dc/(self.dv*lmid)
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
            """
            xnew[:]= np.maximum(0.0,np.maximum(self.x-move,np.minimum(1.0,np.minimum(self.x+move,self.x*np.sqrt(B_e))))) # just a peicewise function in a single line

            #do the passives
            xnew = np.where(self.passive == 1, 0, xnew)
            xnew = np.where(self.passive == 2, 1, xnew)

            gt=self.g+np.sum((self.dv*(xnew-self.x)))

            if gt>0 :
                l1=lmid
            else:
                l2=lmid
            if(l1+l2 == 0):
                print("would have been an error")
                self.isError = True
                break
        return (xnew,gt)


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

    def showAnchors(self):
        print("Anchors")
        fixed = np.zeros(self.ndof)
        fixed[self.free] = 1
        self.drawPlots(fixed)
    
    def showForces(self):
        print("forceVectors")

        for i in range(self.numberOfForces):
            print("force:",i)
            self.drawPlots(self.f[:,i])


    def drawPlots(self,arr):
        print("arr1.shape = {}".format(arr.shape))
        l = max(arr.shape)//2

    def ApplyProblem(self, filledArea,supportArea,forceVector):
        self.passive = (2*filledArea).reshape(self.nelx*self.nely)
        self.updateFixed(supportArea*3)
        self.f = forceVector
        self.numberOfForces = 4
        self.u=np.zeros((2*(self.nelx+1)*(self.nely+1),4))

    def applyConstraints(self):
        #reshape x into a 1D array to perform calculations
        x = np.maximum(0,np.minimum(self.xPhys,1))
        #Apply the force fill and force free areas
        x = np.where(self.passive == 1,0,x)
        x = np.where(self.passive == 2,1,x)
        self.xPhys = x
