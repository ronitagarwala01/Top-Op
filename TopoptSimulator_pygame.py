import pygame
import math
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from matplotlib import pyplot as plt

"""
Pygame based real time part simulator

Controls:
    keyboard:
        a - cycle tool
        w - increase brush size
        s - decrease brush size
        d - clear everything
        space - clear only part
        k - plot anchors and forces(requires you to fully exit out of the plots before the sim will continue)

    mouse:
        Left button(brush) - place/activate tool
        Left button(vector) - begin Drawing Vector
        Left buttion(clear vector) - remove vector from list(pops vectors from array each frame so either click quickly to delete last vector or hold to delete all vectors)
        Left button(slider) - activate slider to move around(slider will stay active until you right click)

        Right buttion(brush) - remove current item
        Right button(vector) - create vector( will only create if the vector is long enough in the correct direction)
        Right button(slider) - deactivate slider to stop moving it around

        Note that the canvas that a tool brushes onto is shared for some tools adding another tool over it may delet what is currently there and deleting with one tool may also delete for other tools.

        The interface between the GUI and the optomizer is very incomplete. I don't think that they line up at all so what you draw is NOT what the optomizer sees.
        The optomizer will complain a lot when drawing, press space to clear the part and try the drawing

Problems to resolve:
    - sometimes the entire part just fills itself in with material.
    - (solved)getting the GUI elements to line up and fit into the optomizer elements
        - Mostly for the vectors
        - the Force fill and force empty are correct and do line up with the part
    - Not really code but knowing what items are required by the optomizer to be stable and actually run
        - currently the baseline is set that a single force and 5 or more achored points is required befor it will run

Notes for personal refinement:
    Around line 660 is all the drawing methods
    If you know pygame and would like to add keyboard inputs keyEvent[1] will return the key the user has just pressed down(not holding and not unpressed)
    keyEvent[1] will return seperate values for lowwer and upper case keys ('A' != 'a') so shift key register as different keys. See my non commented getKeyPressed() method for details


"""

BLACK = [0, 0, 0]
WHITE = [255, 255, 255]
LIGHT_GRAY = [200,200,200]
RED = [255,0,0]
ORANGE = [255,35,35]
GREEN = [0,255,0]
DARK_GREEN = [0,200,0]
BLUE = [0, 0, 255]
LIGHT_BLUE = [100,100,255]
PURPLE = [0,255,255]
DARK_RED = [100,0,0]

PI_OVER_8 = math.pi/8
TWO_PI = math.pi*2

pygame.init()
SIZE = [600,600]
screen = pygame.display.set_mode(SIZE)
pygame.display.set_caption("Part Simulator")


pygame.font.init() # you have to call this at the start, 
                   # if you want to use this module.
Arial_10 = pygame.font.SysFont("Arial", 15)
#textsurface = myfont.render("", False, RED)

VALID_KEYS = [pygame.K_SPACE, pygame.K_EXCLAIM, pygame.K_QUOTEDBL, pygame.K_HASH, pygame.K_DOLLAR, pygame.K_AMPERSAND, pygame.K_QUOTE, pygame.K_LEFTPAREN, pygame.K_RIGHTPAREN, pygame.K_ASTERISK, pygame.K_PLUS, pygame.K_COMMA, pygame.K_MINUS, pygame.K_PERIOD, pygame.K_SLASH, pygame.K_0, pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5, pygame.K_6, pygame.K_7, pygame.K_8, pygame.K_9, pygame.K_COLON, pygame.K_SEMICOLON, pygame.K_LESS, pygame.K_EQUALS, pygame.K_GREATER, pygame.K_QUESTION, pygame.K_AT, pygame.K_LEFTBRACKET, pygame.K_BACKSLASH, pygame.K_RIGHTBRACKET, pygame.K_CARET, pygame.K_UNDERSCORE, pygame.K_BACKQUOTE, pygame.K_a, pygame.K_b, pygame.K_c, pygame.K_d, pygame.K_e, pygame.K_f, pygame.K_g, pygame.K_h, pygame.K_i, pygame.K_j, pygame.K_k, pygame.K_l, pygame.K_m, pygame.K_n, pygame.K_o, pygame.K_p, pygame.K_q, pygame.K_r, pygame.K_s, pygame.K_t, pygame.K_u, pygame.K_v, pygame.K_w, pygame.K_x, pygame.K_y, pygame.K_z, pygame.K_KP_PERIOD, pygame.K_KP_DIVIDE, pygame.K_KP_MULTIPLY, pygame.K_KP_MINUS, pygame.K_KP_PLUS, pygame.K_KP_EQUALS]
VALID_CHARACTERS = [" ", "!", '"', "#", "$", "&", "'", "(", ")", "*", "+", ",", "-", ".", "/", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ":", ";", "<", "=", ">", "?", "@", "[", "\\", "]", "^", "_", "`", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", ".", "/", "*", "-", "+", "="]
VALID_SHIFT_CHARACTERS = [" ", "!", '"', "#", "$", "&", '"', "(", ")", "*", "+", "<", "_", ">", "?", ")", "!", "@", "#", "$", "%", "^", "&", "*", "(", ":", ":", "<", "+", ">", "?", "@", "{", "|", "}", "^", "_", "~", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", ">", "?", "*", "_", "+", "+"]
def getKeyPressed(shiftDown):
    """
    return the key press object
    [bigEvent ID :int, character pressed(see above arrays) :character, are you holding shift : boolean]
    """
    for event in pygame.event.get():   
        if event.type == pygame.QUIT:  
            return [0,"",shiftDown]
        if(event.type == pygame.KEYUP):
            if(event.key == pygame.K_RSHIFT or event.key == pygame.K_LSHIFT):
                shiftDown = False
        if(event.type == pygame.KEYDOWN):
            if(event.key == pygame.K_RSHIFT or event.key == pygame.K_LSHIFT):
                shiftDown = True
            for i,pygameKeypressConstant in enumerate(VALID_KEYS):
                if( event.key == pygameKeypressConstant):
                    if(shiftDown):
                        return [1,VALID_SHIFT_CHARACTERS[i],shiftDown]
                    else:
                        return [1,VALID_CHARACTERS[i],shiftDown]
            if(event.key == pygame.K_BACKSPACE):
                return [2,"",shiftDown]
            elif(event.key == pygame.K_DELETE):
                return [4,"",shiftDown]
        
            
        
    return [3,"",shiftDown]

class topOpter:
    def __init__(self,nelx,nely,volfrac,penal,rmin,ft):
        self.nelx =nelx
        self.nely = nely
        self.volfrac = volfrac
        self.penal = penal
        self.rmin = rmin
        self.ft = ft

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
            elif self.ft==1:	self.xPhys[:]=np.asarray(self.H*self.x[np.newaxis].T/self.Hs)[:,0]

            # Compute the change by the inf. norm
            self.change=np.linalg.norm(self.x.reshape(self.nelx*self.nely,1)-self.xold.reshape(self.nelx*self.nely,1),np.inf)



            # Write iteration history to screen (req. Python 2.6 or newer)
            print("it.: {0} , obj.: {1:.3f} Vol.: {2:.3f}, ch.: {3:.3f}".format(self.loop,obj,(self.g+self.volfrac*self.nelx*self.nely)/(self.nelx*self.nely),self.change))
            return True
        else:
            return False
	
    
    def getPart(self):
        return self.xPhys.reshape((self.nelx,self.nely))

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
            xnew[:]= np.maximum(0.0,np.maximum(self.x-move,np.minimum(1.0,np.minimum(self.x+move,self.x*np.sqrt(-self.dc/self.dv/lmid)))))

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

        arr = arr.reshape((2,l),order='F')
        dof_x = arr[0,:]
        dof_y = arr[1,:]

        print("arr2.shape = {}".format(arr.shape))
        print("dof_x.shape = {}".format(dof_x.shape))
        print("dof_y.shape = {}".format(dof_y.shape))
        
        fig,ax = plt.subplots(2)
        ax[0].imshow(dof_x.reshape((self.nelx+1,self.nely+1)).T)
        ax[0].set_title("X")
        ax[1].imshow(dof_y.reshape((self.nelx+1,self.nely+1)).T)
        ax[1].set_title("Y")
        plt.show()
        print()

class UI_Slider:
    def __init__(self,x,y,width,height,minVal,maxVal):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.rectVal = [x,y,width,height]
        self.minVal = minVal
        self.maxVal = maxVal
        self.currentNum = (maxVal-minVal)/2 + minVal
        self.pixelMin = int(self.x + self.width/10)
        self.pixelMax = int(self.x + 0.9*self.width)
        self.lineStart = [self.pixelMin, int(self.y + self.height/3)]
        self.lineEnd = [self.pixelMax, self.lineStart[1]]
        self.currentLoc = [int(self.x + self.width/2),self.lineStart[1]]
        self.active = False
    
    def updateValues(self,minVal,maxVal):
        self.minVal = minVal
        self.maxVal = maxVal
        self.currentNum = (maxVal-minVal)/2 + minVal
    
    def draw(self):
        pygame.draw.rect(screen,WHITE,self.rectVal)
        pygame.draw.rect(screen, BLACK,self.rectVal,1)
        pygame.draw.line(screen,BLACK,self.lineStart,self.lineEnd,1)
        pygame.draw.circle(screen, RED, self.currentLoc, 3)
        textsurface = Arial_10.render(str(int(self.currentNum*10)/10), False, BLACK)
        screen.blit(textsurface,[self.currentLoc[0]+5,self.currentLoc[1]+10])
    
    def reMap(self):
        self.currentNum = ((self.currentLoc[0]- self.pixelMin) / (self.pixelMax - self.pixelMin))*(self.maxVal - self.minVal) + self.minVal

    def update(self,pos,pressed):
        if(self.active):
            newVal = pos[0]
            if(newVal > self.pixelMax):
                newVal = self.pixelMax
            elif(newVal < self.pixelMin):
                newVal = self.pixelMin
            self.currentLoc[0] = newVal
            self.reMap()
            if(pressed[2]):
                self.active = False
        elif(pressed[0]):
            if(pos[0] > self.x and pos[1] > self.y and pos[0] < self.x+self.width and pos[1] < self.y+self.height):
                self.active = True

        return self.currentNum

    def setVal(self,newVal):
        if(newVal > self.maxVal):
            self.currentNum = self.maxVal
        elif(newVal < self.minVal):
            self.currentNum = self.minVal
        else:
            self.currentNum = newVal
        self.currentLoc[0] = int(((self.currentNum- self.minVal) / (self.maxVal - self.minVal))*(self.pixelMax - self.pixelMin) + self.pixelMin)
        return self.currentNum
  


def updateImageDropOff(imageArray,val):
	#remap image from [-1,0] to some other range
	rec = val/(val+1)
	#v = rec*np.ones(imageArray.shape)
	above0 = np.where(imageArray > val,0,(imageArray/(val+1)) - rec)

	return above0


def dist(p1,p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1] - p2[1])**2)

def colorAdd(c1,c2):
    return [int((c1[0]+c2[0])/2),int((c1[1]+c2[1])/2),int((c1[2]+c2[2])/2)]

def lockOff(num,minVal,maxVal):
    if(num > maxVal):
        return maxVal
    elif(num < minVal):
        return minVal
    return num

def interpolateColor(c1,c2,mix):
    """ 1 = c1\n
        0 = c2"""
    mix = lockOff(mix,0,1)
    inverseMix = 1 - mix
    r = c1[0] * mix + c2[0]*inverseMix
    g = c1[1] * mix + c2[1]*inverseMix
    b = c1[2] * mix + c2[2]*inverseMix
    return [int(r),int(g),int(b)]

def drawPart(nelx,nely,passiveArray,fixedArray,pixelSize,boundingRect,vectors,part,pMin):
    xStep = pixelSize
    for i in range(nelx+1):
        pygame.draw.line(screen,LIGHT_GRAY,[i*xStep+boundingRect[0],boundingRect[1]],[i*xStep+boundingRect[0],boundingRect[1]+boundingRect[3]])
    
    
    yStep = pixelSize
    for i in range(nely+1):
        pygame.draw.line(screen,LIGHT_GRAY,[boundingRect[0],i*yStep+boundingRect[1]],[boundingRect[0]+boundingRect[2],i*yStep+boundingRect[1]])
    
    partArray = updateImageDropOff(-part,-pMin)
    
    for x in range(nelx):
        for y in range(nely):
            c = 0
            pixelColor = WHITE
            if(passiveArray[x][y] == 1):
                c += 1
                pixelColor = colorAdd(pixelColor,BLUE)
            elif(passiveArray[x][y] == 2):
                c += 2
                pixelColor = colorAdd(pixelColor,GREEN)
            if(fixedArray[x][y] == 1):
                c += 4
                pixelColor = colorAdd(pixelColor,ORANGE)
            elif(fixedArray[x][y] == 2):
                c += 8
                pixelColor = colorAdd(pixelColor,PURPLE)
            elif(fixedArray[x][y] == 3):
                c += 16
                pixelColor = colorAdd(pixelColor,DARK_RED)
            if(abs(partArray[x,y]) > pMin):
                pixelColor = interpolateColor(BLACK,pixelColor,abs(partArray[x,y]))
                pygame.draw.rect(screen,pixelColor,[boundingRect[0]+x*pixelSize,boundingRect[1]+y*pixelSize,pixelSize,pixelSize])
            elif(c > 0):
                pygame.draw.rect(screen,pixelColor,[boundingRect[0]+x*pixelSize,boundingRect[1]+y*pixelSize,pixelSize,pixelSize])

    for x,y,Vx,Vy in vectors:
        pygame.draw.line(screen,RED,[boundingRect[0]+x*pixelSize,boundingRect[1]+y*pixelSize],[boundingRect[0]+(x-Vx)*pixelSize,boundingRect[1]+(y-Vy)*pixelSize])
        pygame.draw.circle(screen, RED, [boundingRect[0]+x*pixelSize,boundingRect[1]+y*pixelSize], 2)
            
def addToArray(array,x,y,val,brushSize):
    x_start = max(x - brushSize + 1,0)
    x_end = min(x + brushSize,len(array))
    y_start = max(y - brushSize + 1,0)
    y_end = min(y + brushSize,len(array[0]))
    for i in range(x_start,x_end):
        for j in range(y_start,y_end):
            array[i][j] = val


def main():
    
    nelx=120
    nely=60
    volfrac=0.4
    rmin=5.4
    penal=3.0
    ft=0 # ft==0 -> sens, ft==1 -> dens

    partOptomizer = topOpter(nelx,nely,volfrac,penal,rmin,ft)

    passiveArray = []
    fixedArray = []
    for x in range(nelx):
        passiveArray.append([])
        fixedArray.append([])
        for y in range(nely):
            passiveArray[x].append(0)
            fixedArray[x].append(0)

    
    offset = 50
    pixelSize = (min(SIZE[0],SIZE[1]) - 2*offset)//max(nelx,nely)
    boundingRect = [offset,offset,pixelSize*nelx,pixelSize*nely]
    materialRemovalSlider = UI_Slider(boundingRect[0],boundingRect[1]+boundingRect[3],boundingRect[2],offset,0,.99)
    materialRemovalSlider.setVal(0)

    keyEvent = [3,"",False]  
    FramesPerSecond = 5

    

    drawModes = ["Force Free","Force Fill","Fixed X","Fixed Y","Fixed XY","Add Vector X","Add Vector Y","Clear Vectors"]
    mode = 0
    brushSize = 1
    vectors = []
    creatingVector = False
    v_start = [0,0]
    v_end = [0,0]
    partUpdating = True
    parameterChanged = False

    #clear out all preinitalized data from the part
    partOptomizer.clearPart()
    partOptomizer.updatePassives(passiveArray)
    partOptomizer.updateFixed(fixedArray)
    partOptomizer.updateForceVectors([])

    clock = pygame.time.Clock()
    done = False
    while not done:
        keyEvent = getKeyPressed(keyEvent[2])
        pos = pygame.mouse.get_pos()
        pressed = pygame.mouse.get_pressed()
        if(keyEvent[0] == 0):
            done = True
        elif(keyEvent[1] == 'a'):
            mode += 1
            mode %= len(drawModes)
        elif(keyEvent[1] == 'd'):
            for x in range(nelx):
                for y in range(nely):
                    passiveArray[x][y] = 0
                    fixedArray[x][y] = 0
            parameterChanged = True
            vectors = []
            creatingVector = False
            partOptomizer.clearPart()
            partOptomizer.updatePassives(passiveArray)
            partOptomizer.updateFixed(fixedArray)
            partOptomizer.updateForceVectors(vectors)
        elif(keyEvent[1] == ' '):
            partOptomizer.clearPart()

        elif(keyEvent[1] == 'w'):
            brushSize += 1
            if(brushSize > 10):
                brushSize = 10
        elif(keyEvent[1] == 's'):
            brushSize -= 1
            if(brushSize < 1):
                brushSize = 1
        elif(keyEvent[1] == 'z'):
            partOptomizer.showAnchors()
            #input()
            partOptomizer.showForces()
            #input()
            print("Done showing part.")

        if(pressed[0]):
            if(pos[0] >= boundingRect[0] and pos[0] < boundingRect[0]+boundingRect[2] and pos[1] >= boundingRect[1] and pos[1] < boundingRect[1]+boundingRect[3]):
                x = (pos[0]-boundingRect[0])//pixelSize
                y = (pos[1]-boundingRect[1])//pixelSize
                parameterChanged = True
                if(mode == 0):
                    addToArray(passiveArray,x,y,1,brushSize)
                elif(mode == 1):
                    addToArray(passiveArray,x,y,2,brushSize)
                elif(mode == 2):
                    addToArray(fixedArray,x,y,1,brushSize)
                elif(mode == 3):
                    addToArray(fixedArray,x,y,2,brushSize)
                elif(mode == 4):
                    addToArray(fixedArray,x,y,3,brushSize)
                elif(mode == 5 or mode == 6):
                    creatingVector = True
                    v_start = [x,y]
                elif(mode == 7 and len(vectors) > 0):
                    vectors.pop()
                
        elif(pressed[2]):
            if(pos[0] >= boundingRect[0] and pos[0] < boundingRect[0]+boundingRect[2] and pos[1] >= boundingRect[1] and pos[1] < boundingRect[1]+boundingRect[3]):
                x = (pos[0]-boundingRect[0])//pixelSize
                y = (pos[1]-boundingRect[1])//pixelSize
                parameterChanged = True
                if(mode == 0):
                    addToArray(passiveArray,x,y,0,brushSize)
                elif(mode == 1):
                    addToArray(passiveArray,x,y,0,brushSize)
                elif(mode == 2):
                    addToArray(fixedArray,x,y,0,brushSize)
                elif(mode == 3):
                    addToArray(fixedArray,x,y,0,brushSize)
                elif(mode == 4):
                    addToArray(fixedArray,x,y,0,brushSize)
                elif(mode == 5 and creatingVector):
                    creatingVector = False
                    if(abs(x-v_start[0]) > 1):
                        v_end = [x,y]
                        vectors.append([v_start[0],v_start[1],v_start[0] - v_end[0],0])
                elif(mode == 6 and creatingVector):
                    creatingVector = False
                    if(abs(y-v_start[1]) > 1):
                        v_end = [x,y]
                        vectors.append([v_start[0],v_start[1],0,v_start[1] - v_end[1]])

                
        if(parameterChanged):
            partOptomizer.updatePassives(passiveArray)
            partOptomizer.updateFixed(fixedArray)
            partOptomizer.updateForceVectors(vectors)
            parameterChanged = False
        partUpdating = partOptomizer.itterate()
        

        
        screen.fill(WHITE)
        # draws the part to the screen
        # nelx, nely is the number of elements in x and y direction
        # passive array stores the information regarding the force fill or force free parameters of the part
        # fixed array stores the achors for the part in the x,y directions
        # vectors is the list of force vectors(stored as [x,y,Vx,Vy])
        # part optomizer.GetPart() returns the current shape of the part
        # the materialRemovalSlider update funtion is the pmin witch is a dropoff value for drawing the part to remove any material less than pMin
        # the slider is finiky and requires you to right click it to unselect and stop sliding
        drawPart(nelx,nely,passiveArray,fixedArray,pixelSize,boundingRect,vectors,partOptomizer.getPart(),materialRemovalSlider.update(pos,pressed))
        materialRemovalSlider.draw()
        if(creatingVector):
            pygame.draw.line(screen, RED, [boundingRect[0]+v_start[0]*pixelSize,boundingRect[1]+v_start[1]*pixelSize], pos)

        text = Arial_10.render(str(drawModes[mode]), True, BLACK)
        screen.blit(text,[0,0])

        pygame.display.flip()

        clock.tick(FramesPerSecond)

main()
pygame.quit()
