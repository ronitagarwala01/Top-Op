import pygame
import math
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from matplotlib import pyplot as plt
from topopt import topOpter

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
