"""
This code needs to be converted from matlab to python

You will need to convert the matlab code into python code.
most of this will require changing the array indexing
This is an attempt at a 1-1 translation from MATLAB to python, this will likely not work

Notes:
    - Most of the generic matlab functions have numpy implementations
    - Matlab indexes start at 1 not 0 so they at len() not len()-1
    - When you see <variable> = 0:<variable | integer>
        - it is equivalent to <variable> = np.arange(<variable | integer>)

    - initVal:step:endVal := Increment index by the value step on each iteration, or decrements index when step is negative.

"""
# A 169 LINE 3D TOPOLOGY OPTIMIZATION CODE BY LIU AND TOVAR (JUL 2013)
import numpy as np
from numpy.matlib import repmat
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from matplotlib import colors
import matplotlib.pyplot as plt


def top3d(nelx,nely,nelz,volfrac,penal,rmin):
    # USER-DEFINED LOOP PARAMETERS
    maxloop = 200    # Maximum number of iterations
    tolx = 0.01      # Terminarion criterion
    displayflag = 0  # Display structure flag

    # USER-DEFINED MATERIAL PROPERTIES
    E0 = 1           # Young's modulus of solid material
    Emin = 1e-9      # Young's modulus of void-like material
    nu = 0.3         # Poisson's ratio

    # USER-DEFINED LOAD DOFs
    il,jl,kl = np.meshgrid(nelx, 0, np.arange(nelz))       # Coordinates
    loadnid = kl*(nelx+1)*(nely+1)+il*(nely+1)+(nely+1-jl) # Node IDs
    loaddof = 3*loadnid[:] - 1                             # DOFs

    # USER-DEFINED SUPPORT FIXED DOFs
    iif,jf,kf = np.meshgrid(0,np.arange(nely),np.arange(nelz))  # Coordinates
    fixednid = kf*(nelx+1)*(nely+1)+iif*(nely+1)+(nely+1-jf)    # Node IDs
    fixeddof = np.array([[3*fixednid[:]], [3*fixednid[:]-1], [3*fixednid[:]-2]])  # DOFs

    # PREPARE FINITE ELEMENT ANALYSIS
    nele = nelx*nely*nelz
    ndof = 3*(nelx+1)*(nely+1)*(nelz+1)
    print(loaddof.shape)
    F = coo_matrix((-1*np.ones(10),(loaddof.reshape(10),np.zeros(10))),(ndof,1))#reshape must be changes
    U = np.zeros(ndof)
    freedofs = np.setdiff1d(np.arange(ndof),fixeddof)
    KE = lk_H8(nu)
    nodegrd = np.reshape(np.arange((nely+1)*(nelx+1)),(nely+1,nelx+1)) #node grid
    nodeids = np.reshape(nodegrd[0:-1,0:-1],(nely*nelx,1)) #node ids 


    #nodeidz = (0:(nely+1)*(nelx+1):(nelz-1)*(nely+1)*(nelx+1))
    nodeidz = np.arange(0,(nelz-1)*(nely+1)*(nelx+1),step=(nely+1)*(nelx+1))
    print(nodeidz.shape,nodeids.shape)
    nodeids = nodeids* np.ones((len(nodeids),len(nodeidz))) + nodeidz*np.ones((len(nodeids),len(nodeidz)))

    edofVec = 3*nodeids[:]+1
    #do I know if this next line works? no.
    edofMat = repmat(edofVec,1,24) + repmat([0, 1, 2, 3*nely + [3, 4, 5, 0, 1, 2], -3, -2, -1,  3*(nely+1)*(nelx+1)+[0, 1, 2, 3*nely + [3, 4, 5, 0, 1, 2,], -3, -2, -1]],nele,1)

    iK = np.reshape(np.kron(edofMat,np.ones((24,1))).T, (24*24*nele,1))
    jK = np.reshape(np.kron(edofMat,np.ones((1,24))).T, (24*24*nele,1))

    # PREPARE FILTER
    iH = np.ones((nele*(2*(np.ceil(rmin)-1)+1)^2,1))
    jH = np.ones(len(iH))
    sH = np.zeros(len(iH))
    k = 0
    for k1 in range(nelz):
        for i1 in range(nelx):
            for j1 in range(nely):
                e1 = (k1-1)*nelx*nely + (i1-1)*nely+j1
                for k2 in range(max(k1-(np.ceil(rmin)-1),1),min(k1+(np.ceil(rmin)-1),nelz)):
                    for i2 in range(max(i1-(np.ceil(rmin)-1),1),min(i1+(np.ceil(rmin)-1),nelx)):
                        for j2 in range(max(j1-(np.ceil(rmin)-1),1),min(j1+(np.ceil(rmin)-1),nely)):
                            e2 = (k2-1)*nelx*nely + (i2-1)*nely+j2
                            k = k+1
                            iH[k] = e1
                            jH[k] = e2
                            sH[k] = max(0,rmin-np.sqrt((i1-i2)^2+(j1-j2)^2+(k1-k2)^2))
   


    #H = sparse(iH,jH,sH)
    H = coo_matrix((sH, (iH,jH))).tocsc() 
    Hs = H.sum(1)

    # INITIALIZE ITERATIOn
    x = volfrac * np.ones(nely*nelx*nelz)
    xPhys = x.copy() 
    loop = 0 
    change = 1
    # START ITERATION
    while change > tolx and loop < maxloop:
        loop = loop + 1

        # FE-ANALYSIS
        sK = np.reshape(KE[:] * (Emin+(xPhys[:].T**penal)*(E0-Emin)),24*24*nele,1)
        #K = sparse(iK,jK,sK) K = (K+K')/2
        K = coo_matrix((sK,(iK,jK))).tocsc() 
        K = (K+K.T)/2
        #U[freedofs,:] = K(freedofs,freedofs)\F[freedofs,:]
        U[freedofs,:] = spsolve(K,F[freedofs,:])
        
        """
        Everything above should be correctly implemented in python( with exception of the actual function code)

        start converting the code below here
        """
        # OBJECTIVE FUNCTION AND SENSITIVITY ANALYSIS
        ce = np.reshape(np.sum((U[edofMat]*KE)*U[edofMat],axis=2),[nely,nelx,nelz])
        c = np.sum(np.sum(np.sum((Emin+(xPhys**penal)*(E0-Emin))*ce)))
        dc = -penal*(E0-Emin)*(xPhys**(penal-1))*ce
        dv = np.ones(nely,nelx,nelz)

        # FILTERING AND MODIFICATION OF SENSITIVITIES
        dc[:] = H*(dc[:]/Hs)  
        dv[:] = H*(dv[:]/Hs)

        # OPTIMALITY CRITERIA UPDATE
        l1 = 0
        l2 = 1e9
        move = 0.2
        while ((l2-l1)/(l1+l2) > 1e-3):
            lmid = 0.5*(l2+l1)
            xnew = np.max(0,np.max(x-move,np.min(1,np.min(x+move,x*np.sqrt(-dc/dv/lmid)))))
            xPhys[:] = (H*xnew[:])/Hs
            if (np.sum(xPhys[:]) > volfrac*nele):
                l1 = lmid 
            else: 
                l2 = lmid
        
        change = np.linalg.norm(xnew[:]-x[:], ord = np.inf)
        x = xnew

        # PRINT RESULTS
        print("It.{:#5i} Obj.{:#11.4f} Vol.{:#7.3f} ch.{:#7.3f}".format(loop,c,np.mean(xPhys[:]),change))

        # PLOT DENSITIES
        if displayflag:
            display_3D(xPhys) ##ok<UNRCH>



# === GENERATE ELEMENT STIFFNESS MATRIX ===
def lk_H8(nu):
    A = np.array([[32, 6, -8, 6, -6, 4, 3, -6, -10, 3, -3, -3, -4, -8],
        [-48, 0, 0, -24, 24, 0, 0, 0, 12, -12, 0, 12, 12, 12]])
    print(A.T.shape,np.array([[1], [nu]]).shape)
    k = 1/144*A.T@np.array([[1], [nu]])

    K1 = np.array([[k[1-1], k[2-1], k[2-1], k[3-1], k[5-1], k[5-1]],
        [k[2-1], k[1-1], k[2-1], k[4-1], k[6-1], k[7-1]],
        [k[2-1], k[2-1], k[1-1], k[4-1], k[7-1], k[6-1]],
        [k[3-1], k[4-1], k[4-1], k[1-1], k[8-1], k[8-1]],
        [k[5-1], k[6-1], k[7-1], k[8-1], k[1-1], k[2-1]],
        [k[5-1], k[7-1], k[6-1], k[8-1], k[2-1], k[1-1]]])
    K2 = np.array([[k[9-1],  k[8-1],  k[12-1], k[6-1],  k[4-1],  k[7-1]],
        [k[8-1],  k[9-1],  k[12-1], k[5-1],  k[3-1],  k[5-1]],
        [k[10-1], k[10-1], k[13-1], k[7-1],  k[4-1],  k[6-1]],
        [k[6-1],  k[5-1],  k[11-1], k[9-1],  k[2-1],  k[10-1]],
        [k[4-1],  k[3-1],  k[5-1],  k[2-1],  k[9-1],  k[12-1]],
        [k[11-1], k[4-1],  k[6-1],  k[12-1], k[10-1], k[13-1]]])
    K3 = np.array([[k[6-1],  k[7-1],  k[4-1],  k[9-1],  k[12-1], k[8-1]],
        [k[7-1],  k[6-1],  k[4-1],  k[10-1], k[13-1], k[10-1]],
        [k[5-1],  k[5-1],  k[3-1],  k[8-1],  k[12-1], k[9-1]],
        [k[9-1],  k[10-1], k[2-1],  k[6-1],  k[11-1], k[5-1]],
        [k[12-1], k[13-1], k[10-1], k[11-1], k[6-1],  k[4-1]],
        [k[2-1],  k[12-1], k[9-1],  k[4-1],  k[5-1],  k[3-1]]])
    K4 = np.array([[k[14-1], k[11-1], k[11-1], k[13-1], k[10-1], k[10-1]],
        [k[11-1], k[14-1], k[11-1], k[12-1], k[9-1],  k[8-1]],
        [k[11-1], k[11-1], k[14-1], k[12-1], k[8-1],  k[9-1]],
        [k[13-1], k[12-1], k[12-1], k[14-1], k[7-1],  k[7-1]],
        [k[10-1], k[9-1],  k[8-1],  k[7-1],  k[14-1], k[11-1]],
        [k[10-1], k[8-1],  k[9-1],  k[7-1],  k[11-1], k[14-1]]])
    K5 = np.array([[k[1-1], k[2-1],  k[8-1],  k[3-1], k[5-1],  k[4-1]],
        [k[2-1], k[1-1],  k[8-1],  k[4-1], k[6-1],  k[11-1]],
        [k[8-1], k[8-1],  k[1-1],  k[5-1], k[11-1], k[6-1]],
        [k[3-1], k[4-1],  k[5-1],  k[1-1], k[8-1],  k[2-1]],
        [k[5-1], k[6-1],  k[11-1], k[8-1], k[1-1],  k[8-1]],
        [k[4-1], k[11-1], k[6-1],  k[2-1], k[8-1],  k[1-1]]])
    K6 = np.array([[k[14-1], k[11-1], k[7-1],  k[13-1], k[10-1], k[12-1]],
        [k[11-1], k[14-1], k[7-1],  k[12-1], k[9-1],  k[2-1]],
        [k[7-1],  k[7-1],  k[14-1], k[10-1], k[2-1],  k[9-1]],
        [k[13-1], k[12-1], k[10-1], k[14-1], k[7-1],  k[11-1]],
        [k[10-1], k[9-1],  k[2-1],  k[7-1],  k[14-1], k[7-1]],
        [k[12-1], k[2-1],  k[9-1],  k[11-1], k[7-1],  k[14-1]]])
    KE = 1/((nu+1)*(1-2*nu))* np.array([[K1,  K2,  K3,  K4],
                                        [K2.T,  K5,  K6,  K3.T],
                                        [K3.T, K6,  K5.T, K2.T],
                                        [K4,  K3,  K2,  K1.T]])
# === DISPLAY 3D TOPOLOGY (ISO-VIEW) ===
def display_3D(rho):
    return


"""
def display_3D(rho):
    nely,nelx,nelz = rho.shape
    hx = 1 
    hy = 1 
    hz = 1            # User-defined unit element size
    face = np.array([   [1, 2, 3, 4],
                        [2, 6, 7, 3], 
                        [4, 3, 7, 8], 
                        [1, 5, 8, 4], 
                        [1, 2, 6, 5], 
                        [5, 6, 7, 8]])
    #set(gcf,'Name','ISO display','NumberTitle','off')
    for k in range(0,nelz):
        z = (k-1)*hz
        for i in range(0,nelx):
            x = (i-1)*hx
            for j in range(0,nely):
                y = nely*hy - (j-1)*hy

                if (rho(j,i,k) > 0.5):  # User-defined display density threshold
                    vert = np.array([   [x, y, z], 
                                        [ x, y-hx, z], 
                                        [ x+hx, y-hx, z], 
                                        [ x+hx, y, z], 
                                        [ x, y, z+hx], 
                                        [x, y-hx, z+hx], 
                                        [ x+hx, y-hx, z+hx], 
                                        [x+hx, y, z+hx]])
                    vert[:,[2, 3]] = vert[:,[3, 2]] 
                    vert[:,2,:] = -vert[:,2,:]
                    patch('Faces',face,'Vertices',vert,'FaceColor',[0.2+0.8*(1-rho[j,i,k]),0.2+0.8*(1-rho[j,i,k]),0.2+0.8*(1-rho[j,i,k])])
"""


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




# The real main driver    
if __name__ == "__main__":
    nelx=10
    nely=10
    nelz = 10
    volfrac=0.4
    rmin=5.4
    penal=3.0

    top3d(nelx,nely,nelz,volfrac,penal,rmin)