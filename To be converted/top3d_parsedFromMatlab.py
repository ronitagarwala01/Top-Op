# Generated with SMOP  0.41-beta
from libsmop import *

import numpy as np
# Topopt3D.m

    # AN 169 LINE 3D TOPOLOGY OPITMIZATION CODE BY LIU AND TOVAR (JUL 2013)
    
@function
def top3d(nelx=None,nely=None,nelz=None,volfrac=None,penal=None,rmin=None,*args,**kwargs):
    varargin = top3d.varargin
    nargin = top3d.nargin

    # USER-DEFINED LOOP PARAMETERS
    maxloop=200
# Topopt3D.m:4
    
    tolx=0.01
# Topopt3D.m:5
    
    displayflag=0
# Topopt3D.m:6
    
    # USER-DEFINED MATERIAL PROPERTIES
    E0=1
# Topopt3D.m:8
    
    Emin=1e-09
# Topopt3D.m:9
    
    nu=0.3
# Topopt3D.m:10
    
    # USER-DEFINED LOAD DOFs
    il,jl,kl=np.meshgrid(nelx,0,arange(0,nelz))
# Topopt3D.m:12
    
    loadnid=dot(dot(kl,(nelx + 1)),(nely + 1)) + dot(il,(nely + 1)) + (nely + 1 - jl)
# Topopt3D.m:13
    
    loaddof=dot(3,ravel(loadnid)) - 1
# Topopt3D.m:14
    
    # USER-DEFINED SUPPORT FIXED DOFs
    iif,jf,kf=np.meshgrid(0,arange(0,nely),arange(0,nelz))
# Topopt3D.m:16
    
    fixednid=dot(dot(kf,(nelx + 1)),(nely + 1)) + dot(iif,(nely + 1)) + (nely + 1 - jf)
# Topopt3D.m:17
    
    fixeddof=concat([[dot(3,ravel(fixednid))],[dot(3,ravel(fixednid)) - 1],[dot(3,ravel(fixednid)) - 2]])
# Topopt3D.m:18
    
    # PREPARE FINITE ELEMENT ANALYSIS
    nele=dot(dot(nelx,nely),nelz)
# Topopt3D.m:20
    ndof=dot(dot(dot(3,(nelx + 1)),(nely + 1)),(nelz + 1))
# Topopt3D.m:21
    F=sparse(loaddof,1,- 1,ndof,1)
# Topopt3D.m:22
    U=zeros(ndof,1)
# Topopt3D.m:23
    freedofs=setdiff(arange(1,ndof),fixeddof)
# Topopt3D.m:24
    KE=lk_H8(nu)
# Topopt3D.m:25
    nodegrd=reshape(arange(1,dot((nely + 1),(nelx + 1))),nely + 1,nelx + 1)
# Topopt3D.m:26
    nodeids=reshape(nodegrd(arange(1,end() - 1),arange(1,end() - 1)),dot(nely,nelx),1)
# Topopt3D.m:27
    nodeidz=arange(0,dot(dot((nelz - 1),(nely + 1)),(nelx + 1)),dot((nely + 1),(nelx + 1)))
# Topopt3D.m:28
    nodeids=repmat(nodeids,size(nodeidz)) + repmat(nodeidz,size(nodeids))
# Topopt3D.m:29
    edofVec=dot(3,ravel(nodeids)) + 1
# Topopt3D.m:30
    edofMat=repmat(edofVec,1,24) + repmat(concat([0,1,2,dot(3,nely) + concat([3,4,5,0,1,2]),- 3,- 2,- 1,dot(dot(3,(nely + 1)),(nelx + 1)) + concat([0,1,2,dot(3,nely) + concat([3,4,5,0,1,2]),- 3,- 2,- 1])]),nele,1)
# Topopt3D.m:31
    iK=reshape(kron(edofMat,ones(24,1)).T,dot(dot(24,24),nele),1)
# Topopt3D.m:34
    jK=reshape(kron(edofMat,ones(1,24)).T,dot(dot(24,24),nele),1)
# Topopt3D.m:35
    # PREPARE FILTER
    iH=ones(dot(nele,(dot(2,(ceil(rmin) - 1)) + 1) ** 2),1)
# Topopt3D.m:37
    jH=ones(size(iH))
# Topopt3D.m:38
    sH=zeros(size(iH))
# Topopt3D.m:39
    k=0
# Topopt3D.m:40
    for k1 in arange(1,nelz).reshape(-1):
        for i1 in arange(1,nelx).reshape(-1):
            for j1 in arange(1,nely).reshape(-1):
                e1=dot(dot((k1 - 1),nelx),nely) + dot((i1 - 1),nely) + j1
# Topopt3D.m:44
                for k2 in arange(max(k1 - (ceil(rmin) - 1),1),min(k1 + (ceil(rmin) - 1),nelz)).reshape(-1):
                    for i2 in arange(max(i1 - (ceil(rmin) - 1),1),min(i1 + (ceil(rmin) - 1),nelx)).reshape(-1):
                        for j2 in arange(max(j1 - (ceil(rmin) - 1),1),min(j1 + (ceil(rmin) - 1),nely)).reshape(-1):
                            e2=dot(dot((k2 - 1),nelx),nely) + dot((i2 - 1),nely) + j2
# Topopt3D.m:48
                            k=k + 1
# Topopt3D.m:49
                            iH[k]=e1
# Topopt3D.m:50
                            jH[k]=e2
# Topopt3D.m:51
                            sH[k]=max(0,rmin - sqrt((i1 - i2) ** 2 + (j1 - j2) ** 2 + (k1 - k2) ** 2))
# Topopt3D.m:52
    
    H=sparse(iH,jH,sH)
# Topopt3D.m:59
    Hs=sum(H,2)
# Topopt3D.m:60
    # INITIALIZE ITERATION
    x=repmat(volfrac,concat([nely,nelx,nelz]))
# Topopt3D.m:62
    xPhys=copy(x)
# Topopt3D.m:63
    loop=0
# Topopt3D.m:64
    change=1
# Topopt3D.m:65
    # START ITERATION
    while change > tolx and loop < maxloop:

        loop=loop + 1
# Topopt3D.m:68
        sK=reshape(dot(ravel(KE),(Emin + dot(ravel(xPhys).T ** penal,(E0 - Emin)))),dot(dot(24,24),nele),1)
# Topopt3D.m:70
        K=sparse(iK,jK,sK)
# Topopt3D.m:71
        K=(K + K.T) / 2
# Topopt3D.m:71
        U[freedofs,arange()]=numpy.linalg.solve(K(freedofs,freedofs),F(freedofs,arange()))
# Topopt3D.m:72
        ce=reshape(sum(multiply((dot(U(edofMat),KE)),U(edofMat)),2),concat([nely,nelx,nelz]))
# Topopt3D.m:74
        c=sum(sum(sum(multiply((Emin + dot(xPhys ** penal,(E0 - Emin))),ce))))
# Topopt3D.m:75
        dc=multiply(dot(dot(- penal,(E0 - Emin)),xPhys ** (penal - 1)),ce)
# Topopt3D.m:76
        dv=ones(nely,nelx,nelz)
# Topopt3D.m:77
        ravel[dc]=dot(H,(ravel(dc) / Hs))
# Topopt3D.m:79
        ravel[dv]=dot(H,(ravel(dv) / Hs))
# Topopt3D.m:80
        l1=0
# Topopt3D.m:82
        l2=1000000000.0
# Topopt3D.m:82
        move=0.2
# Topopt3D.m:82
        while (l2 - l1) / (l1 + l2) > 0.001:

            lmid=dot(0.5,(l2 + l1))
# Topopt3D.m:84
            xnew=max(0,max(x - move,min(1,min(x + move,multiply(x,sqrt(- dc / dv / lmid))))))
# Topopt3D.m:85
            ravel[xPhys]=(dot(H,ravel(xnew))) / Hs
# Topopt3D.m:86
            if sum(ravel(xPhys)) > dot(volfrac,nele):
                l1=copy(lmid)
# Topopt3D.m:87
            else:
                l2=copy(lmid)
# Topopt3D.m:87

        change=max(abs(ravel(xnew) - ravel(x)))
# Topopt3D.m:89
        x=copy(xnew)
# Topopt3D.m:90
        fprintf(' It.:%5i Obj.:%11.4f Vol.:%7.3f ch.:%7.3f\n',loop,c,mean(ravel(xPhys)),change)
        #if displayflag:
            #clf
            #display_3D(xPhys)

    
    #clf
    #display_3D(xPhys)
    return

    
    # === GENERATE ELEMENT STIFFNESS MATRIX ===
    
@function
def lk_H8(nu=None,*args,**kwargs):
    varargin = lk_H8.varargin
    nargin = lk_H8.nargin

    A=concat([[32,6,- 8,6,- 6,4,3,- 6,- 10,3,- 3,- 3,- 4,- 8],[- 48,0,0,- 24,24,0,0,0,12,- 12,0,12,12,12]])
# Topopt3D.m:102
    k=dot(dot(1 / 144,A.T),concat([[1],[nu]]))
# Topopt3D.m:104
    K1=concat([[k(1),k(2),k(2),k(3),k(5),k(5)],[k(2),k(1),k(2),k(4),k(6),k(7)],[k(2),k(2),k(1),k(4),k(7),k(6)],[k(3),k(4),k(4),k(1),k(8),k(8)],[k(5),k(6),k(7),k(8),k(1),k(2)],[k(5),k(7),k(6),k(8),k(2),k(1)]])
# Topopt3D.m:106
    K2=concat([[k(9),k(8),k(12),k(6),k(4),k(7)],[k(8),k(9),k(12),k(5),k(3),k(5)],[k(10),k(10),k(13),k(7),k(4),k(6)],[k(6),k(5),k(11),k(9),k(2),k(10)],[k(4),k(3),k(5),k(2),k(9),k(12),k(11),k(4),k(6),k(12),k(10),k(13)]])
# Topopt3D.m:112
    K3=concat([[k(6),k(7),k(4),k(9),k(12),k(8)],[k(7),k(6),k(4),k(10),k(13),k(10)],[k(5),k(5),k(3),k(8),k(12),k(9)],[k(9),k(10),k(2),k(6),k(11),k(5)],[k(12),k(13),k(10),k(11),k(6),k(4)],[k(2),k(12),k(9),k(4),k(5),k(3)]])
# Topopt3D.m:118
    K4=concat([[k(14),k(11),k(11),k(13),k(10),k(10)],[k(11),k(14),k(11),k(12),k(9),k(8)],[k(11),k(11),k(14),k(12),k(8),k(9)],[k(13),k(12),k(12),k(14),k(7),k(7)],[k(10),k(9),k(8),k(7),k(14),k(11)],[k(10),k(8),k(9),k(7),k(11),k(14)]])
# Topopt3D.m:124
    K5=concat([[k(1),k(2),k(8),k(3),k(5),k(4)],[k(2),k(1),k(8),k(4),k(6),k(11)],[k(8),k(8),k(1),k(5),k(11),k(6)],[k(3),k(4),k(5),k(1),k(8),k(2)],[k(5),k(6),k(11),k(8),k(1),k(8)],[k(4),k(11),k(6),k(2),k(8),k(1)]])
# Topopt3D.m:130
    K6=concat([[k(14),k(11),k(7),k(13),k(10),k(12)],[k(11),k(14),k(7),k(12),k(9),k(2)],[k(7),k(7),k(14),k(10),k(2),k(9)],[k(13),k(12),k(10),k(14),k(7),k(11)],[k(10),k(9),k(2),k(7),k(14),k(7)],[k(12),k(2),k(9),k(11),k(7),k(14)]])
# Topopt3D.m:136
    KE=dot(1 / (dot((nu + 1),(1 - dot(2,nu)))),concat([[K1,K2,K3,K4],[K2.T,K5,K6,K3.T],[K3.T,K6,K5.T,K2.T],[K4,K3,K2,K1.T]]))
# Topopt3D.m:142
    return KE
    
    
    # === DISPLAY 3D TOPOLOGY (ISO-VIEW) ===
    
@function
def display_3D(rho=None,*args,**kwargs):
    varargin = display_3D.varargin
    nargin = display_3D.nargin

    nely,nelx,nelz=size(rho,nargout=3)
# Topopt3D.m:150
    hx=1
# Topopt3D.m:151
    hy=1
# Topopt3D.m:151
    hz=1
# Topopt3D.m:151
    
    face=concat([[1,2,3,4],[2,6,7,3],[4,3,7,8],[1,5,8,4],[1,2,6,5],[5,6,7,8]])
# Topopt3D.m:152
    set(gcf,'Name','ISO display','NumberTitle','off')
    for k in arange(1,nelz).reshape(-1):
        z=dot((k - 1),hz)
# Topopt3D.m:155
        for i in arange(1,nelx).reshape(-1):
            x=dot((i - 1),hx)
# Topopt3D.m:157
            for j in arange(1,nely).reshape(-1):
                y=dot(nely,hy) - dot((j - 1),hy)
# Topopt3D.m:159
                if (rho(j,i,k) > 0.5):
                    vert=concat([[x,y,z],[x,y - hx,z],[x + hx,y - hx,z],[x + hx,y,z],[x,y,z + hx],[x,y - hx,z + hx],[x + hx,y - hx,z + hx],[x + hx,y,z + hx]])
# Topopt3D.m:161
                    vert[arange(),concat([2,3])]=vert(arange(),concat([3,2]))
# Topopt3D.m:162
                    vert[arange(),2,arange()]=- vert(arange(),2,arange())
# Topopt3D.m:162
                    patch('Faces',face,'Vertices',vert,'FaceColor',concat([0.2 + dot(0.8,(1 - rho(j,i,k))),0.2 + dot(0.8,(1 - rho(j,i,k))),0.2 + dot(0.8,(1 - rho(j,i,k)))]))
                    hold('on')
    
    axis('equal')
    axis('tight')
    axis('off')
    box('on')
    view(concat([30,30]))
    pause(1e-06)
    return
    

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
    nelx=5
    nely=5
    nelz = 5
    volfrac=0.4
    rmin=5.4
    penal=3.0

    top3d(nelx,nely,nelz,volfrac,penal,rmin)