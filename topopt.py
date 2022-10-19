# A 165 LINE TOPOLOGY OPTIMIZATION CODE BY NIELS AAGE AND VILLADS EGEDE JOHANSEN, JANUARY 2013
from __future__ import division
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from matplotlib import colors
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from functools import partial

#global p value for material removal
p=0

# MAIN DRIVER
def main(nelx,nely,volfrac,penal,rmin,ft):
	print("Minimum compliance problem with OC")
	print("ndes: " + str(nelx) + " x " + str(nely))
	print("volfrac: " + str(volfrac) + ", rmin: " + str(rmin) + ", penal: " + str(penal))
	print("Filter method: " + ["Sensitivity based","Density based"][ft])

	# Max and min stiffness
	Emin=1e-9
	Emax=1.0
	# dofs:
	ndof = 2*(nelx+1)*(nely+1)

	# Allocate design variables (as array), initialize and allocate sens.
	x=volfrac * np.ones(nely*nelx,dtype=float)
	xold=x.copy()
	xPhys=x.copy()

	g=0 # must be initialized to use the NGuyen/Paulino OC approach
	dc=np.zeros((nely,nelx), dtype=float)

	# FE: Build the index vectors for the for coo matrix format.
	KE=lk()
	edofMat=np.zeros((nelx*nely,8),dtype=int)
	for elx in range(nelx):
		for ely in range(nely):
			el = ely+elx*nely
			n1=(nely+1)*elx+ely
			n2=(nely+1)*(elx+1)+ely
			edofMat[el,:]=np.array([2*n1+2, 2*n1+3, 2*n2+2, 2*n2+3,2*n2, 2*n2+1, 2*n1, 2*n1+1])

	# Construct the index pointers for the coo format
	iK = np.kron(edofMat,np.ones((8,1))).flatten()
	jK = np.kron(edofMat,np.ones((1,8))).flatten()    

	# Filter: Build (and assemble) the index+data vectors for the coo matrix format
	nfilter=int(nelx*nely*((2*(np.ceil(rmin)-1)+1)**2))
	iH = np.zeros(nfilter)
	jH = np.zeros(nfilter)
	sH = np.zeros(nfilter)
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
					iH[cc]=row
					jH[cc]=col
					sH[cc]=np.maximum(0.0,fac)
					cc=cc+1
	# Finalize assembly and convert to csc format
	H=coo_matrix((sH,(iH,jH)),shape=(nelx*nely,nelx*nely)).tocsc()	
	Hs=H.sum(1)
	def IX(x,y):
		return y*nely + x
	# BC's and support
	dofs=np.arange(2*(nelx+1)*(nely+1))
	#fixed=np.union1d(dofs[0:2*(5):2],np.array([2*(nelx+1)*(nely+1)-1]))
	#fixed = np.array([0,1,3,4])
	fixed = dofs[0:2*(nely-1):1]
	free=np.setdiff1d(dofs,fixed)

	# Solution and RHS vectors
	# for n number of forces each of these must be and n-dimesional column vector
	f=np.zeros((ndof,2))
	u=np.zeros((ndof,2))

	# Set load
	f[2*IX(nelx-10,nely-10)-1,0] = -1
	#f[2*IX(10,10),1] = 0.5
	

	if(False):
		
		showNdof(f[:,0],nelx,nely)
		showNdof(f[:,1],nelx,nely)
		
		input()
		

	# Initialize plot and plot the initial design
	plt.ion() # Ensure that redrawing is possible
	fig,ax = plt.subplots(2,1)
	
	
	#slider_for_material.on_changed()
	im1 = ax[0].imshow(-xPhys.reshape((nelx,nely)).T, cmap='gray', interpolation='none',norm=colors.Normalize(vmin=-1,vmax=0))
	fig.show()
	sliderAxis = fig.add_axes([0.1, 0.25, 0.0225, 0.63])
	slider_for_material = Slider(
		ax=sliderAxis,
		label="p",
		valmin=-.999,
		valmax=0,
		valinit=0,
		orientation="vertical"
	)
	prevSliderVal = slider_for_material.val

	#passive elements
	passive = np.zeros((nely) * (nelx))
	for i in range(nelx):
		for j in range(nely):
			if np.sqrt((j-50)**2+(i-60)**2) < 20:
				passive[IX(i,j)] = 1
			if np.sqrt((j-35)**2+(i-20)**2) < 15:
				passive[IX(i,j)] = 1
			if np.sqrt((j-20)**2+(i-60)**2) < 10:
				passive[IX(i,j)] = 1
	

   	# Set loop counter and gradient vectors 
	loop=0
	change=1
	dv = np.ones(nely*nelx)
	dc = np.ones(nely*nelx)
	ce = np.ones(nely*nelx)
	im2 = ax[1].imshow(dc.reshape((nelx,nely)).T, cmap='plasma_r', interpolation='none',norm=colors.Normalize(vmin=-1,vmax=0))

	while change>0.01 and loop<2000:
		loop=loop+1

		# Setup and solve FE problem
		sK=((KE.flatten()[np.newaxis]).T*(Emin+(xPhys)**penal*(Emax-Emin))).flatten(order='F')
		K = coo_matrix((sK,(iK,jK)),shape=(ndof,ndof)).tocsc()

		# Remove constrained dofs from matrix
		K = K[free,:][:,free]

		# Solve system 
		u[free,:]=spsolve(K,f[free,:])    

		# Objective and sensitivity
		obj = 0
		for i in range(2):
			Ui = u[:,i]
			ce[:] = (np.dot(Ui[edofMat].reshape(nelx*nely,8),KE) * Ui[edofMat].reshape(nelx*nely,8) ).sum(1)
			obj += ((Emin+xPhys**penal*(Emax-Emin))*ce).sum()
			if(i == 0):
				dc[:]=(-penal*xPhys**(penal-1)*(Emax-Emin))*ce
			else:
				dc[:]= dc[:] + (-penal*xPhys**(penal-1)*(Emax-Emin))*ce
			dv[:] = np.ones(nely*nelx)

		# Sensitivity filtering:
		if ft==0:
			dc[:] = np.asarray((H*(x*dc))[np.newaxis].T/Hs)[:,0] / np.maximum(0.001,x)
		elif ft==1:
			dc[:] = np.asarray(H*(dc[np.newaxis].T/Hs))[:,0]
			dv[:] = np.asarray(H*(dv[np.newaxis].T/Hs))[:,0]

		# Optimality criteria
		xold[:]=x
		(x[:],g)=oc(nelx,nely,x,volfrac,dc,dv,g,passive)

		# Filter design variables
		if ft==0:   xPhys[:]=x
		elif ft==1:	xPhys[:]=np.asarray(H*x[np.newaxis].T/Hs)[:,0]

		# Compute the change by the inf. norm
		change=np.linalg.norm(x.reshape(nelx*nely,1)-xold.reshape(nelx*nely,1),np.inf)

		# Plot to screen
		if(plt.fignum_exists(fig.number)):
			if(prevSliderVal == slider_for_material.val):
				im1.set_array(updateImageDropOff(-xPhys.reshape((nelx,nely)).T,slider_for_material.val))
			else:
				im1.set_array(-xPhys.reshape((nelx,nely)).T)
			im2.set_array(dc.reshape((nelx,nely)).T)
			fig.canvas.draw()
			fig.canvas.flush_events()
		else:
			break

		# Write iteration history to screen (req. Python 2.6 or newer)
		print("it.: {0} , obj.: {1:.3f} Vol.: {2:.3f}, ch.: {3:.3f}".format(loop,obj,(g+volfrac*nelx*nely)/(nelx*nely),change))
		prevSliderVal = slider_for_material.val
	# Make sure the plot stays and that the shell remains	
	#plt.show()
	print("Done!")
	while(plt.fignum_exists(fig.number)):
		im1.set_array(updateImageDropOff(-xPhys.reshape((nelx,nely)).T,slider_for_material.val))
		im2.set_array(dc.reshape((nelx,nely)).T)
		fig.canvas.draw()
		fig.canvas.flush_events()
	
	#input("Press any key...")


#element stiffness matrix
def lk():
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
def oc(nelx,nely,x,volfrac,dc,dv,g,passive):
	l1=0
	l2=1e9
	move=0.2
	# reshape to perform vector operations
	xnew=np.zeros(nelx*nely)
	while ((l2-l1)/(l1+l2))>1e-3:
		lmid=0.5*(l2+l1)
		xnew[:]= np.maximum(0.0,np.maximum(x-move,np.minimum(1.0,np.minimum(x+move,x*np.sqrt(-dc/dv/lmid)))))

		#do the passives
		xnew = np.where(passive == 1, 0, xnew)
		xnew = np.where(passive == 2, 1, xnew)

		gt=g+np.sum((dv*(xnew-x)))

		if gt>0 :
			l1=lmid
		else:
			l2=lmid
		if(l1+l2 == 0):
			print("would have been an error")
			break
	return (xnew,gt)

def showNdof(arr :np.ndarray,nelx,nely):
	#2*(nelx+1)*(nely+1)
	print("arr1.shape = {}".format(arr.shape))
	l = max(arr.shape)//2

	arr = arr.reshape((2,l),order='F')
	dof_x = arr[0,:]
	dof_y = arr[1,:]

	print("arr2.shape = {}".format(arr.shape))
	print("dof_x.shape = {}".format(dof_x.shape))
	print("dof_y.shape = {}".format(dof_y.shape))
	
	fig,ax = plt.subplots(2)
	ax[0].imshow(dof_x.reshape((nelx+1,nely+1)).T)
	ax[0].set_title("X")
	ax[1].imshow(dof_y.reshape((nelx+1,nely+1)).T)
	ax[1].set_title("Y")
	plt.show()
	print()

def updateImageDropOff(imageArray,val):
	#remap image from [-1,0] to some other range
	rec = val/(val+1)
	v = rec*np.ones(imageArray.shape)
	above0 = np.where(imageArray > val,0,(imageArray/(val+1)) - rec)

	return above0

# The real main driver    
if __name__ == "__main__":
	# Default input parameters
	nelx=160
	nely=80
	volfrac=0.4
	rmin=5.4
	penal=3.0
	ft=0 # ft==0 -> sens, ft==1 -> dens
	import sys
	if len(sys.argv)>1: nelx   =int(sys.argv[1])
	if len(sys.argv)>2: nely   =int(sys.argv[2])
	if len(sys.argv)>3: volfrac=float(sys.argv[3])
	if len(sys.argv)>4: rmin   =float(sys.argv[4])
	if len(sys.argv)>5: penal  =float(sys.argv[5])
	if len(sys.argv)>6: ft     =int(sys.argv[6])
	main(nelx,nely,volfrac,penal,rmin,ft)


