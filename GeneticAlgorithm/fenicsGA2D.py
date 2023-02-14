from fenics import *
from ufl import nabla_div
import ufl as ufl
from fenics_adjoint import *
import numpy as np
import math


import petsc4py
petsc4py.init()
from petsc4py import PETSc


# Everything below, should be static
p = Constant(4)                                 # Penalization Factor for SIMP
eps = Constant(1.0e-3)                          # Epsilon value for SIMP to remove singularities
E = Constant(2.0e+11)                           # Young Modulus
nu = Constant(0.3)                              # Poisson's Ratio
lmbda = (E*nu) / ((1.0 + nu)*(1.0 - (2.0*nu)))  # Lame's first parameter
G = E / (2.0*(1.0 + nu))                        # Lame's second parameter / Shear Modulus

# Volumetric Load
q = -5000000.0
f = Constant((0.0, q))

nelx, nely = 15, 30


def calcRatio(a, b):
    gcd = np.gcd(a, b)

    aReduced = a / gcd
    bReduced = b / gcd
    
    return aReduced, bReduced

L, W = calcRatio(nelx, nely) # Length, Width

# Define mesh
mesh = RectangleMesh(Point(0.0, 0.0), Point(L, W), nelx - 1, nely - 1)

# Define Function Spaces
U = VectorFunctionSpace(mesh, "Lagrange", 1) # Displacement Function Space
X = FunctionSpace(mesh, "Lagrange", 1)       # Density Function Space

# Dirichlet BC em x[0] = 0
def Left_boundary(x, on_boundary):
    return on_boundary and abs(x[0]) < DOLFIN_EPS

u_L = Constant((0.0, 0.0))
bc = DirichletBC(U, u_L, Left_boundary)


# Calculate Strain from Displacements
def strain(u):
    return 0.5*(nabla_grad(u) + nabla_grad(u).T)

# Calculate Stress
def sigma(u):
    return lmbda*nabla_div(u)*Identity(2) + 2*G*strain(u)

# Calculate Von Mises Stress
def von_mises(u):
    s = sigma(u) - (1./3)*tr(sigma(u))*Identity(2)  # deviatoric stress
    return sqrt(3./2*inner(s, s))


def solveConstraints(solution):
    x = Function(X)
    solution = solution.flatten()
    x.vector()[:] = solution[vertex_to_dof_map(X)]

    u = forward(x, U, bc)

    stress = von_mises(u)

    vm = project(von_mises(u), X)

    stress = vm.vector()[:]
    compliance = assemble(dot(f, u)*dx)


    return stress, compliance


# Forward problem solution. Solves for displacement u given density x.
def forward(x, U, bc):
    # rho = helmholtz_filter(x, 0.05)
    u = TrialFunction(U)  # Trial Function
    v = TestFunction(U)   # Test Function
    
    # a = rho*inner(sigma(u), strain(v))*dx
    a = x*inner(sigma(u), strain(v))*dx
    L = dot(f, v)*dx
    u = Function(U)
    
    solve(a == L, u, bc)
    
    return u  