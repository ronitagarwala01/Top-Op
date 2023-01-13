from fenics import *
from ufl import nabla_div
from fenics_adjoint import *
import numpy as np

# L = 1.0                                         # Length
# W = 1.0                                         # Width


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


def calcRatio(a, b):
    gcd = np.gcd(a, b)

    aReduced = a / gcd
    bReduced = b / gcd
    
    return aReduced, bReduced

def solveConstraints(nelx, nely, solution):

    L, W = calcRatio(nelx, nely) # Length, Width

    # Define mesh
    mesh = RectangleMesh(Point(0.0, 0.0), Point(L, W), nelx, nely)

    # Define Function Spaces
    U = VectorFunctionSpace(mesh, "Lagrange", 1) # Displacement Function Space
    X = FunctionSpace(mesh, "Lagrange", 1)       # Density Function Space

    x = Function(X)
    # This does not work
    x.vector() = solution

    u_L = Constant((0.0, 0.0))
    bc = DirichletBC(U, u_L, Left_boundary)

    u = forward(x, U)

    stress = sigma(u)

    compliance = assemble(dot(b, u)*dx + Constant(1.0e-8)*dot(grad(x_in),grad(x_in))*dx)

    return stress, compliance

# SIMP Function
def simp(x):
    return eps + (1 - eps) * x**p

# Calculate Strain from Displacements
def strain(u):
    return 0.5*(nabla_grad(u) + nabla_grad(u).T)

# Calculate Stress
def sigma(u):
    return lmbda*nabla_div(u)*Identity(2) + 2*G*strain(u)


# Dirichlet BC em x[0] = 0
def Left_boundary(x, on_boundary):
    return on_boundary and abs(x[0]) < DOLFIN_EPS

# Forward problem solution. Solves for displacement u given density x.
def forward(x, U):
    u = TrialFunction(U)  ## Trial and test functions
    w = TestFunction(U)

    sigma = lmbda*tr(sym(grad(u)))*Identity(2) + 2*G*sym(grad(u)) ## Stress
    F = simp(x)*inner(sigma, grad(w))*dx - dot(f, w)*dx

    a, L = lhs(F), rhs(F)
    u = Function(U)
    solve(a == L, u, bc)

    return u