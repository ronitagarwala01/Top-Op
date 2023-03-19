import numpy as np
from fenics import *
from dolfin import *
from mshr import *
import ufl as ufl
from ufl import nabla_div
from fenics_adjoint import *
from pyadjoint import ipopt

# turn off redundant output in parallel
parameters["std_out_all_processes"] = False

L = 2.0                                         # Length
W = 1.0                                         # Width
D = 0.5                                         # Depth
p = Constant(5.0)                               # Penalization Factor for SIMP
p_norm = Constant(8.0)                          # P-Normalization Term
q = Constant(0.5)                               # Relaxation Factor for Stress
eps = Constant(1.0e-3)                          # Epsilon value for SIMP to remove singularities
nu = Constant(0.3)                              # Poisson's Ratio
Y = Constant(2.0e+11)                           # Young Modulus
nelx = 100                                      # Number of elements in x-direction
nely = 50                                       # Number of elements in y-direction
nelz = 25                                       # Number of elements in z-direction
C_max = Constant(20.0)                          # Max Compliance
S_max = Constant(3.0e+6)                        # Max Stress
r = Constant(0.025)                             # Length Parameter for Helmholtz Filter
radius_1 = Constant(0.15)                       # Radius for first circle
x_1 = Constant(0.5)                             # X-coordinate for first circle
y_1 = Constant(0.5)                            # Y-coordinate for first circle
radius_2 = Constant(0.15)                       # Radius for second circle
x_2 = Constant(1.0)                             # X-coordinate for second circle
y_2 = Constant(0.5)                            # Y-coordinate for second circle
radius_3 = Constant(0.15)                       # Radius for third circle
x_3 = Constant(1.5)                             # X-coordinate for third circle
y_3 = Constant(0.5)                            # Y-coordinate for third circle

# Define Mesh
mesh = RectangleMesh(Point(0.0, 0.0), Point(L, W), nelx, nely)

circle_1 = CompiledSubDomain('(x[0]-x_1)*(x[0]-x_1) + (x[1]-y_1)*(x[1]-y_1) <= radius_1*radius_1 + tol', x_1=x_1, y_1=y_1, radius_1=radius_1, tol=DOLFIN_EPS)
circle_2 = CompiledSubDomain('(x[0]-x_2)*(x[0]-x_2) + (x[1]-y_2)*(x[1]-y_2) <= radius_2*radius_2 + tol', x_2=x_2, y_2=y_2, radius_2=radius_2, tol=DOLFIN_EPS)
circle_3 = CompiledSubDomain('(x[0]-x_3)*(x[0]-x_3) + (x[1]-y_3)*(x[1]-y_3) <= radius_3*radius_3 + tol', x_3=x_3, y_3=y_3, radius_3=radius_3, tol=DOLFIN_EPS)

boundary_1 = CompiledSubDomain('(x[0]-x_1)*(x[0]-x_1) + (x[1]-y_1)*(x[1]-y_1) <= (radius_1+0.030)*(radius_1+0.030) + tol', x_1=x_1, y_1=y_1, radius_1=radius_1, tol=DOLFIN_EPS)
boundary_2 = CompiledSubDomain('(x[0]-x_2)*(x[0]-x_2) + (x[1]-y_2)*(x[1]-y_2) <= (radius_2+0.030)*(radius_2+0.030) + tol', x_2=x_2, y_2=y_2, radius_2=radius_2, tol=DOLFIN_EPS)
boundary_3 = CompiledSubDomain('(x[0]-x_3)*(x[0]-x_3) + (x[1]-y_3)*(x[1]-y_3) <= (radius_3+0.030)*(radius_3+0.030) + tol', x_3=x_3, y_3=y_3, radius_3=radius_3, tol=DOLFIN_EPS)

rigid_bar = CompiledSubDomain('x[1] <= 0.75 + tol && x[1] >= 0.25 + tol', tol=DOLFIN_EPS)

# Initialize mesh function for interior domains
domains = MeshFunction("size_t", mesh, mesh.topology().dim())
domains.set_all(0)
boundary_1.mark(domains, 4)
boundary_2.mark(domains, 4)
boundary_3.mark(domains, 4)
rigid_bar.mark(domains, 5)
circle_1.mark(domains, 1)
circle_2.mark(domains, 2)
circle_3.mark(domains, 3)

v_domains = MeshFunction("size_t", mesh, 0)
v_domains.set_all(0)
boundary_1.mark(v_domains, 4)
boundary_2.mark(v_domains, 4)
boundary_3.mark(v_domains, 4)
rigid_bar.mark(v_domains, 5)
circle_1.mark(v_domains, 1)
circle_2.mark(v_domains, 2)
circle_3.mark(v_domains, 3)

# Define new measures associated with the interior domains
dx = Measure('dx', domain = mesh, subdomain_data = domains)

# Define Function Spaces
U = VectorFunctionSpace(mesh, "Lagrange", 1) # Displacement Function Space
X = FunctionSpace(mesh, "DG", 0)       # Density Function Space

# Creating Function for Youngs Modulus across Mesh
Q = FunctionSpace(mesh, "DG", 0)
E = Function(Q)
x_ = Q.tabulate_dof_coordinates()

for i in range(x_.shape[0]):
    if domains.array()[i] == 1 or domains.array()[i] == 2 or domains.array()[i] == 3 or domains.array()[i] == 5:
        E.vector().vec().setValueLocal(i, Constant(1.0e+22))
    else:
        E.vector().vec().setValueLocal(i, Y)

lmbda = (E*nu) / ((1.0 + nu)*(1.0 - (2.0*nu)))  # Lame's first parameter
G = E / (2.0*(1.0 + nu))                        # Lame's second parameter / Shear Modulus

# Creating Function for Load Vectors across Mesh
F = VectorFunctionSpace(mesh, "DG", 0)
f = Function(F)
f_ = F.tabulate_dof_coordinates().reshape(x_.shape[0], 2, 2)

for i in range(f_.shape[0]):
    if domains.array()[i] == 1:
        f.vector().vec().setValueLocal(2*i, 0.0)
        f.vector().vec().setValueLocal(2*i+1, 10000000.0)
    elif domains.array()[i] == 2:
        f.vector().vec().setValueLocal(2*i, 0.0)
        f.vector().vec().setValueLocal(2*i+1, -20000000.0)
    elif domains.array()[i] == 3:
        f.vector().vec().setValueLocal(2*i, 0.0)
        f.vector().vec().setValueLocal(2*i+1, 10000000.0)
    else:
        f.vector().vec().setValueLocal(2*i, 0.0)
        f.vector().vec().setValueLocal(2*i+1, 0.0)

# SIMP Function for Intermediate Density Penalization
def simp(x):
    return eps + (1 - eps) * x**p

# Calculate Strain from Displacements u
def strain(u):
    return 0.5*(nabla_grad(u) + nabla_grad(u).T)

# Calculate Cauchy Stress Matrix from Displacements u
def sigma(u):
    return lmbda*nabla_div(u)*Identity(2) + 2*G*strain(u)

# Calculate Von Mises Stress from Displacements u
def von_mises(u):
    VDG = FunctionSpace(mesh, "DG", 0)
    s = sigma(u) - (1./3)*tr(sigma(u))*Identity(2) # Deviatoric Stress
    von_Mises = sqrt(3./2*inner(s, s))
    u, v = TrialFunction(VDG), TestFunction(VDG)
    a = inner(u, v)*dx
    L = inner(von_Mises, v)*dx(0) + inner(von_Mises, v)*dx(4)
    stress = Function(VDG)
    solve(a==L, stress)
    return stress

# RELU^2 Function for Global Stress Constraint Computation
def relu(x):
    return (x * (ufl.sign(x) + 1) * 0.5) ** 2

# Helmholtz Filter for Design Variables. Helps remove checkerboarding and introduces mesh independency.
# Taken from work done by de Souza et al. (2020)
def helmholtz_filter(rho_n):
      V = rho_n.function_space()

      rho = TrialFunction(V)
      w = TestFunction(V)

      a = (r**2)*inner(grad(rho), grad(w))*dx(0) + (r**2)*inner(grad(rho), grad(w))*dx(4) + rho*w*dx
      L = rho_n*w*dx

      A, b = assemble_system(a, L)
      rho = Function(V)
      solve(A, rho.vector(), b)

      return rho

# Forward problem solution. Solves for displacement u given density x.
def forward(x):
    rho = helmholtz_filter(x)
    u = TrialFunction(U)  # Trial Function
    v = TestFunction(U)   # Test Function
    a = simp(rho)*inner(sigma(u), strain(v))*dx
    L = dot(f, v)*dx
    u = Function(U)
    solve(a == L, u)
    return u  

# MAIN
if __name__ == "__main__":
    x = interpolate(Constant(0.99), X)  # Initial value of 0.5 for each element's density

    BAR = Function(X)

    x_cords = X.tabulate_dof_coordinates()
    print(x_cords.shape)
    print(v_domains.array().shape)

    for i in range(x_cords.shape[0]):
        if domains.array()[i] == 1 or domains.array()[i] == 2 or domains.array()[i] == 3 or domains.array()[i] == 5:
            # print(v_domains.)
            print(x_cords)
            BAR.vector().vec().setValueLocal(i, 1.0)
        else:
            BAR.vector().vec().setValueLocal(i, 0.0)

    File("output2/bar_density.pvd") << BAR
