import numpy as np
from fenics import *
import ufl as ufl
from ufl import nabla_div
from fenics_adjoint import *
from pyadjoint import ipopt

# turn off redundant output in parallel
parameters["std_out_all_processes"] = False

L = 2.0                                         # Length
W = 1.0                                         # Width
p = Constant(3.0)                               # Penalization Factor for SIMP
eps = Constant(1.0e-3)                          # Epsilon value for SIMP to remove singularities
E = Constant(2.0e+11)                           # Young Modulus
nu = Constant(0.3)                              # Poisson's Ratio
lmbda = (E*nu) / ((1.0 + nu)*(1.0 - (2.0*nu)))  # Lame's first parameter
G = E / (2.0*(1.0 + nu))                        # Lame's second parameter / Shear Modulus
nelx = 100                                      # Number of elements in x-direction
nely = 50                                       # Number of elements in y-direction
C_max = Constant(50000.0)                       # Max Compliance
S_max = Constant(4.5e+8)                        # Max Stress
r = Constant(0.05)                              # Length Parameter for Helmholtz Filter

# Define Mesh
mesh = RectangleMesh(Point(0.0, 0.0), Point(L, W), nelx, nely)

# Define Function Spaces
U = VectorFunctionSpace(mesh, "Lagrange", 1) # Displacement Function Space
X = FunctionSpace(mesh, "Lagrange", 1)       # Density Function Space

# SIMP Function for Intermediate Density Penalization
def simp(x):
    return eps + (1 - eps) * x**p

# Calculate Strain from Displacements u
def strain(u):
    return 0.5*(nabla_grad(u) + nabla_grad(u).T)

# Calculate Stress from Displacements u
def sigma(u):
    return lmbda*nabla_div(u)*Identity(2) + 2*G*strain(u)

# Calculate Von Mises Stress from Displacements u
def von_mises(u):
    s = sigma(u) - (1./3)*tr(sigma(u))*Identity(2)  # deviatoric stress
    return sqrt(3./2*inner(s, s))

# RELU Function for Global Stress Constraint Computation
def relu(x):
       return x * (ufl.sign(x) + 1) * 0.5

load_marker = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
CompiledSubDomain("x[0]==l && x[1]<=2", l=nelx).mark(load_marker, 1)
ds = Measure("ds")(subdomain_data=load_marker)
    
# Volumetric Load
q = 5.0
f = Constant((0.0, q))

# Dirichlet BC em x[0] = 0
def Left_boundary(x, on_boundary):
    return on_boundary and abs(x[0]) < DOLFIN_EPS
u_L = Constant((0.0, 0.0))
bc = DirichletBC(U, u_L, Left_boundary)

# Helmholtz Filter for Design Variables. Helps remove checkerboarding and introduces mesh independency.
# Taken from work done by de Souza et al. (2020)
def helmholtz_filter(rho_n, r_min):
      V = rho_n.function_space()

      rho = TrialFunction(V)
      w = TestFunction(V)

      a = (r_min**2)*inner(grad(rho), grad(w))*dx + rho*w*dx
      L = rho_n*w*ds(1)

      A, b = assemble_system(a, L)
      rho = Function(V)
      solve(A, rho.vector(), b)

      return rho

# Forward problem solution. Solves for displacement u given density x.
def forward(x, l_min):
    rho = helmholtz_filter(x, l_min)
    u = TrialFunction(U)  # Trial Function
    v = TestFunction(U)   # Test Function
    a = simp(x)*inner(sigma(u), strain(v))*dx
    L = dot(f, v)*dx
    u = Function(U)
    solve(a == L, u, bc)
    return u  

# MAIN
if __name__ == "__main__":
    x = interpolate(Constant(0.5), X)  # Initial value of 0.5 for each element's density
    u = forward(x, 0.05)                     # Forward problem

    # Objective Functional to be Minimized
    # Includes Regularization Term to Avoid Checkerboarding
    J = assemble(x*dx)# + p_reg*dot(grad(x),grad(x))*dx)
    m = Control(x)  # Control
    Jhat = ReducedFunctional(J, m) # Reduced Functional

    lb = 0.0  # Inferior
    ub = 1.0  # Superior

    # Class for Enforcing Compliance Constraint
    class ComplianceConstraint(InequalityConstraint):
        def __init__(self, C):
            self.C  = float(C)
            self.tmpvec = Function(X)
            self.u = Function(U)

        def function(self, m):
            self.tmpvec.vector()[:] = m
            self.u = forward(self.tmpvec, 0.05)
            c_current = assemble(dot(f, self.u)*dx)
            if MPI.rank(MPI.comm_world) == 0:
                print("Current compliance: ", c_current)
            
            return [self.C - c_current]

        def jacobian(self, m):
            self.tmpvec.vector()[:] = m
            self.u = forward(self.tmpvec, 0.05)
            J_comp = assemble(dot(f, self.u)*dx)
            m_comp = Control(self.tmpvec)
            dJ_comp = compute_gradient(J_comp, m_comp)
            if MPI.rank(MPI.comm_world) == 0:
                print("Computed Derivative: ", -(dJ_comp.vector()[:]))

            return [-dJ_comp.vector()]

        def output_workspace(self):
            return [0.0]

        def length(self):
            """Return the number of components in the constraint vector (here, one)."""
            return 1

    # Class for Enforcing Stress Constraint
    class StressConstraint(InequalityConstraint):
        def __init__(self, S):
            self.S  = float(S)
            self.tmpvec = Function(X)
            self.u = Function(U)

        def function(self, m):
            self.tmpvec.vector()[:] = m
            self.u = forward(self.tmpvec, 0.05)
            s_current = assemble(relu((von_mises(self.u)*(self.tmpvec**0.5))-self.S)*dx)
            if MPI.rank(MPI.comm_world) == 0:
                print("Current stress: ", -s_current)

            return [-s_current]

        def jacobian(self, m):
            self.tmpvec.vector()[:] = m
            self.u = forward(self.tmpvec, 0.05)
            J_stress = assemble(relu((von_mises(self.u)*(self.tmpvec**0.5))-self.S)*dx)
            m_stress = Control(self.tmpvec)
            dJ_stress = compute_gradient(J_stress, m_stress)
            if MPI.rank(MPI.comm_world) == 0:
                print("Computed Derivative: ", -(dJ_stress.vector()[:]))

            return [-dJ_stress.vector()]

        def output_workspace(self):
            return [0.0]

        def length(self):
            """Return the number of components in the constraint vector (here, one)."""
            return 1  

    problem = MinimizationProblem(Jhat, bounds=(lb, ub), constraints =  [ComplianceConstraint(C_max), StressConstraint(S_max)])
    parameters = {"acceptable_tol": 1.0e-3, "maximum_iterations": 200}

    solver = IPOPTSolver(problem, parameters=parameters)
    rho_opt = solver.solve()

    File("output2/final_solution.pvd") << rho_opt
    xdmf_filename = XDMFFile(MPI.comm_world, "output2/final_solution.xdmf")
    xdmf_filename.write(rho_opt)
