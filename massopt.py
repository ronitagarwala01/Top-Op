from fenics import *
from ufl import nabla_div
from fenics_adjoint import *
from pyadjoint import ipopt

# turn off redundant output in parallel
parameters["std_out_all_processes"] = False

L = 2.0                                         # Length
W = 1.0                                         # Width
p = Constant(4)                               # Penalization Factor for SIMP
eps = Constant(1.0e-3)                          # Epsilon value for SIMP to remove singularities
E = Constant(2.0e+11)                           # Young Modulus
nu = Constant(0.3)                              # Poisson's Ratio
lmbda = (E*nu) / ((1.0 + nu)*(1.0 - (2.0*nu)))  # Lame's first parameter
G = E / (2.0*(1.0 + nu))                        # Lame's second parameter / Shear Modulus
nelx = 100                                      # Number of elements in x-direction
nely = 50                                       # Number of elements in y-direction
C_max = Constant(49000.0)
S_max = 100000000.0

# Define mesh
mesh = RectangleMesh(Point(0.0, 0.0), Point(L, W), nelx, nely)

# Define Function Spaces
U = VectorFunctionSpace(mesh, "Lagrange", 1) # Displacement Function Space
X = FunctionSpace(mesh, "Lagrange", 1)       # Density Function Space

# SIMP Function
def simp(x):
    return eps + (1 - eps) * x**p

# Calculate Strain from Displacements
def strain(u):
    return 0.5*(nabla_grad(u) + nabla_grad(u).T)

# Calculate Stress
def sigma(u):
    return lmbda*nabla_div(u)*Identity(2) + 2*G*strain(u)

# Volumetric Load
q = -5000000.0
f = Constant((0.0, q))

# Dirichlet BC em x[0] = 0
def Left_boundary(x, on_boundary):
    return on_boundary and abs(x[0]) < DOLFIN_EPS
u_L = Constant((0.0, 0.0))
bc = DirichletBC(U, u_L, Left_boundary)

# Forward problem solution. Solves for displacement u given density x.
def forward(x):
    '''
    u = TrialFunction(U)  # Trial Function
    v = TestFunction(U)   # Test Function
    a = simp(x)*inner(sigma(u), strain(v))*dx
    L = dot(f, v)*dx
    u = Function(U)
    solve(a == L, u, bc)
    return u
    '''
    u = TrialFunction(U)  ## Trial and test functions
    w = TestFunction(U)
    sigma = lmbda*tr(sym(grad(u)))*Identity(2) + 2*G*sym(grad(u)) ## Stress
    F = simp(x)*inner(sigma, grad(w))*dx - dot(f, w)*dx
    a, L = lhs(F), rhs(F)
    u = Function(U)
    solve(a == L, u, bc)
    return u

# MAIN
if __name__ == "__main__":
    x = interpolate(Constant(0.5), X)  # Initial value of 0.5 for each element's density
    u = forward(x)                     # Forward problem

    # Objective Functional to be Minimized
    # Includes Regularization Term to Avoid Checkerboarding
    J = assemble(simp(x)*dx)# + Constant(1.0e-3)*dot(grad(x),grad(x))*dx)
    m = Control(x)                        # Control
    Jhat = ReducedFunctional(J, m)  # Reduced Functional

    lb = 0.0  # Inferior
    ub = 1.0  # Superior

    # Class for Enforcing Compliance Constraint
    class ComplianceConstraint(InequalityConstraint):
        def __init__(self, C):
            self.C  = float(C)
            self.tmpvec = Function(X)
            self.u = interpolate(u, U)

        def function(self, m):
            self.tmpvec.vector()[:]=m
            print(self.tmpvec.vector()[:])
            self.u = forward(self.tmpvec)
            c_current = assemble(dot(f, self.u)*dx)
            if MPI.rank(MPI.comm_world) == 0:
                print("Current compliance: ", c_current)
            return [self.C - c_current]

        def jacobian(self, m):
            print(self.tmpvec.vector()[:])
            self.tmpvec.vector()[:] = -p*m**(p-1)*assemble(dot(f, self.u)*dx)
            print(self.tmpvec.vector()[:])
            return [self.tmpvec.vector()]

        def output_workspace(self):
            return [0.0]

        def length(self):
            """Return the number of components in the constraint vector (here, one)."""
            return 1   


    #compliance_constraint = UFLInequalityConstraint(C_max - dot(f, u)*dx)
    #C_max = assemble(dot(f, u)*dx)
    #C = dot(f, u)
    #print((C_max-dot(f,u))*dx)
    #compliance_constraint = UFLInequalityConstraint((C_max-dot(f,u))*dx, m2)

    problem = MinimizationProblem(Jhat, bounds=(lb, ub), constraints=ComplianceConstraint(C_max))
    parameters = {"acceptable_tol": 1.0e-3, "maximum_iterations": 100}

    solver = IPOPTSolver(problem, parameters=parameters)
    rho_opt = solver.solve()

    File("output2/final_solution.pvd") << rho_opt
    xdmf_filename = XDMFFile(MPI.comm_world, "output2/final_solution.xdmf")
    xdmf_filename.write(rho_opt)