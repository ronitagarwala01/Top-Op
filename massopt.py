from fenics import *
from ufl import nabla_div
import numpy as np
from fenics_adjoint import *
from pyadjoint import ipopt

# turn off redundant output in parallel
parameters["std_out_all_processes"] = False

L = 2.0                                         # Length
W = 1.0                                         # Width
p = Constant(4)                                 # Penalization Factor for SIMP
alpha = Constant(0.9)                           # Mass Penalization Constant
eps = Constant(1.0e-3)                          # Epsilon value for SIMP to remove singularities
E = Constant(2.0e+11)                           # Young Modulus
nu = Constant(0.3)                              # Poisson's Ratio
lmbda = (E*nu) / ((1.0 + nu)*(1.0 - (2.0*nu)))  # Lame's first parameter
G = E / (2.0*(1.0 + nu))                        # Lame's second parameter / Shear Modulus
nelx = 100                                      # Number of elements in x-direction
nely = 50                                       # Number of elements in y-direction
C_max = Constant(50000.0)                           # Max Compliance
S_max = Constant(10000000.0)                    # Max Stress

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

# Calculate Von Mises Stress
def von_mises(u):
    s = sigma(u) - (1./3)*tr(sigma(u))*Identity(2)  # deviatoric stress
    return sqrt(3./2*inner(s, s))

# Volumetric Load
q = 5000000.0
f = Constant((0.0, q))

# Dirichlet BC em x[0] = 0
def Left_boundary(x, on_boundary):
    return on_boundary and abs(x[0]) < DOLFIN_EPS
u_L = Constant((0.0, 0.0))
bc = DirichletBC(U, u_L, Left_boundary)

# Forward problem solution. Solves for displacement u given density x.
def forward(x):
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
    u = forward(x)                     # Forward problem

    vm=project(von_mises(u), X)
    print(vm.vector()[:])

    def eval_cb(j, dj, x):
        u = forward(x)
        print("Objective (Mass): ", j)
        print("Compliance: ", assemble(dot(f,u)*dx))
        print("Derivative: ", dj.vector()[:])
        print("Mass: ", assemble(x*dx))

    # Objective Functional to be Minimized
    # Includes Regularization Term to Avoid Checkerboarding
    J = assemble(x*dx + Constant(1.0e-4)*dot(grad(x),grad(x))*dx)
    m = Control(x)  # Control
    Jhat = ReducedFunctional(J, m, derivative_cb_post=eval_cb)  # Reduced Functional

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
            print("Current control vector (density): ", self.tmpvec.vector()[:])
            self.u = forward(self.tmpvec)
            print("Current u vector: ", self.u.vector()[:])
            c_current = assemble(dot(f, self.u)*dx)
            if MPI.rank(MPI.comm_world) == 0:
                print("Current compliance: ", c_current)
                #print("Current Compliance 2: ", np.sum(project(self.tmpvec ** p * psi(self.u), X).vector()[:])*0.0004)

            return [self.C - c_current]

        def jacobian(self, m):
            J_comp = assemble(dot(f, u)*dx)
            m_comp = Control(x)
            dJ_comp = compute_gradient(J_comp, m_comp)
            print("Computed Derivative: ", dJ_comp.vector()[:])
            return [-dJ_comp.vector()]

        def output_workspace(self):
            return [0.0]

        def length(self):
            """Return the number of components in the constraint vector (here, one)."""
            return 1  


    #print(psi(u).vector()[:])

    #compliance_constraint = UFLInequalityConstraint(C_max-dot(f, u)*dx + Constant(1.0e-8)*dot(grad(x),grad(x))*dx, m)

    problem = MinimizationProblem(Jhat, bounds=(lb, ub), constraints = ComplianceConstraint(C_max))
    parameters = {"acceptable_tol": 1.0e-3, "maximum_iterations": 100}

    solver = IPOPTSolver(problem, parameters=parameters)
    rho_opt = solver.solve()

    File("output2/final_solution.pvd") << rho_opt
    xdmf_filename = XDMFFile(MPI.comm_world, "output2/final_solution.xdmf")
    xdmf_filename.write(rho_opt)