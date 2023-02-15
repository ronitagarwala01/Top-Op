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
p = Constant(5.0)                               # Penalization Factor for SIMP
q = Constant(0.5)                               # Relaxation Factor for Stress
eps = Constant(1.0e-3)                          # Epsilon value for SIMP to remove singularities
nu = Constant(0.3)                              # Poisson's Ratio
E = Constant(2.0e+11)                           # Young Modulus
lmbda = (E*nu) / ((1.0 + nu)*(1.0 - (2.0*nu)))  # Lame's first parameter
G = E / (2.0*(1.0 + nu))                        # Lame's second parameter / Shear Modulus
E_void = Constant(1.0e+100)                     # Young Modulus for Voids
lmbda_void=(E_void*nu)/((1.0+nu)*(1.0-(2.0*nu)))# Lame's first parameter for Voids
G_void = E_void / (2.0*(1.0 + nu))              # Lame's second parameter / Shear Modulus for Voids
nelx = 100                                      # Number of elements in x-direction
nely = 50                                       # Number of elements in y-direction
nelz = 25                                       # Number of elements in z-direction
C_max = Constant(50000.0)                       # Max Compliance
S_max = Constant(3.0e+9)                        # Max Stress
r = Constant(0.025)                             # Length Parameter for Helmholtz Filter
radius_1 = Constant(0.15)                        # Radius for first circle
x_1 = Constant(0.5)                             # X-coordinate for first circle
y_1 = Constant(0.5)                             # Y-coordinate for first circle
radius_2 = Constant(0.15)                        # Radius for second circle
x_2 = Constant(1.0)                             # X-coordinate for second circle
y_2 = Constant(0.5)                             # Y-coordinate for second circle
radius_3 = Constant(0.15)                        # Radius for third circle
x_3 = Constant(1.5)                             # X-coordinate for third circle
y_3 = Constant(0.5)                             # Y-coordinate for third circle

# Define Mesh
mesh = RectangleMesh(Point(0.0, 0.0), Point(L, W), nelx, nely)

circle_1 = CompiledSubDomain('(x[0]-x_1)*(x[0]-x_1) + (x[1]-y_1)*(x[1]-y_1) <= radius_1*radius_1 + tol',x_1=x_1, y_1=y_1, radius_1=radius_1, tol=DOLFIN_EPS)
circle_2 = CompiledSubDomain('(x[0]-x_2)*(x[0]-x_2) + (x[1]-y_2)*(x[1]-y_2) <= radius_2*radius_2 + tol',x_2=x_2, y_2=y_2, radius_2=radius_2, tol=DOLFIN_EPS)
circle_3 = CompiledSubDomain('(x[0]-x_3)*(x[0]-x_3) + (x[1]-y_3)*(x[1]-y_3) <= radius_3*radius_3 + tol',x_3=x_3, y_3=y_3, radius_3=radius_3, tol=DOLFIN_EPS)

# load_1 = CompiledSubDomain('(x[0]-x_1)*(x[0]-x_1) + (x[1]-y_1)*(x[1]-y_1) <= radius_1*radius_1 + tol',x_1=x_1, y_1=y_1, radius_1=0.01, tol=DOLFIN_EPS)

# Initialize mesh function for interior domains
domains = MeshFunction("size_t", mesh, mesh.topology().dim())
domains.set_all(0)
circle_1.mark(domains, 1)
circle_2.mark(domains, 1)
circle_3.mark(domains, 1)

# load_1.mark(domains, 0)

# class PointLoad(UserExpression):
#     def __init__(self, pt, vl, tol, **kwargs):
#         super().__init__(**kwargs)
#         self.point = pt
#         self.value = vl
#         self.tol = tol
#     def eval(self, values, x):
#         if near (x[0], self.point[0], self.tol) and near(x[1], self.point[1], self.tol):
#             values[0] = self.value[0]
#             values[1] = self.value[1]
#         else:
#             values[0] = 0
#             values[1] = 0
#     def value_shape(self):
#         return (2,)

# f1 = PointLoad(pt=(0.5, 0.5), vl=(0.0, 50000000.0), tol=1e-1, degree = 1)
# f2 = PointLoad(pt=(1.0, 0.5), vl=(0.0, 50000000.0), tol=1e-1, degree = 1)
# f3 = PointLoad(pt=(1.5, 0.5), vl=(0.0, 50000000.0), tol=1e-1, degree = 1)

# load_1 = CompiledSubDomain('near(x[0], x_1, tol) && near(x[1], y_1, tol)',x_1=x_1, y_1=y_1, tol=1.0e-2)
# # load_1 = CompiledSubDomain('x[0] == 25 && x[1] == 25')
# load_1.mark(domains, 2)

# # Class for adding point loads
# class PointLoad(UserExpression):
#     def __init__(self, domains, pt, vl, tol, **kwargs):
#         super().__init__(**kwargs)
#         self.domains = domains
#         self.point = pt
#         self.value = vl
#         self.tol = tol
#     def eval_cell(self, values, x, cell):
#         if self.domains[cell.index] == 2:
#             values[0] = self.value[0]
#             values[1] = self.value[1]
#         else:
#             values[0] = 0
#             values[1] = 0
#     def value_shape(self):
#         return (2,)

# f_1 = PointLoad(domains, pt=(x_1, y_1), vl=(0.0, 5000000.0), tol=1e-1,degree = 1)
# # Initialize mesh function for loads
# load_markers = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
# load_markers.set_all(0)

# load_2 = CompiledSubDomain('x[0] == x_2 && x[1] == y_2',x_2=x_2, y_2=y_2)
# load_2.mark(load_markers, 3)

# load_3 = CompiledSubDomain('x[0] == x_3 && x[1] == y_3',x_3=x_3, y_3=y_3)
# load_3.mark(load_markers, 4)

# class E(UserExpression): # UserExpression instead of Expression
#     def __init__(self, domains, **kwargs):
#         super().__init__(**kwargs)
#         self.domains = domains
#     def eval_cell(self, values, x, cell):
#         if self.domains[cell.index] == 0:
#             values[0] = Y 
#         elif self.domains[cell.index] == 1:
#             values[0] = Y#1.0e+100
#     def value_shape(self):
#         return ()     

# Define new measures associated with the interior domains
dx = Measure('dx', domain = mesh, subdomain_data = domains)
#ds = Measure('ds', domain = mesh, subdomain_data = load_markers)

# Define Function Spaces
U = VectorFunctionSpace(mesh, "Lagrange", 1) # Displacement Function Space
X = FunctionSpace(mesh, "Lagrange", 1)       # Density Function Space

# SIMP Function for Intermediate Density Penalization
def simp(x):
    return eps + (1 - eps) * x**p

# Calculate Strain from Displacements u
def strain(u):
    return 0.5*(nabla_grad(u) + nabla_grad(u).T)

# Calculate Cauchy Stress Matrix from Displacements u
def sigma(u):
    return lmbda*nabla_div(u)*Identity(2) + 2*G*strain(u)

# Calculate Cauchy Stress Matrix from Displacements u for Voids
def sigma_void(u):
    return lmbda_void*nabla_div(u)*Identity(2) + 2*G_void*strain(u)

# Calculate Von Mises Stress from Displacements u
def von_mises(u):
    s = sigma(u) - (1./3)*tr(sigma(u))*Identity(2)  # deviatoric stress
    return sqrt(3./2*inner(s, s))

# RELU^2 Function for Global Stress Constraint Computation
def relu(x):
    return (x * (ufl.sign(x) + 1) * 0.5) ** 2
    
# Volumetric Load
qv = 5000000.0
f = Constant((0.0, qv))

# Dirichlet BC em x[0] = 0
def Left_boundary(x, on_boundary):
    return on_boundary and abs(x[0]) < DOLFIN_EPS
u_L = Constant((0.0, 0.0))
bc = DirichletBC(U, u_L, Left_boundary)

# Helmholtz Filter for Design Variables. Helps remove checkerboarding and introduces mesh independency.
# Taken from work done by de Souza et al. (2020)
def helmholtz_filter(rho_n):
      V = rho_n.function_space()

      rho = TrialFunction(V)
      w = TestFunction(V)

      a = (r**2)*inner(grad(rho), grad(w))*dx(0) + rho*w*dx(0) + rho*w*dx(1)
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
    a = simp(rho)*inner(sigma(u), strain(v))*dx(0) + (inner(sigma_void(u), strain(v))*dx(1))
    L = dot(f, v)*dx(1)# + dot(f2, v)*dx + dot(f3, v)*dx# + dot(-f, v)*ds(3) + dot(f, v)*ds(4)
    u = Function(U)
    solve(a == L, u, bc)
    return u  

# MAIN
if __name__ == "__main__":
    x = interpolate(Constant(0.5), X)  # Initial value of 0.5 for each element's density
    # x.vector()[domains.where_equal(1)] = 1.0
    # print(domains.where_equal(1))
    # print(domains.where_equal(0))
    File("output2/domains.pvd") << domains
    # File("output2/loads.pvd") << load_markers
    # x.vector()[domains.where_equal(1)] = 0.001
    # print(len(x.vector()[domains.where_equal(1)]))
    u = forward(x)                     # Forward problem
    # v = TestFunction(U)
    # prod = project(inner(sigma(u), strain(u)), X)
    # print(len(prod.vector()[:]))

    # Objective Functional to be Minimized
    # Includes Regularization Term to Avoid Checkerboarding
    J = assemble(x*dx(0))
    m = Control(x)  # Control
    Jhat = ReducedFunctional(J, m) # Reduced Functional

    lb = 0.0  # Inferior
    ub = 1.0  # Superior

    # Class for Enforcing Compliance Constraint
    class ComplianceConstraint(InequalityConstraint):
        def __init__(self, C):
            self.C  = float(C)

        def function(self, m):
            c_current = assemble(dot(f, Control(u).tape_value())*dx) # Uses the value of u stored in the dolfin adjoint tape
            if MPI.rank(MPI.comm_world) == 0:
                print("Current compliance: ", c_current)
            
            return [self.C - c_current]

        def jacobian(self, m):
            J_comp = assemble(dot(f, u)*dx)
            m_comp = Control(x)
            dJ_comp = compute_gradient(J_comp, m_comp) # compute_gradient() uses the current value of u stored in the dolfin adjoint tape
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
        def __init__(self, S, q):
            self.S  = float(S)
            self.q = float(q)
            self.temp_x = Function(X)

        def function(self, m):
            self.temp_x.vector()[:] = m
            print("Current control vector (density): ", self.temp_x.vector()[:])
            s_current = assemble(relu((von_mises(Control(u).tape_value())*(Control(x).tape_value()**self.q))-self.S)*dx)
            if MPI.rank(MPI.comm_world) == 0:
                print("Current stress: ", s_current)

            return [-s_current]

        def jacobian(self, m):
            self.temp_x.vector()[:] = m
            J_stress = assemble(relu((von_mises(u)*(x**self.q))-self.S)*dx)
            m_stress = Control(x)
            dJ_stress = compute_gradient(J_stress, m_stress)
            
            return [-dJ_stress.vector()]

        def output_workspace(self):
            return [0.0]

        def length(self):
            """Return the number of components in the constraint vector (here, one)."""
            return 1

    class ExtensionConstraint(InequalityConstraint):

        "Ensure there is some mass above 1 over all points with x coordinate L"
        def __init__(self, _FunctionSpace, _L, _pa=8) -> None:
            super().__init__()
            self.L = _L
            self.pa = _pa
            self.FunctionSpace = _FunctionSpace
            self.Coordinates = _FunctionSpace.tabulate_dof_coordinates()
            # This provides the indices of the degrees of freedom in functions from X as they coincide with the canonical order of coordinates
            self.V2D = vertex_to_dof_map(self.FunctionSpace)

        def function(self, m):
            a = 0.0
            v = Function(self.FunctionSpace)
            v.vector()[:] = m
            for Coordinate in self.Coordinates:
                if Coordinate[0] == self.L:
                    a += v(Coordinate)**self.pa
            return [a-1]

        def jacobian(self, m):
            a = m.copy()
            for (Index, Coordinate) in zip(self.V2D, self.Coordinates):
                if Coordinate[0]== self.L:
                    a[Index] = a[Index]**(self.pa-1)*self.pa
                else:
                    a[Index] = 0.0

            return [a]

        def length(self):
            return 1

        def output_workspace(self):
            return [0.0]  

    problem = MinimizationProblem(Jhat, bounds=(lb, ub), constraints = [ComplianceConstraint(C_max)])#, ExtensionConstraint(X, L)])
    parameters = {"acceptable_tol": 1.0e-3, "maximum_iterations": 100}

    solver = IPOPTSolver(problem, parameters=parameters)
    rho_opt = solver.solve()

    File("output2/final_solution.pvd") << rho_opt
    xdmf_filename = XDMFFile(MPI.comm_world, "output2/final_solution.xdmf")
    xdmf_filename.write(rho_opt)