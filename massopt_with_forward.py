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
x_1 = Constant(0.6)                             # X-coordinate for first circle
y_1 = Constant(0.75)                            # Y-coordinate for first circle
radius_2 = Constant(0.15)                       # Radius for second circle
x_2 = Constant(1.0)                             # X-coordinate for second circle
y_2 = Constant(0.25)                            # Y-coordinate for second circle
radius_3 = Constant(0.15)                       # Radius for third circle
x_3 = Constant(1.4)                             # X-coordinate for third circle
y_3 = Constant(0.75)                            # Y-coordinate for third circle

# Define Mesh
mesh = RectangleMesh(Point(0.0, 0.0), Point(L, W), nelx, nely)

circle_1 = CompiledSubDomain('(x[0]-x_1)*(x[0]-x_1) + (x[1]-y_1)*(x[1]-y_1) <= radius_1*radius_1 + tol', x_1=x_1, y_1=y_1, radius_1=radius_1, tol=DOLFIN_EPS)
circle_2 = CompiledSubDomain('(x[0]-x_2)*(x[0]-x_2) + (x[1]-y_2)*(x[1]-y_2) <= radius_2*radius_2 + tol', x_2=x_2, y_2=y_2, radius_2=radius_2, tol=DOLFIN_EPS)
circle_3 = CompiledSubDomain('(x[0]-x_3)*(x[0]-x_3) + (x[1]-y_3)*(x[1]-y_3) <= radius_3*radius_3 + tol', x_3=x_3, y_3=y_3, radius_3=radius_3, tol=DOLFIN_EPS)

boundary_1 = CompiledSubDomain('(x[0]-x_1)*(x[0]-x_1) + (x[1]-y_1)*(x[1]-y_1) <= (radius_1+0.030)*(radius_1+0.030) + tol', x_1=x_1, y_1=y_1, radius_1=radius_1, tol=DOLFIN_EPS)
boundary_2 = CompiledSubDomain('(x[0]-x_2)*(x[0]-x_2) + (x[1]-y_2)*(x[1]-y_2) <= (radius_2+0.030)*(radius_2+0.030) + tol', x_2=x_2, y_2=y_2, radius_2=radius_2, tol=DOLFIN_EPS)
boundary_3 = CompiledSubDomain('(x[0]-x_3)*(x[0]-x_3) + (x[1]-y_3)*(x[1]-y_3) <= (radius_3+0.030)*(radius_3+0.030) + tol', x_3=x_3, y_3=y_3, radius_3=radius_3, tol=DOLFIN_EPS)

# Initialize mesh function for interior domains
domains = MeshFunction("size_t", mesh, mesh.topology().dim())
domains.set_all(0)
boundary_1.mark(domains, 4)
boundary_2.mark(domains, 4)
boundary_3.mark(domains, 4)
circle_1.mark(domains, 1)
circle_2.mark(domains, 2)
circle_3.mark(domains, 3)
 
# Define new measures associated with the interior domains
dx = Measure('dx', domain = mesh, subdomain_data = domains)

# Define Function Spaces
U = VectorFunctionSpace(mesh, "Lagrange", 1) # Displacement Function Space
X = FunctionSpace(mesh, "Lagrange", 1)       # Density Function Space

# Creating Function for Youngs Modulus across Mesh
Q = FunctionSpace(mesh, "DG", 0)
E = Function(Q)
x_ = Q.tabulate_dof_coordinates()

for i in range(x_.shape[0]):
    if domains.array()[i] == 1 or domains.array()[i] == 2 or domains.array()[i] == 3:
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
    u = forward(x)                      # Forward problem
    vm = von_mises(u)                   # Calculate Von Mises Stress for outer subdomain

    #print("Max Von Mises = ", max(vm.vector()[:]))
    File("output2/domains.pvd") << domains

    controls = File("output2/control_iterations.pvd")
    x_viz = Function(X, name="ControlVisualisation")

    def derivative_cb(j, dj, m):
        x_viz.assign(m)
        controls << x_viz
        
    # Objective Functional to be Minimized
    J = assemble(x*dx(0))
    m = Control(x)  # Control
    Jhat = ReducedFunctional(J, m, ) # Reduced Functional

    lb = 0.0  # Inferior
    ub = 1.0  # Superior

    # Class for Enforcing Compliance Constraint
    class ComplianceConstraint(InequalityConstraint):
        def __init__(self, C):
            self.C = float(C)
            self.u = Function(U)
            self.temp_x = Function(X)

        def function(self, m):
            self.temp_x.vector()[:] = m
            self.u = forward(self.temp_x)
            c_current = assemble(dot(f, self.u)*dx) # Uses the value of u stored in the dolfin adjoint tape
            if MPI.rank(MPI.comm_world) == 0:
                print("Current compliance: ", c_current)
            
            return [self.C - c_current]

        def jacobian(self, m):
            self.temp_x.vector()[:] = m
            self.u = forward(self.temp_x)
            J_comp = assemble(dot(f,self.u)*dx)
            m_comp = Control(self.temp_x)
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
        def __init__(self, S, q, p_norm):
            self.S  = float(S)
            self.q = float(q)
            self.p_norm = float(p_norm)
            self.temp_x = Function(X)
            self.u = Function(U)

        def function(self, m):
            self.temp_x.vector()[:] = m
            print("Current control vector (density): ", self.temp_x.vector()[:])
            self.u = forward(self.temp_x)
            integral = assemble((((von_mises(self.u)*(self.temp_x**self.q))/self.S)**self.p_norm)*dx)
            s_current = 1.0 - (integral ** (1.0/self.p_norm))
            if MPI.rank(MPI.comm_world) == 0:
                print("Current stress integral: ", integral)
                print("Current stress constraint: ", s_current)

            return [s_current]

        def jacobian(self, m):
            temp_x = Function(X)
            temp_x.vector()[:] = m
            self.u = forward(temp_x)
            J_stress = assemble(((((von_mises(self.u))*(temp_x**self.q))/self.S)**self.p_norm)*dx)
            print("J_Stress: ", J_stress)
            m_stress = Control(temp_x)
            dJ_stress = compute_gradient(J_stress, m_stress)
            print("Derivative: ", dJ_stress.vector()[:])
            dJ_stress.vector()[:] = np.multiply((1.0/self.p_norm) * np.power(J_stress, ((1.0/self.p_norm)-1.0)), dJ_stress.vector()[:])
            print("Computed Derivative: ", -dJ_stress.vector()[:])
            return [-dJ_stress.vector()]

        def output_workspace(self):
            return [0.0]

        def length(self):
            """Return the number of components in the constraint vector (here, one)."""
            return 1

    # # Class for Enforcing Stress Constraint
    # class StressConstraint(InequalityConstraint):
    #     def __init__(self, S, q):
    #         self.S  = float(S)
    #         self.q = float(q)
    #         self.temp_x = Function(X)

    #     def function(self, m):
    #         self.temp_x.vector()[:] = m
    #         print("Current control vector (density): ", self.temp_x.vector()[:])
    #         s_current = assemble(relu(Control(vm).tape_value()*(Control(x).tape_value()**self.q)-self.S)*dx)
    #         if MPI.rank(MPI.comm_world) == 0:
    #             print("Current stress: ", s_current)

    #         return [-s_current]

    #     def jacobian(self, m):
    #         self.temp_x.vector()[:] = m
    #         J_stress = assemble(relu(vm*(x**self.q)-self.S)*dx)
    #         m_stress = Control(x)
    #         dJ_stress = compute_gradient(J_stress, m_stress)
            
    #         return [-dJ_stress.vector()]

    #     def output_workspace(self):
    #         return [0.0]

    #     def length(self):
    #         """Return the number of components in the constraint vector (here, one)."""
    #         return 1


    problem = MinimizationProblem(Jhat, bounds=(lb, ub), constraints = [ComplianceConstraint(C_max), StressConstraint(S_max, q, p_norm)])
    parameters = {"acceptable_tol": 1.0e-3, "maximum_iterations": 300}

    solver = IPOPTSolver(problem, parameters=parameters)
    rho_opt = solver.solve()
    u_opt = forward(rho_opt)
    vm_opt = von_mises(u_opt)

    File("output2/final_solution.pvd") << rho_opt
    File("output2/von_mises.pvd") << vm
    xdmf_filename = XDMFFile(MPI.comm_world, "output2/final_solution.xdmf")
    xdmf_filename.write(rho_opt)
