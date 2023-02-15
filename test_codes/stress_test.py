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
p = Constant(4)                                 # Penalization Factor for SIMP
alpha = Constant(0.9)                           # Mass Penalization Constant
eps = Constant(1.0e-3)                          # Epsilon value for SIMP to remove singularities
E = Constant(2.0e+11)                           # Young Modulus
nu = Constant(0.3)                              # Poisson's Ratio
lmbda = (E*nu) / ((1.0 + nu)*(1.0 - (2.0*nu)))  # Lame's first parameter
G = E / (2.0*(1.0 + nu))                        # Lame's second parameter / Shear Modulus
nelx = 100                                      # Number of elements in x-direction
nely = 50                                       # Number of elements in y-direction
C_max = Constant(50000.0)                       # Max Compliance
S_max = Constant(4.5e+8)                        # Max Stress
p_norm = Constant(1.0)

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
# def relu(x):
#     return np.maximum(0.0, x)
def relu(x):
       return x*(ufl.sign(x)+1)*0.5

# Computes the Gradient of the RELU Function for Stress Constraint Jacobian Computation
def d_relu(x):
    return np.greater(x, 0.0).astype(np.float32)
    
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
    # print(x.vector()[:])
    # x_pow = x.vector()[:]**0.5
    # print(x_pow)
    # #print(relu(von_mises(u)))
    # func = project(von_mises(u), X)
    # print(func.vector()[:])
    # func = project(von_mises(u)*(x**0.5), X)
    # print(func.vector()[:])
    # der=assemble(relu(von_mises(u)-S_max)*dx)
    # print(der)



    #vm = project(von_mises(u), X)
    #print(vm.vector()[:])
    # diff = project((von_mises(u) - S_max), X)
    # print(diff.vector()[:])
    # relu_diff = Function(X)
    # relu_diff.vector()[:] = relu(diff.vector()[:])
    # print(relu_diff.vector()[:])
    # print(relu_diff.vector().sum())
    # print(assemble(relu_diff*dx))
    #relu_diff_func = Function(X)
    #relu_diff_func.vector().set_local(relu_diff)
    #print(np.sum(relu_diff))
    #print(relu_diff_func)
    #print(relu_diff_func.vector()[:])
    #print(assemble(relu_diff_func * dx))
    #print(relu_diff_func.vector().sum())
    #print(type(relu_diff_func.vector().sum()))
    #func = assemble(TestFunction(X) * Constant(1) * dx)
    #summ = func.inner(relu_diff_func.vector())
    #print(summ)
    #print(f.vector()[:])
    #print(len(u.vector()[:]))
    #print()
    # der_relu_diff = d_relu(diff.vector()[:])
    # print(der_relu_diff)
    # der_relu_func = Function(X)
    # der_relu_func.vector().set_local(der_relu_diff)
    # print(der_relu_func)
    # print(der_relu_func.vector()[:])
    # print(relu(vm.vector()[:] - S_max))

    def eval_cb(j, dj, x):
        u = forward(x)
        print("Objective (Mass): ", j)
        print("Compliance: ", assemble(dot(f,u)*dx))
        print("Derivative: ", dj.vector()[:])
        print("Mass: ", assemble(x*dx))
        # vm = project(von_mises(u), X)
        # print(vm.vector()[:])
        # s_max_func = project(S_max, X)
        # print(s_max_func.vector()[:])
        # diff = project(vm - s_max_func, X)
        # print(diff.vector()[:])
        # ans = compute_gradient(assemble(diff.vector().sum()), x)
        # print(ans.vector()[:])

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
                # print("Current Compliance 2: ", np.sum(project(self.tmpvec ** p * psi(self.u), X).vector()[:])*0.0004)

            return [self.C - c_current]

        def jacobian(self, m):
            J_comp = assemble(dot(f, u)*dx)
            print("J_Comp: ", type(J_comp))
            m_comp = Control(x)
            dJ_comp = compute_gradient(J_comp, m_comp)
            print("Computed Derivative: ", dJ_comp.vector()[:])
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
            self.u = interpolate(u, U)

        def function(self, m):
            self.tmpvec.vector()[:]=m
            print("Current control vector (density): ", self.tmpvec.vector()[:])
            self.u = forward(self.tmpvec)
            print("Current u vector: ", self.u.vector()[:])
            s_current = assemble(relu((von_mises(self.u)*(self.tmpvec**0.5))-self.S)*dx)
            if MPI.rank(MPI.comm_world) == 0:
                print("Current stress: ", s_current)
                # print("Current Compliance 2: ", np.sum(project(self.tmpvec ** p * psi(self.u), X).vector()[:])*0.0004)

            return [-s_current]

            # diff = project(von_mises(self.u) - self.S, X)
            # relu_diff = Function(X)
            # relu_diff.vector()[:] = relu(diff.vector()[:])
            # s_current = assemble(relu_diff*dx)
            # if MPI.rank(MPI.comm_world) == 0:
            #     print("Current Stress Constraint: ", s_current)

            # return [-s_current]

        def jacobian(self, m):
            J_stress = assemble(relu((von_mises(u)*(x**0.5))-self.S)*dx)
            print("J_Stress: ", type(J_stress))
            m_stress = Control(x)
            dJ_stress = compute_gradient(J_stress, m_stress)
            print("Computed Derivative: ", dJ_stress.vector()[:])
            
            return [-dJ_stress.vector()]
            # dif = project(von_mises(u) - self.S, X)
            # relu_diff = Function(X)
            # relu_diff.vector()[:] = relu(dif.vector()[:])
            # d_relu_diff = Function(X)
            # d_relu_diff.vector()[:] = d_relu(dif.vector()[:])
            # J_stress = assemble(relu(dif)*dx)
            # print("J_stress: ", J_stress)
            # m_stress = Control(x)
            # dJ_stress = compute_gradient(J_stress, m_stress)
            # jac_stress = Function(X)
            # jac_stress.vector()[:] = np.multiply(dJ_stress.vector()[:], d_relu_diff.vector()[:])
            # print(type(diff))
            # print(type(m_stress))
            # J_stress = assemble(dot(diff, Constant(1.0)) * dx)
            # dJ_stress = dot(der_relu_func, compute_gradient(J_stress, m_stress)) #Get vector of compute_gradient, do element-wise mult
            # print("Computed Derivative Length: ", len(dJ_stress.vector()[:]))
            # print("Computed Derivative: ", d_relu_diff.vector()[:])
            # print("Computed Derivative: ", np.multiply(dJ_stress.vector()[:], d_relu_diff.vector()[:]))

            # return [dJ_stress.vector()]

        def output_workspace(self):
            return [0.0]

        def length(self):
            """Return the number of components in the constraint vector (here, one)."""
            return 1  

    # # Class for Enforcing Stress Constraint
    # class StressConstraint(InequalityConstraint):
    #     def __init__(self, S):
    #         self.S  = float(S)
    #         self.tmpvec = Function(X)
    #         self.u = interpolate(u, U)

    #     def function(self, m):
    #         self.tmpvec.vector()[:]=m
    #         print("Current control vector (density): ", self.tmpvec.vector()[:])
    #         self.u = forward(self.tmpvec)
    #         print("Current u vector: ", self.u.vector()[:])
    #         integral = assemble(((von_mises(self.u)/self.S)**p_norm)*dx)
    #         s_current = 1.0 - (integral ** (1/p_norm))
    #         if MPI.rank(MPI.comm_world) == 0:
    #             print("Current stress integral: ", integral)
    #             print("Current stress constraint: ", s_current)
    #             # print("Current Compliance 2: ", np.sum(project(self.tmpvec ** p * psi(self.u), X).vector()[:])*0.0004)

    #         return [s_current]

    #     def jacobian(self, m):
    #         J_stress = assemble(((von_mises(u)/self.S)**p_norm)*dx)
    #         print("J_Stress: ", J_stress)
    #         m_stress = Control(x)
    #         dJ_stress = compute_gradient(J_stress, m_stress)

    #         print("Computed Derivative: ", type(dJ_stress))
    #         return [dJ_stress]

    #     def output_workspace(self):
    #         return [0.0]

    #     def length(self):
    #         """Return the number of components in the constraint vector (here, one)."""
    #         return 1

    #print(psi(u).vector()[:])

    #stress_constraint = UFLInequalityConstraint(S_max-dot(f, u)*dx + Constant(1.0e-8)*dot(grad(x),grad(x))*dx, m)

    problem = MinimizationProblem(Jhat, bounds=(lb, ub), constraints =  StressConstraint(S_max))
    parameters = {"acceptable_tol": 1.0e-1, "maximum_iterations": 100}

    solver = IPOPTSolver(problem, parameters=parameters)
    rho_opt = solver.solve()

    File("output2/final_solution.pvd") << rho_opt
    xdmf_filename = XDMFFile(MPI.comm_world, "output2/final_solution.xdmf")
    xdmf_filename.write(rho_opt)
