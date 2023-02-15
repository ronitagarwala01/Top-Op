from fenics import *
from fenics_adjoint import *

## Geometry and elasticity
t, h, L = 2., 1., 5.                          # Thickness, height and length
#Rectangle = Rectangle(Point(0,0),Point(L,h))  # Geometry
E, nu = 210*10**9, 0.3                        # Young Modulus
G = E/(2.0*(1.0 + nu))                        # Shear Modulus
lmbda = E*nu/((1.0 + nu)*(1.0 -2.0*nu))       # Lambda

## SIMP
def simp(x):
    return eps+(1-eps)*x**p

V = Constant(0.4*L*h)  # Volume constraint
p = Constant(4)      # Exponent
eps = Constant(1.0e-6) # Epsilon for SIMP

## Mesh, Control and Solution Spaces
nelx = 250
nely = 50
mesh = RectangleMesh(Point(0.0, 0.0), Point(L, h), nelx, nely)    
#mesh = generate_mesh(Rectangle, 100) # Mesh

A = VectorFunctionSpace(mesh, "Lagrange", 1) # Displacements
C = FunctionSpace(mesh, "Lagrange", 1)       # Control

## Volumetric Load
q = -10000000.0/t
b = Constant((0.0, q))

## Dirichlet BC em x[0] = 0
def Left_boundary(x, on_boundary):
    return on_boundary and abs(x[0]) < DOLFIN_EPS
u_L = Constant((0.0, 0.0))
bc = DirichletBC(A, u_L, Left_boundary)

## Forward problem solution
def forward(x):
    u = TrialFunction(A)  ## Trial and test functions
    w = TestFunction(A)
    sigma = lmbda*tr(sym(grad(u)))*Identity(2) + 2*G*sym(grad(u)) ## Stress
    F = simp(x)*inner(sigma, grad(w))*dx - dot(b, w)*dx
    a, L = lhs(F), rhs(F)
    u = Function(A)
    solve(a == L, u, bc)
    return u

## MAIN
if __name__ == "__main__":
    Vin = Constant(V/(L*h))
    x_in = interpolate(Vin, C)  # Initial value interpolated in V
    u = forward(x_in)           # Forward problem

    J = assemble(dot(b, u)*dx + Constant(1.0e-8)*dot(grad(x_in),grad(x_in))*dx)
    m = Control(x_in)               # Control
    Jhat = ReducedFunctional(J, m)  # Reduced Functional

    lb = 0.0  # Inferior
    ub = 1.0  # Superior

    class VolumeConstraint(InequalityConstraint):
        """A class that enforces the volume constraint g(a) = V - a*dx >= 0."""
        def __init__(self, V):
            self.V  = float(V)
            self.smass  = assemble(TestFunction(C) * Constant(1) * dx)
            self.tmpvec = Function(C)

        def function(self, m):
            self.tmpvec.vector()[:] = m
            integral = self.smass.inner(self.tmpvec.vector())
            if MPI.rank(MPI.comm_world) == 0:
                print("Current control integral: ", integral)
            return [self.V - integral]

        def jacobian(self, m):
            return [-self.smass]

        def output_workspace(self):
            return [0.0]

        def length(self):
            """Return the number of components in the constraint vector (here, one)."""
            return 1

    problem = MinimizationProblem(Jhat, bounds=(lb, ub), constraints=VolumeConstraint(V))
    parameters = {"acceptable_tol": 1.0e-3, "maximum_iterations": 100}

    solver = IPOPTSolver(problem, parameters=parameters)
    rho_opt = solver.solve()

    File("output1/final_solution.pvd") << rho_opt
    xdmf_filename = XDMFFile(MPI.comm_world, "output1/final_solution.xdmf")
    xdmf_filename.write(rho_opt)
