def v_boundary(x, on_boundary):
    return on_boundary

def q_boundary(x, on_boundary):
    return x[0] < dl.DOLFIN_EPS and x[1] < dl.DOLFIN_EPS

def computeVelocityField(mesh):
    Xh = dl.VectorFunctionSpace(mesh, 'Lagrange', 2)
    Wh = dl.FunctionSpace(mesh, 'Lagrange', 1)
    mixed_element = ufl.MixedElement([Xh.ufl_element(), Wh.ufl_element()])
    XW = dl.FunctionSpace(mesh, mixed_element)

    Re = dl.Constant(100.0)

    g = dl.Expression(('4.0*x[1]*(1.0-x[1])', '0.0'), degree=2)
    
    bc1 = dl.DirichletBC(XW.sub(0), g, "near(x[0], 0.0)")
    bc2 = dl.DirichletBC(XW.sub(0), dl.Constant((0.0, 0.0)), "near(x[1], 0.0) || near(x[1], 1.0)")
    bc3 = dl.DirichletBC(XW.sub(1), dl.Constant(0), q_boundary, 'pointwise')
    
    bcs = [bc1, bc2, bc3]

    vq = dl.Function(XW)
    (v, q) = ufl.split(vq)
    (v_test, q_test) = dl.TestFunctions(XW)

    def strain(v):
        return ufl.sym(ufl.grad(v))

    F = ( (2./Re)*ufl.inner(strain(v), strain(v_test)) 
          + ufl.inner(ufl.nabla_grad(v)*v, v_test)
          - (q * ufl.div(v_test)) 
          + (ufl.div(v) * q_test) ) * ufl.dx

    dl.solve(F == 0, vq, bcs, solver_parameters={"newton_solver":
                                         {"relative_tolerance":1e-4, "maximum_iterations":100}})

    vh = dl.project(v, Xh)
    qh = dl.project(q, Wh)
    
    plt.figure(figsize=(15, 5))
    
    mesh_coarse = dl.UnitSquareMesh(60, 60)
    vh_coarse = dl.interpolate(vh, dl.VectorFunctionSpace(mesh_coarse, "CG", 1))
    
    nb.plot(vh_coarse, subplot_loc=121, mytitle="Velocity (Parabolic)")
    nb.plot(qh, subplot_loc=122, mytitle="Pressure")
    plt.show()

    return vh
