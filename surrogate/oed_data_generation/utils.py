"""Utility functions for OED data generation."""

import numpy as np
import dolfin as dl
import ufl, sys


sys.path.append('../../../')
from hippylib.hippylib.modeling.prior import BiLaplacianPrior

def compute_velocity_field(mesh):
    """Compute base velocity field (from original code)."""
    Xh = dl.VectorFunctionSpace(mesh, 'Lagrange', 2)
    Wh = dl.FunctionSpace(mesh, 'Lagrange', 1)
    mixed_element = ufl.MixedElement([Xh.ufl_element(), Wh.ufl_element()])
    XW = dl.FunctionSpace(mesh, mixed_element)

    Re = dl.Constant(100.0)
    g  = dl.Expression(('4.0*x[1]*(1.0-x[1])', '0.0'), degree=2)

    bc1 = dl.DirichletBC(XW.sub(0), g, "near(x[0], 0.0)")
    bc2 = dl.DirichletBC(XW.sub(0), dl.Constant((0.0, 0.0)),
                         "near(x[1], 0.0) || near(x[1], 1.0)")
    
    def q_boundary(x, on_boundary):
        return x[0] < dl.DOLFIN_EPS and x[1] < dl.DOLFIN_EPS
    
    bc3 = dl.DirichletBC(XW.sub(1), dl.Constant(0), q_boundary, 'pointwise')
    bcs = [bc1, bc2, bc3]

    vq = dl.Function(XW)
    (v, q) = ufl.split(vq)
    (v_test, q_test) = dl.TestFunctions(XW)

    def strain(v):
        return ufl.sym(ufl.grad(v))

    F = ((2./Re)*ufl.inner(strain(v), strain(v_test))
         + ufl.inner(ufl.nabla_grad(v)*v, v_test)
         - q*ufl.div(v_test)
         + ufl.div(v)*q_test) * ufl.dx

    dl.solve(F == 0, vq, bcs,
             solver_parameters={"newton_solver":
                                 {"relative_tolerance": 1e-4,
                                  "maximum_iterations": 100}})

    vh = dl.project(v, Xh)
    return vh


def initialize_problem(nx, ny, gamma, delta):
    """Initialize Vh, prior, true IC, and wind velocity."""
    mesh = dl.UnitSquareMesh(nx, ny)
    
    # Compute wind velocity
    wind_velocity = compute_velocity_field(mesh)
    
    # Function space
    Vh = dl.FunctionSpace(mesh, "Lagrange", 1)
    
    # True initial condition
    ic_expr = dl.Expression(
        'std::min(0.5, std::exp(-100*(std::pow(x[0]-0.35,2)'
        '+std::pow(x[1]-0.7,2))))',
        element=Vh.ufl_element())
    true_initial_condition = dl.interpolate(ic_expr, Vh).vector()
    
    # Prior
    prior = BiLaplacianPrior(Vh, gamma, delta, robin_bc=True)
    prior.mean = dl.interpolate(dl.Constant(0.25), Vh).vector()
    
    return Vh, prior, true_initial_condition, wind_velocity




def compute_velocity_field(mesh):
    """Compute base velocity field (from original code)."""
    Xh = dl.VectorFunctionSpace(mesh, 'Lagrange', 2)
    Wh = dl.FunctionSpace(mesh, 'Lagrange', 1)
    mixed_element = ufl.MixedElement([Xh.ufl_element(), Wh.ufl_element()])
    XW = dl.FunctionSpace(mesh, mixed_element)

    Re = dl.Constant(100.0)
    g  = dl.Expression(('4.0*x[1]*(1.0-x[1])', '0.0'), degree=2)

    bc1 = dl.DirichletBC(XW.sub(0), g, "near(x[0], 0.0)")
    bc2 = dl.DirichletBC(XW.sub(0), dl.Constant((0.0, 0.0)),
                         "near(x[1], 0.0) || near(x[1], 1.0)")
    
    def q_boundary(x, on_boundary):
        return x[0] < dl.DOLFIN_EPS and x[1] < dl.DOLFIN_EPS
    
    bc3 = dl.DirichletBC(XW.sub(1), dl.Constant(0), q_boundary, 'pointwise')
    bcs = [bc1, bc2, bc3]

    vq = dl.Function(XW)
    (v, q)           = ufl.split(vq)
    (v_test, q_test) = dl.TestFunctions(XW)

    def strain(v):
        return ufl.sym(ufl.grad(v))

    F = ((2./Re)*ufl.inner(strain(v), strain(v_test))
         + ufl.inner(ufl.nabla_grad(v)*v, v_test)
         - q*ufl.div(v_test)
         + ufl.div(v)*q_test) * ufl.dx

    dl.solve(F == 0, vq, bcs,
             solver_parameters={"newton_solver":
                                 {"relative_tolerance": 1e-4,
                                  "maximum_iterations": 100}})

    vh = dl.project(v, Xh)
    return vh


def initialize_problem(nx, ny, gamma, delta):
    """Initialize Vh, prior, true IC, and wind velocity."""
    mesh = dl.UnitSquareMesh(nx, ny)
    
    # Compute wind velocity
    wind_velocity = compute_velocity_field(mesh)
    
    # Function space
    Vh = dl.FunctionSpace(mesh, "Lagrange", 1)
    
    # True initial condition
    ic_expr = dl.Expression(
        'std::min(0.5, std::exp(-100*(std::pow(x[0]-0.35,2)'
        '+std::pow(x[1]-0.7,2))))',
        element=Vh.ufl_element())
    true_initial_condition = dl.interpolate(ic_expr, Vh).vector()
    
    # Prior
    prior = BiLaplacianPrior(Vh, gamma, delta, robin_bc=True)
    prior.mean = dl.interpolate(dl.Constant(0.25), Vh).vector()
    
    return Vh, prior, true_initial_condition, wind_velocity