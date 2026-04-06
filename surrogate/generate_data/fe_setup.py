"""Finite element setup functions."""

import dolfin as dl
import numpy as np
import sys
sys.path.append('../../../')
from hippylib import BiLaplacianPrior
from wind_utils import compute_velocity_field_navier_stokes


def create_mesh(nx=60, ny=60):
    """
    Create unit square mesh.
    
    Parameters
    ----------
    nx : int
        Number of elements in x-direction
    ny : int
        Number of elements in y-direction
        
    Returns
    -------
    dolfin Mesh
        Unit square mesh
    """
    return dl.UnitSquareMesh(nx, ny)


def setup_fe_spaces(mesh=None, nx=60, ny=60):
    """
    Set up finite element spaces and velocity field.
    
    Parameters
    ----------
    mesh : dolfin Mesh or None
        If None, create new mesh
    nx, ny : int
        Mesh resolution
        
    Returns
    -------
    tuple
        (mesh, Vh, wind_velocity)
    """
    if mesh is None:
        mesh = create_mesh(nx, ny)
    
    Vh = dl.FunctionSpace(mesh, "Lagrange", 1)
    wind_velocity = compute_velocity_field_navier_stokes(mesh)
    
    return mesh, Vh, wind_velocity


def setup_prior(Vh, gamma=1.0, delta=8.0, mean_value=0.25):
    """
    Set up BiLaplacian prior.
    
    Parameters
    ----------
    Vh : dolfin FunctionSpace
        Finite element space
    gamma, delta : float
        Prior parameters
    mean_value : float
        Constant mean value
        
    Returns
    -------
    BiLaplacianPrior
        Prior distribution
    """
    prior = BiLaplacianPrior(Vh, gamma, delta, robin_bc=True)
    prior.mean = dl.interpolate(dl.Constant(mean_value), Vh).vector()
    return prior


def setup_true_initial_condition(Vh):
    """
    Set up true initial condition for testing.
    
    Parameters
    ----------
    Vh : dolfin FunctionSpace
        Finite element space
        
    Returns
    -------
    dl.Vector
        True initial condition vector
    """
    ic_expr = dl.Expression(
        'std::min(0.5, std::exp(-100*(std::pow(x[0]-0.35,2)'
        '+std::pow(x[1]-0.7,2))))',
        element=Vh.ufl_element()
    )
    return dl.interpolate(ic_expr, Vh).vector()

def setup_fe_spaces_buildings(mesh_file='/home/fredkhouri/hippylib/applications/ad_diff/ad_20.xml'):
    """
    Set up FE spaces using the hIPPYlib ad_20 mesh with buildings.
    """
    mesh = dl.refine(dl.Mesh(mesh_file))
    Vh = dl.FunctionSpace(mesh, "Lagrange", 1)
    
    # Lid-driven cavity wind (same as hIPPYlib tutorial)
    Xh = dl.VectorFunctionSpace(mesh, 'Lagrange', 2)
    Wh = dl.FunctionSpace(mesh, 'Lagrange', 1)
    mixed_element = ufl.MixedElement([Xh.ufl_element(), Wh.ufl_element()])
    XW = dl.FunctionSpace(mesh, mixed_element)

    Re = dl.Constant(1e2)
    g = dl.Expression(('0.0','(x[0] < 1e-14) - (x[0] > 1 - 1e-14)'), degree=1)
    
    bc1 = dl.DirichletBC(XW.sub(0), g, v_boundary)
    bc2 = dl.DirichletBC(XW.sub(1), dl.Constant(0), q_boundary, 'pointwise')
    bcs = [bc1, bc2]

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

    wind_velocity = dl.project(v, Xh)
    return mesh, Vh, wind_velocity
