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