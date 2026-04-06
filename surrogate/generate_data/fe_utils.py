"""Finite element utilities for function evaluation and gradients."""

import dolfin as dl
import numpy as np


_cached_bbt = None


def get_snapshot(traj, t, sim_times, Vh):
    """
    Retrieve solution snapshot at time t.
    
    Parameters
    ----------
    traj : TimeDependentVector
        Trajectory object
    t : float
        Time to retrieve
    sim_times : np.ndarray
        Simulation times
    Vh : dolfin FunctionSpace
        Finite element space
        
    Returns
    -------
    dl.Vector
        Snapshot vector
    """
    idx = int(np.argmin(np.abs(sim_times - t)))
    t_closest = float(sim_times[idx])
    assert abs(t_closest - t) < 1e-10, \
        f"Time {t} not in simulation grid (nearest: {t_closest})"
    snap = dl.Function(Vh).vector()
    traj.retrieve(snap, t_closest)
    return snap


def eval_fn_and_grad_P1(func, mesh, pt_xy):
    """
    Evaluate a P1 function and its gradient at a point.
    
    Parameters
    ----------
    func : dolfin Function
        Function to evaluate (P1)
    mesh : dolfin Mesh
        Computational mesh
    pt_xy : (2,) array
        Point coordinates
        
    Returns
    -------
    tuple
        (value, gradient) where gradient is (2,) array
    """
    global _cached_bbt
    pt = dl.Point(float(pt_xy[0]), float(pt_xy[1]))
    try:
        val = func(pt)
    except RuntimeError:
        # Point is outside mesh (inside a building) — no measurement
        return 0.0, np.array([0.0, 0.0])

    if _cached_bbt is None:
        _cached_bbt = mesh.bounding_box_tree()
    cid = _cached_bbt.compute_first_entity_collision(pt)
    if cid >= mesh.num_cells():
        return float(val), np.zeros(2)

    cell = dl.Cell(mesh, cid)

    try:
        coords = cell.get_vertex_coordinates().reshape(-1, 2)
    except AttributeError:
        verts = cell.entities(0)
        all_coords = mesh.coordinates()
        coords = all_coords[verts]

    dofs = func.function_space().dofmap().cell_dofs(cid)
    u_nod = np.array(func.vector()[dofs])

    A = np.column_stack([np.ones(3), coords])
    try:
        abc = np.linalg.solve(A, u_nod)
    except np.linalg.LinAlgError:
        return float(val), np.zeros(2)
    return float(val), abc[1:]


def reset_cached_bbt():
    """Reset cached bounding box tree."""
    global _cached_bbt
    _cached_bbt = None
