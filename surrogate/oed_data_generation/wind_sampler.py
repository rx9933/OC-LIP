"""Spectral wind field sampling."""

import numpy as np
import dolfin as dl


def sample_spectral_wind(mesh, r_wind=3, sigma=1.0, alpha=2.0,
                         mean_vx=0.5, mean_vy=0.0, seed=None):
    """
    Sample a wind field from the spectral prior.
    
    u(x,y) = mean_vx + Σ a_ij cos(iπy) cos(jπx)
    v(x,y) = mean_vy + Σ b_ij sin(iπy) cos(jπx)   [sin ensures v=0 at y=0,1]
    
    Parameters
    ----------
    mesh     : dolfin mesh
    r_wind   : int, number of spectral modes per direction
    sigma    : float, overall amplitude scale
    alpha    : float, spectral decay rate (higher = smoother)
    mean_vx  : float, mean horizontal wind
    mean_vy  : float, mean vertical wind (should be 0 for wall BCs)
    seed     : int or None, random seed
    
    Returns
    -------
    v_func   : dolfin Function on VectorFunctionSpace
    coeffs   : dict with 'a_ij', 'b_ij', 'mean_vx', 'mean_vy',
                'r_wind', 'sigma', 'alpha'
    """
    if seed is not None:
        np.random.seed(seed)
    
    Lx, Ly = 1.0, 1.0  # unit square
    
    # Draw random coefficients
    a_ij = np.zeros((r_wind, r_wind))
    b_ij = np.zeros((r_wind, r_wind))
    for i in range(r_wind):
        for j in range(r_wind):
            # i+1, j+1 because modes start at 1 not 0
            mode_i = i + 1
            mode_j = j + 1
            variance = sigma**2 / (mode_i**2 + mode_j**2)**alpha
            a_ij[i, j] = np.sqrt(variance) * np.random.randn()
            b_ij[i, j] = np.sqrt(variance) * np.random.randn()
    
    # Evaluate on mesh coordinates
    Xh = dl.VectorFunctionSpace(mesh, 'Lagrange', 1)
    xy = mesh.coordinates()
    x = xy[:, 0]
    y = xy[:, 1]
    
    # Build velocity field
    vx = np.full(len(xy), mean_vx)
    vy = np.full(len(xy), mean_vy)
    
    for i in range(r_wind):
        for j in range(r_wind):
            mode_i = i + 1
            mode_j = j + 1
            vx += a_ij[i, j] * np.cos(mode_i * np.pi * y / Ly) * np.cos(mode_j * np.pi * x / Lx)
            vy += b_ij[i, j] * np.sin(mode_i * np.pi * y / Ly) * np.cos(mode_j * np.pi * x / Lx)
    
    # Build dolfin function
    Vh_scalar = dl.FunctionSpace(mesh, 'Lagrange', 1)
    
    vx_func = dl.Function(Vh_scalar)
    vy_func = dl.Function(Vh_scalar)
    
    v2d = dl.vertex_to_dof_map(Vh_scalar)
    
    vx_vals = vx_func.vector().get_local()
    vy_vals = vy_func.vector().get_local()
    for i in range(len(xy)):
        vx_vals[v2d[i]] = vx[i]
        vy_vals[v2d[i]] = vy[i]
    vx_func.vector().set_local(vx_vals)
    vy_func.vector().set_local(vy_vals)
    
    v_func = dl.Function(Xh)
    fa = dl.FunctionAssigner(Xh, [Vh_scalar, Vh_scalar])
    fa.assign(v_func, [vx_func, vy_func])
    
    # Safety: cap maximum velocity
    max_speed = np.sqrt(vx**2 + vy**2).max()
    safe_max = 2.0
    if max_speed > safe_max:
        scale_factor = safe_max / max_speed
        print(f"  Rescaling wind: max speed {max_speed:.2f} → {safe_max:.2f}")
        v_func.vector()[:] *= scale_factor
        a_ij *= scale_factor
        b_ij *= scale_factor
        mean_vx *= scale_factor
        mean_vy *= scale_factor
    
    coeffs = {
        'a_ij': a_ij.copy(),
        'b_ij': b_ij.copy(),
        'mean_vx': mean_vx,
        'mean_vy': mean_vy,
        'r_wind': r_wind,
        'sigma': sigma,
        'alpha': alpha,
    }
    
    return v_func, coeffs


def coeffs_to_nn_input(coeffs, c_init):
    """
    Flatten spectral coefficients + drone position into a single NN input vector.
    
    Layout: [mean_vx, mean_vy, a_11, a_12, ..., a_rr, b_11, b_12, ..., b_rr, cx, cy]
    """
    a_flat = coeffs['a_ij'].flatten()
    b_flat = coeffs['b_ij'].flatten()
    return np.concatenate([
        [coeffs['mean_vx'], coeffs['mean_vy']],
        a_flat,
        b_flat,
        c_init
    ])


def nn_input_dim(r_wind):
    """Dimension of the NN input vector."""
    return 2 + 2 * r_wind**2 + 2