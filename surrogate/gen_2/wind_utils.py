"""Spectral wind field generation utilities."""

import dolfin as dl
import numpy as np
import ufl


def q_boundary(x, on_boundary):
    """Boundary condition for pressure."""
    return x[0] < dl.DOLFIN_EPS and x[1] < dl.DOLFIN_EPS


def compute_velocity_field_navier_stokes(mesh, direction='east', speed=4.0, Re_val=100.0):
    """
    Compute velocity field by solving Navier-Stokes.
    
    Parameters
    ----------
    mesh : dolfin Mesh
        Computational mesh
    direction : str
        'east', 'west', 'north', 'south'
    speed : float
        Peak inlet velocity
    Re_val : float
        Reynolds number
        
    Returns
    -------
    dolfin Function
        Velocity field
    """
    Xh = dl.VectorFunctionSpace(mesh, 'Lagrange', 2)
    Wh = dl.FunctionSpace(mesh, 'Lagrange', 1)
    mixed_element = ufl.MixedElement([Xh.ufl_element(), Wh.ufl_element()])
    XW = dl.FunctionSpace(mesh, mixed_element)

    Re = dl.Constant(Re_val)
    bcs = []

    if direction == 'east':
        g = dl.Expression(('s*x[1]*(1.0-x[1])', '0.0'), degree=2, s=speed)
        bcs.append(dl.DirichletBC(XW.sub(0), g, "near(x[0], 0.0)"))
        bcs.append(dl.DirichletBC(XW.sub(0), dl.Constant((0.0, 0.0)),
                   "near(x[1], 0.0) || near(x[1], 1.0)"))

    elif direction == 'west':
        g = dl.Expression(('-s*x[1]*(1.0-x[1])', '0.0'), degree=2, s=speed)
        bcs.append(dl.DirichletBC(XW.sub(0), g, "near(x[0], 1.0)"))
        bcs.append(dl.DirichletBC(XW.sub(0), dl.Constant((0.0, 0.0)),
                   "near(x[1], 0.0) || near(x[1], 1.0)"))

    elif direction == 'north':
        g = dl.Expression(('0.0', 's*x[0]*(1.0-x[0])'), degree=2, s=speed)
        bcs.append(dl.DirichletBC(XW.sub(0), g, "near(x[1], 0.0)"))
        bcs.append(dl.DirichletBC(XW.sub(0), dl.Constant((0.0, 0.0)),
                   "near(x[0], 0.0) || near(x[0], 1.0)"))

    elif direction == 'south':
        g = dl.Expression(('0.0', '-s*x[0]*(1.0-x[0])'), degree=2, s=speed)
        bcs.append(dl.DirichletBC(XW.sub(0), g, "near(x[1], 1.0)"))
        bcs.append(dl.DirichletBC(XW.sub(0), dl.Constant((0.0, 0.0)),
                   "near(x[0], 0.0) || near(x[0], 1.0)"))

    bcs.append(dl.DirichletBC(XW.sub(1), dl.Constant(0), q_boundary, 'pointwise'))

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


def sample_spectral_wind(mesh, r_wind=3, sigma=1.0, alpha=2.0,
                         mean_vx=0.5, mean_vy=0.0, seed=None, safe_max=2.0):
    """
    Sample a wind field from the spectral prior.
    
    u(x,y) = mean_vx + Σ a_ij cos(iπy) cos(jπx)
    v(x,y) = mean_vy + Σ b_ij sin(iπy) cos(jπx)   [sin ensures v=0 at y=0,1]
    
    Coefficients: a_ij, b_ij ~ N(0, σ²/(i²+j²)^α)
    
    Parameters
    ----------
    mesh : dolfin Mesh
        Computational mesh
    r_wind : int
        Number of spectral modes per direction
    sigma : float
        Overall amplitude scale
    alpha : float
        Spectral decay rate (higher = smoother)
    mean_vx : float
        Mean horizontal wind
    mean_vy : float
        Mean vertical wind (should be 0 for wall BCs)
    seed : int or None
        Random seed
    safe_max : float
        Maximum allowed speed (for rescaling)
    
    Returns
    -------
    tuple
        (v_func, coeffs) where v_func is dolfin Function and
        coeffs is dict with spectral coefficients
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
    v_func = dl.Function(Xh)
    
    # Split into sub-functions and assign each component separately
    Vh_scalar = dl.FunctionSpace(mesh, 'Lagrange', 1)
    
    vx_func = dl.Function(Vh_scalar)
    vy_func = dl.Function(Vh_scalar)
    
    # vertex_to_dof_map gives the correct mapping from vertex index to dof index
    v2d = dl.vertex_to_dof_map(Vh_scalar)
    
    vx_vals = vx_func.vector().get_local()
    vy_vals = vy_func.vector().get_local()
    for i in range(len(xy)):
        vx_vals[v2d[i]] = vx[i]
        vy_vals[v2d[i]] = vy[i]
    vx_func.vector().set_local(vx_vals)
    vy_func.vector().set_local(vy_vals)
    
    # Assign to vector function
    fa = dl.FunctionAssigner(Xh, [Vh_scalar, Vh_scalar])
    fa.assign(v_func, [vx_func, vy_func])
    
    # Safety: cap maximum velocity
    max_speed = np.sqrt(vx**2 + vy**2).max()
    if max_speed > safe_max:
        scale_factor = safe_max / max_speed
        print(f"  Rescaling wind: max speed {max_speed:.2f} → {safe_max:.2f}")
        v_func.vector()[:] *= scale_factor
        a_ij *= scale_factor
        b_ij *= scale_factor
        mean_vx *= scale_factor
        mean_vy *= scale_factor
    
    # Pack coefficients as compact NN input
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
    
    Size: 2 + 2*r_wind^2 + 2
    
    Parameters
    ----------
    coeffs : dict
        Spectral coefficients from sample_spectral_wind
    c_init : (2,) array
        Initial drone position
        
    Returns
    -------
    np.ndarray
        Flattened input vector
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
    """
    Dimension of the NN input vector.
    
    Parameters
    ----------
    r_wind : int
        Number of spectral modes per direction
        
    Returns
    -------
    int
        Input dimension
    """
    return 2 + 2 * r_wind**2 + 2