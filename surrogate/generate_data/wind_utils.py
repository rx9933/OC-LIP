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
def spectral_wind_to_field(mesh, wind_coeffs):
    """
    Reconstruct wind field from spectral coefficients.
    
    Parameters
    ----------
    mesh : dolfin Mesh
        Computational mesh
    wind_coeffs : dict
        Spectral coefficients dictionary
        
    Returns
    -------
    dolfin Function
        Reconstructed velocity field (vector function)
    """
    # Extract parameters
    r_wind = wind_coeffs['r_wind']
    a_ij = wind_coeffs['a_ij']
    b_ij = wind_coeffs['b_ij']
    mean_vx = wind_coeffs['mean_vx']
    mean_vy = wind_coeffs['mean_vy']
    
    Lx, Ly = 1.0, 1.0  # unit square
    
    # Create a UserExpression that evaluates the wind field at any point
    class WindExpression(dl.UserExpression):
        def __init__(self, a_ij, b_ij, mean_vx, mean_vy, r_wind, **kwargs):
            super().__init__(**kwargs)
            self.a_ij = a_ij
            self.b_ij = b_ij
            self.mean_vx = mean_vx
            self.mean_vy = mean_vy
            self.r_wind = r_wind
            
        def eval(self, values, x):
            # Evaluate at point x
            vx_val = self.mean_vx
            vy_val = self.mean_vy
            
            for i in range(self.r_wind):
                for j in range(self.r_wind):
                    mode_i = i + 1
                    mode_j = j + 1
                    vx_val += self.a_ij[i, j] * np.cos(mode_i * np.pi * x[1] / Ly) * np.cos(mode_j * np.pi * x[0] / Lx)
                    vy_val += self.b_ij[i, j] * np.sin(mode_i * np.pi * x[1] / Ly) * np.cos(mode_j * np.pi * x[0] / Lx)
            
            values[0] = vx_val
            values[1] = vy_val
            
        def value_shape(self):
            return (2,)
    
    # Create the expression
    wind_expr = WindExpression(a_ij, b_ij, mean_vx, mean_vy, r_wind, degree=2)
    
    # Create vector function space for velocity
    V_vec = dl.VectorFunctionSpace(mesh, 'Lagrange', 1)
    
    # Interpolate to vector function space
    v_func = dl.interpolate(wind_expr, V_vec)
    
    return v_func, V_vec  # Return both the function and the space
def spectral_wind_to_coeffs(v_func, mesh, r_wind=3):
    """
    Extract spectral coefficients from a wind field (inverse of spectral_wind_to_field).
    
    This computes the projection of the field onto the spectral basis.
    
    Parameters
    ----------
    v_func : dolfin Function
        Velocity field
    mesh : dolfin Mesh
        Computational mesh
    r_wind : int
        Number of spectral modes per direction
        
    Returns
    -------
    dict
        Spectral coefficients dictionary
    """
    Lx, Ly = 1.0, 1.0
    
    # Get mesh coordinates
    xy = mesh.coordinates()
    x = xy[:, 0]
    y = xy[:, 1]
    
    # Evaluate velocity at mesh vertices
    vx_vals = np.zeros(len(xy))
    vy_vals = np.zeros(len(xy))
    
    # Get vertex to DOF mapping for scalar function space
    Vh_scalar = dl.FunctionSpace(mesh, 'Lagrange', 1)
    v2d = dl.vertex_to_dof_map(Vh_scalar)
    
    # Get the vector function's components
    vx_func, vy_func = v_func.split(deepcopy=True)
    
    vx_dofs = vx_func.vector().get_local()
    vy_dofs = vy_func.vector().get_local()
    
    # Map DOFs to vertices
    for i, dof_idx in enumerate(v2d):
        vx_vals[i] = vx_dofs[dof_idx]
        vy_vals[i] = vy_dofs[dof_idx]
    
    # Compute mean velocities
    mean_vx = np.mean(vx_vals)
    mean_vy = np.mean(vy_vals)
    
    # Subtract mean
    vx_centered = vx_vals - mean_vx
    vy_centered = vy_vals - mean_vy
    
    # Compute coefficients via least squares (or simple projection)
    a_ij = np.zeros((r_wind, r_wind))
    b_ij = np.zeros((r_wind, r_wind))
    
    for i in range(r_wind):
        for j in range(r_wind):
            mode_i = i + 1
            mode_j = j + 1
            
            # Basis functions
            phi_x = np.cos(mode_i * np.pi * y / Ly) * np.cos(mode_j * np.pi * x / Lx)
            phi_y = np.sin(mode_i * np.pi * y / Ly) * np.cos(mode_j * np.pi * x / Lx)
            
            # Project (least squares)
            # For unit square with uniform mesh, simple dot product works
            # For better accuracy, we should integrate, but this approximation is fine
            a_ij[i, j] = np.dot(vx_centered, phi_x) / np.dot(phi_x, phi_x)
            b_ij[i, j] = np.dot(vy_centered, phi_y) / np.dot(phi_y, phi_y)
    
    coeffs = {
        'a_ij': a_ij,
        'b_ij': b_ij,
        'mean_vx': mean_vx,
        'mean_vy': mean_vy,
        'r_wind': r_wind,
        'sigma': 1.0,  # Placeholder, not used
        'alpha': 2.0,  # Placeholder, not used
    }
    
    return coeffs

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