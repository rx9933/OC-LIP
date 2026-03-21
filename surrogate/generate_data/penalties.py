"""Penalty functions for sensor path constraints."""

import numpy as np

from fourier_utils import generate_targets, m_to_xbar_coeffs, fourier_velocity


def boundary_penalty_dense(m_fourier, t_param, K, omegas, 
                           zeta=None, eps_bdy=0.02, n_dense=200):
    """
    Evaluate boundary penalty on a dense grid.
    
    Parameters
    ----------
    m_fourier : np.ndarray
        Flat Fourier parameter vector
    t_param : np.ndarray
        Observation times
    K : int
        Number of Fourier modes
    omegas : np.ndarray
        Angular frequencies
    zeta : float
        Penalty weight
    eps_bdy : float
        Boundary margin
    n_dense : int
        Number of dense grid points
        
    Returns
    -------
    tuple
        (val, grad) penalty value and gradient
    """
    if zeta is None:
        from config import ZETA_BDY
        zeta = ZETA_BDY
        
    t_dense = np.linspace(t_param[0], t_param[-1], n_dense)
    dt_dense = t_dense[1] - t_dense[0]
    targets_dense = generate_targets(m_fourier, t_dense, K, omegas)
    
    val = 0.0
    S_B_dense = np.zeros((n_dense, 2))
    margin = 0.05
    for d in range(2):
        for j in range(n_dense):
            lo = targets_dense[j, d] - eps_bdy
            hi = (1.0 - eps_bdy) - targets_dense[j, d]
            if lo < margin:
                val += dt_dense * zeta * (margin - lo)**2
                S_B_dense[j, d] -= zeta * 2.0 * (margin - lo)
            if hi < margin:
                val += dt_dense * zeta * (margin - hi)**2
                S_B_dense[j, d] += zeta * 2.0 * (margin - hi)
    
    # Project onto Fourier basis
    g = np.zeros(4*K + 2)
    g[0] = dt_dense * np.sum(S_B_dense[:, 0])
    g[1] = dt_dense * np.sum(S_B_dense[:, 1])
    for kk in range(K):
        cos_v = np.cos(omegas[kk] * t_dense)
        sin_v = np.sin(omegas[kk] * t_dense)
        g[2 + 4*kk] = dt_dense * np.dot(S_B_dense[:, 0], cos_v)
        g[3 + 4*kk] = dt_dense * np.dot(S_B_dense[:, 0], sin_v)
        g[4 + 4*kk] = dt_dense * np.dot(S_B_dense[:, 1], cos_v)
        g[5 + 4*kk] = dt_dense * np.dot(S_B_dense[:, 1], sin_v)
    
    return val, g


def speed_penalty_dense(m_fourier, t_param, K, omegas,
                        v_max_arg=None, zeta_speed_arg=None, n_dense=200):
    """
    Penalty for exceeding maximum speed.
    
    Parameters
    ----------
    m_fourier : np.ndarray
        Flat Fourier parameter vector
    t_param : np.ndarray
        Observation times
    K : int
        Number of Fourier modes
    omegas : np.ndarray
        Angular frequencies
    v_max_arg : float
        Maximum speed
    zeta_speed_arg : float
        Penalty weight
    n_dense : int
        Number of dense grid points
        
    Returns
    -------
    tuple
        (val, grad) penalty value and gradient
    """
    if v_max_arg is None:
        from config import V_MAX
        v_max_arg = V_MAX
    if zeta_speed_arg is None:
        from config import ZETA_SPEED
        zeta_speed_arg = ZETA_SPEED
        
    t_dense = np.linspace(t_param[0], t_param[-1], n_dense)
    dt_dense = t_dense[1] - t_dense[0]
    
    vx, vy = fourier_velocity(m_fourier, t_dense, K, omegas)
    speed2 = vx**2 + vy**2
    v_max2 = v_max_arg**2
    
    # Penalty: sum over times where speed exceeds v_max
    excess = np.maximum(0.0, speed2 - v_max2)
    val = dt_dense * zeta_speed_arg * np.sum(excess**2)
    
    # Weight at each time: 4 ζ (|v|² − v_max²) where excess > 0
    weight = 4.0 * zeta_speed_arg * excess
    
    g = np.zeros(4*K + 2)
    
    for kk in range(K):
        w = omegas[kk]
        sin_v = np.sin(w * t_dense)
        cos_v = np.cos(w * t_dense)
        
        dvx_dtheta = -w * sin_v
        dvx_dphi = w * cos_v
        dvy_dpsi = -w * sin_v
        dvy_deta = w * cos_v
        
        g[2 + 4*kk] = dt_dense * np.dot(weight * vx, dvx_dtheta)
        g[3 + 4*kk] = dt_dense * np.dot(weight * vx, dvx_dphi)
        g[4 + 4*kk] = dt_dense * np.dot(weight * vy, dvy_dpsi)
        g[5 + 4*kk] = dt_dense * np.dot(weight * vy, dvy_deta)
    
    return val, g


def acceleration_penalty_dense(
    m_fourier,
    t_param,
    K,
    omegas,
    a_max=None,
    zeta_accel=None,
    n_dense=200,
):
    """
    Penalty for exceeding maximum acceleration.
    """
    if a_max is None:
        from config import A_MAX
        a_max = A_MAX

    if zeta_accel is None:
        from config import ZETA_ACCEL
        zeta_accel = ZETA_ACCEL

    t_dense = np.linspace(t_param[0], t_param[-1], n_dense)
    dt_dense = t_dense[1] - t_dense[0]

    _, coeffs = m_to_xbar_coeffs(m_fourier, K)

    # Compute acceleration
    ax_arr = np.zeros(n_dense)
    ay_arr = np.zeros(n_dense)

    for k in range(K):
        w = omegas[k]

        ax_arr += -w**2 * (
            coeffs[k, 0] * np.cos(w * t_dense)
            + coeffs[k, 1] * np.sin(w * t_dense)
        )

        ay_arr += -w**2 * (
            coeffs[k, 2] * np.cos(w * t_dense)
            + coeffs[k, 3] * np.sin(w * t_dense)
        )

    accel2 = ax_arr**2 + ay_arr**2
    a_max2 = a_max**2

    excess = np.maximum(0.0, accel2 - a_max2)

    val = dt_dense * zeta_accel * np.sum(excess**2)

    weight = 4.0 * zeta_accel * excess

    g = np.zeros(4 * K + 2)

    for kk in range(K):
        w = omegas[kk]
        cos_v = np.cos(w * t_dense)
        sin_v = np.sin(w * t_dense)

        g[2 + 4 * kk] = dt_dense * np.dot(weight * ax_arr, -w**2 * cos_v)
        g[3 + 4 * kk] = dt_dense * np.dot(weight * ax_arr, -w**2 * sin_v)
        g[4 + 4 * kk] = dt_dense * np.dot(weight * ay_arr, -w**2 * cos_v)
        g[5 + 4 * kk] = dt_dense * np.dot(weight * ay_arr, -w**2 * sin_v)

    return val, g

def initial_position_penalty_dense(m, t_param, K, omegas, c0, weight=200.0):
    """
    Penalty to enforce that the path starts at c0.
    """
    from fourier_utils import get_position_at_time  # Import the new helper
    
    # Get position at first time - use the correct function
    pos0 = get_position_at_time(m, t_param[0], K, omegas)
    
    # Position error
    dx = pos0[0] - c0[0]
    dy = pos0[1] - c0[1]
    
    # Penalty (squared error)
    penalty = 0.5 * weight * (dx**2 + dy**2)
    
    # Gradient computation
    grad = np.zeros_like(m)
    
    t0 = t_param[0]
    
    # Gradient w.r.t x̄
    grad[0] = weight * dx
    
    # Gradient w.r.t ȳ
    grad[1] = weight * dy
    
    # Gradients for Fourier coefficients
    for k in range(K):
        cos_kt = np.cos(omegas[k] * t0)
        sin_kt = np.sin(omegas[k] * t0)
        
        # θ_k (x cosine coefficient)
        grad[2 + 4*k] = weight * dx * cos_kt
        
        # φ_k (x sine coefficient)
        grad[3 + 4*k] = weight * dx * sin_kt
        
        # ψ_k (y cosine coefficient)
        grad[4 + 4*k] = weight * dy * cos_kt
        
        # η_k (y sine coefficient)
        grad[5 + 4*k] = weight * dy * sin_kt
    
    return penalty, grad
