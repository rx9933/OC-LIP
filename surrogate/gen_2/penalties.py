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