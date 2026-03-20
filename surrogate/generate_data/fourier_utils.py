"""Fourier path utilities for sensor trajectory representation."""

import numpy as np


def fourier_frequencies(Ty, Nf):
    """
    Fourier angular frequencies for time window Ty = (t_start, t_end).
    
    Parameters
    ----------
    Ty : tuple
        (t_start, t_end) time window
    Nf : int
        Number of frequencies
        
    Returns
    -------
    np.ndarray
        Angular frequencies
    """
    T = Ty[1] - Ty[0]
    return np.array([2.0 * np.pi * (k + 1) / T for k in range(Nf)])


def fourier_path(t, xbar, coeffs, omegas):
    """
    Evaluate the Fourier curve at parameter t.

    Parameters
    ----------
    t : float
        Time parameter
    xbar : (2,) array
        Mean position [x̄, ȳ]
    coeffs : (Nf, 4) array
        [θ_k, φ_k, ψ_k, η_k] per mode
    omegas : (Nf,) array
        Angular frequencies

    Returns
    -------
    (2,) array
        [x(t), y(t)]
    """
    x = xbar[0]
    y = xbar[1]
    for k, w in enumerate(omegas):
        x += coeffs[k, 0] * np.cos(w * t) + coeffs[k, 1] * np.sin(w * t)
        y += coeffs[k, 2] * np.cos(w * t) + coeffs[k, 3] * np.sin(w * t)
    return np.array([x, y])


def m_to_xbar_coeffs(m_fourier, K):
    """
    Unpack flat optimisation vector → (xbar, coeffs).
    
    Parameters
    ----------
    m_fourier : np.ndarray
        Flat vector of length 4*K + 2
    K : int
        Number of Fourier modes
        
    Returns
    -------
    tuple
        (xbar, coeffs) where xbar is (2,) and coeffs is (K, 4)
    """
    xbar = m_fourier[:2].copy()
    coeffs = np.zeros((K, 4))
    for k in range(K):
        coeffs[k, 0] = m_fourier[2 + 4*k]      # θ_k
        coeffs[k, 1] = m_fourier[3 + 4*k]      # φ_k
        coeffs[k, 2] = m_fourier[4 + 4*k]      # ψ_k
        coeffs[k, 3] = m_fourier[5 + 4*k]      # η_k
    return xbar, coeffs


def xbar_coeffs_to_m(xbar, coeffs, K):
    """
    Pack (xbar, coeffs) → flat optimisation vector.
    
    Parameters
    ----------
    xbar : (2,) array
        Mean position
    coeffs : (K, 4) array
        Fourier coefficients
    K : int
        Number of Fourier modes
        
    Returns
    -------
    np.ndarray
        Flat vector of length 4*K + 2
    """
    m = np.zeros(4*K + 2)
    m[:2] = xbar
    for k in range(K):
        m[2 + 4*k] = coeffs[k, 0]
        m[3 + 4*k] = coeffs[k, 1]
        m[4 + 4*k] = coeffs[k, 2]
        m[5 + 4*k] = coeffs[k, 3]
    return m

def get_position_at_time(m, t, K, omegas):
    """
    Get position [x, y] at a specific time from Fourier parameters.
    
    Parameters
    ----------
    m : np.ndarray
        Fourier parameter vector (4*K + 2)
    t : float
        Time
    K : int
        Number of Fourier modes
    omegas : np.ndarray
        Angular frequencies
        
    Returns
    -------
    np.ndarray
        [x, y] position
    """
    x = m[0]  # x̄
    y = m[1]  # ȳ
    
    for k in range(K):
        cos_kt = np.cos(omegas[k] * t)
        sin_kt = np.sin(omegas[k] * t)
        
        x += m[2 + 4*k] * cos_kt + m[3 + 4*k] * sin_kt      # θ_k cos + φ_k sin
        y += m[4 + 4*k] * cos_kt + m[5 + 4*k] * sin_kt      # ψ_k cos + η_k sin
    
    return np.array([x, y])

def generate_targets(m_fourier, t_param, K, omegas, eps=1e-6):
    """
    Generate sensor positions at specified times.
    
    Parameters
    ----------
    m_fourier : np.ndarray
        Flat Fourier parameter vector
    t_param : np.ndarray
        Times at which to evaluate
    K : int
        Number of Fourier modes
    omegas : np.ndarray
        Angular frequencies
    eps : float
        Clipping epsilon for boundary
        
    Returns
    -------
    np.ndarray
        (n_times, 2) array of positions
    """
    xbar, coeffs = m_to_xbar_coeffs(m_fourier, K)
    targets = np.array([fourier_path(t, xbar, coeffs, omegas)
                        for t in t_param])
    
    n_clipped = np.sum(targets < eps) + np.sum(targets > 1.0 - eps)
    if n_clipped > 0:
        # print(f"  WARNING: path clipped at {n_clipped} coordinates")
        pass
    
    targets[:, 0] = np.clip(targets[:, 0], eps, 1.0 - eps)
    targets[:, 1] = np.clip(targets[:, 1], eps, 1.0 - eps)
    return targets


def fourier_velocity(m_fourier, t_arr, K, omegas):
    """
    Compute dx/dt, dy/dt at each time in t_arr.
    
    Parameters
    ----------
    m_fourier : np.ndarray
        Flat Fourier parameter vector
    t_arr : np.ndarray
        Times at which to evaluate
    K : int
        Number of Fourier modes
    omegas : np.ndarray
        Angular frequencies
        
    Returns
    -------
    tuple
        (vx, vy) arrays of velocities
    """
    xbar, coeffs = m_to_xbar_coeffs(m_fourier, K)
    vx = np.zeros(len(t_arr))
    vy = np.zeros(len(t_arr))
    for k, w in enumerate(omegas):
        vx += w * (-coeffs[k, 0] * np.sin(w * t_arr) + coeffs[k, 1] * np.cos(w * t_arr))
        vy += w * (-coeffs[k, 2] * np.sin(w * t_arr) + coeffs[k, 3] * np.cos(w * t_arr))
    return vx, vy