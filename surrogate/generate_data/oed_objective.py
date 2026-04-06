"""OED objective function and gradient computation."""

import numpy as np
import dolfin as dl
import time

from hippylib import (STATE, PARAMETER, ADJOINT, MultiVector, 
                      ReducedHessian, doublePassG, parRandom)

from fourier_utils import generate_targets
from penalties import boundary_penalty_dense, speed_penalty_dense, acceleration_penalty_dense, initial_position_penalty_dense
import sys
sys.path.append('../../')
from model_ad_diff_bwd import TimeDependentAD
from moving_sensor import MovingSensorMisfit
from fe_utils import get_snapshot, eval_fn_and_grad_P1


class CachedEigensolver:
    """Cache random vectors for eigensolver to ensure consistency."""
    
    def __init__(self):
        self.Omega = None
    
    def solve(self, prob, prior, r, misfit_only=True):
        """Compute eigenvalues and eigenvectors."""
        u_lin = prob.generate_vector(STATE)
        m_lin = prob.generate_vector(PARAMETER)
        p_lin = prob.generate_vector(ADJOINT)

        prob.solveFwd(u_lin, [u_lin, m_lin, p_lin])
        prob.solveAdj(p_lin, [u_lin, m_lin, p_lin])

        H = ReducedHessian(prob, misfit_only=misfit_only)
        Omega = MultiVector(m_lin, r + 10)

        if self.Omega is None:
            parRandom.normal(1.0, Omega)
            self.Omega = MultiVector(m_lin, r + 10)
            for i in range(r + 10):
                self.Omega[i].zero()
                self.Omega[i].axpy(1.0, Omega[i])
        else:
            for i in range(r + 10):
                Omega[i].zero()
                Omega[i].axpy(1.0, self.Omega[i])

        lmbda, V = doublePassG(H, prior.R, prior.Rsolver, Omega, r)
        # Clip tiny/negative eigenvalues from numerical noise
        lmbda = np.maximum(lmbda, 0.0)
        EIG = float(np.sum(np.log(1.0 + lmbda)))
        return lmbda, V, EIG
        
    def reset(self):
        """Reset cached random vectors."""
        self.Omega = None


def build_problem(m_fourier, Vh, prior, simulation_times, observation_times,
                  wind_velocity, K, omegas, noise_variance, mesh):
    """
    Build the time-dependent advection-diffusion problem.
    
    Parameters
    ----------
    m_fourier : np.ndarray
        Flat Fourier parameter vector
    Vh : dolfin FunctionSpace
        Finite element space
    prior : BiLaplacianPrior
        Prior distribution
    simulation_times : np.ndarray
        Times for simulation
    observation_times : np.ndarray
        Times for observations
    wind_velocity : dolfin Function
        Velocity field
    K : int
        Number of Fourier modes
    omegas : np.ndarray
        Angular frequencies
    noise_variance : float
        Noise variance for observations
    mesh : dolfin Mesh
        Computational mesh (needed for TimeDependentAD)
        
    Returns
    -------
    tuple
        (problem, misfit, targets)
    """
    targets = generate_targets(m_fourier, observation_times, K, omegas)
    misfit = MovingSensorMisfit(Vh, observation_times, targets)
    misfit.noise_variance = noise_variance

    problem = TimeDependentAD(
        mesh, [Vh, Vh, Vh], prior, misfit,
        simulation_times, wind_velocity, True
    )
    return problem, misfit, targets


def compute_eig_decomposition(prob, prior, r_modes, eigsolver):
    """
    Compute eigendecomposition using cached eigensolver.
    
    Parameters
    ----------
    prob : TimeDependentAD
        Problem instance
    prior : BiLaplacianPrior
        Prior distribution
    r_modes : int
        Number of eigenvalues to keep
    eigsolver : CachedEigensolver
        Eigensolver with caching
        
    Returns
    -------
    tuple
        (lmbda, V, EIG_val)
    """
    return eigsolver.solve(prob, prior, r_modes)


def project_to_fourier(S_mat, t_param, K, omegas):
    """
    Project sensitivity matrix onto Fourier basis.
    
    Parameters
    ----------
    S_mat : np.ndarray
        (n_obs, 2) sensitivity matrix
    t_param : np.ndarray
        Observation times
    K : int
        Number of Fourier modes
    omegas : np.ndarray
        Angular frequencies
        
    Returns
    -------
    np.ndarray
        Gradient in Fourier space
    """
    g = np.zeros(4*K + 2)
    g[0] = np.sum(S_mat[:, 0])
    g[1] = np.sum(S_mat[:, 1])
    for kk in range(K):
        cos_v = np.cos(omegas[kk] * t_param)
        sin_v = np.sin(omegas[kk] * t_param)
        g[2 + 4*kk] = np.dot(S_mat[:, 0], cos_v)
        g[3 + 4*kk] = np.dot(S_mat[:, 0], sin_v)
        g[4 + 4*kk] = np.dot(S_mat[:, 1], cos_v)
        g[5 + 4*kk] = np.dot(S_mat[:, 1], sin_v)
    return g


def compute_eig_gradient(m_fourier, prob, prior, r_modes, eigsolver,
                         Vh, mesh, simulation_times, observation_times,
                         K, omegas, noise_variance, t_param, lmbda_thresh=1e-2):
    """
    Compute gradient of EIG with respect to Fourier parameters.
    
    Parameters
    ----------
    m_fourier : np.ndarray
        Flat Fourier parameter vector
    prob : TimeDependentAD
        Problem instance
    prior : BiLaplacianPrior
        Prior distribution
    r_modes : int
        Number of eigenvalues to keep
    eigsolver : CachedEigensolver
        Eigensolver with caching
    Vh : dolfin FunctionSpace
        Finite element space
    mesh : dolfin Mesh
        Computational mesh
    simulation_times : np.ndarray
        Times for simulation
    observation_times : np.ndarray
        Times for observations
    K : int
        Number of Fourier modes
    omegas : np.ndarray
        Angular frequencies
    noise_variance : float
        Noise variance for observations
    t_param : np.ndarray
        Observation times
    lmbda_thresh : float
        Threshold for active eigenvalues
        
    Returns
    -------
    tuple
        (EIG_val, grad_eig, lmbda, V)
    """
    # Get eigenpairs
    lmbda, V, EIG_val = compute_eig_decomposition(prob, prior, r_modes, eigsolver)
    
    # Get targets
    targets = generate_targets(m_fourier, observation_times, K, omegas)
    
    # Forward trajectories ũ_i (one per eigenmode)
    n_active = max(1, int(np.sum(lmbda > lmbda_thresh)))
    
    u_tilde_trajs = []
    for i in range(n_active):
        v_i = dl.Function(Vh).vector()
        v_i.axpy(1.0, V[i])
        u_t = prob.generate_vector(STATE)
        prob.solveFwd(u_t, [u_t, v_i, None])
        u_tilde_trajs.append(u_t)
    
    # Sensitivity S(t_j) ∈ R^{M×2}
    n_obs = len(observation_times)
    S = np.zeros((n_obs, 2))
    for j in range(n_obs):
        cj = targets[j]
        tj = observation_times[j]
        for i in range(n_active):
            scale_i = 2.0 / ((1.0 + max(lmbda[i], 0.0)) * noise_variance)
            snap = get_snapshot(u_tilde_trajs[i], tj, simulation_times, Vh)
            f_t = dl.Function(Vh)
            f_t.vector().zero()
            f_t.vector().axpy(1.0, snap)
            ut_val, gut = eval_fn_and_grad_P1(f_t, mesh, cj)
            S[j, 0] += scale_i * ut_val * gut[0]
            S[j, 1] += scale_i * ut_val * gut[1]
    
    # Project onto Fourier basis
    grad_eig = project_to_fourier(S, t_param, K, omegas)
    
    return EIG_val, grad_eig, lmbda, V


def oed_objective_and_grad(c0, m_fourier, Vh, mesh, prior, simulation_times,
                           observation_times, wind_velocity, K, omegas,
                           r_modes, noise_variance, t_param, eigsolver,
                           obstacles=None, include_penalties=True):
    """
    Compute OED objective and gradient.
    
    Parameters
    ----------
    m_fourier : np.ndarray
        Flat Fourier parameter vector
    Vh : dolfin FunctionSpace
        Finite element space
    mesh : dolfin Mesh
        Computational mesh
    prior : BiLaplacianPrior
        Prior distribution
    simulation_times : np.ndarray
        Times for simulation
    observation_times : np.ndarray
        Times for observations
    wind_velocity : dolfin Function
        Velocity field
    K : int
        Number of Fourier modes
    omegas : np.ndarray
        Angular frequencies
    r_modes : int
        Number of eigenvalues to keep
    noise_variance : float
        Noise variance for observations
    t_param : np.ndarray
        Observation times
    eigsolver : CachedEigensolver
        Eigensolver with caching
    obstacles : list or None
        List of obstacles (not used in training)
    include_penalties : bool
        Whether to include penalty terms
        
    Returns
    -------
    tuple
        (J, grad, EIG_val, pen_val, spd_val, elapsed)
    """
    _t0 = time.time()
    
    # Build problem
    prob, msft, tgts = build_problem(
        m_fourier, Vh, prior, simulation_times, observation_times,
        wind_velocity, K, omegas, noise_variance, mesh
    )
    
    # Compute EIG and its gradient
    EIG_val, grad_eig, lmbda, V = compute_eig_gradient(
        m_fourier, prob, prior, r_modes, eigsolver, Vh, mesh,
        simulation_times, observation_times, K, omegas,
        noise_variance, t_param
    )
    
    # Initialize gradient
    grad = -grad_eig
    
    # Add penalties if requested
    pen_val = 0.0
    spd_val = 0.0
    
    if include_penalties:
        # Boundary penalty
        bdy_val, grad_bdy = boundary_penalty_dense(
            m_fourier, t_param, K, omegas
        )
        pen_val += bdy_val
        grad += grad_bdy
        
        # Speed penalty
        spd_val, grad_spd = speed_penalty_dense(
            m_fourier, t_param, K, omegas
        )
        pen_val += spd_val
        grad += grad_spd

        # acceleration / curvature
        acc_val, grad_acc = acceleration_penalty_dense(
        m_fourier, t_param, K, omegas
        )
        pen_val += acc_val
        grad += grad_acc
                # Initial position penalty (NEW)
        if c0 is not None:
            init_pen_val, grad_init = initial_position_penalty_dense(
                m_fourier, t_param, K, omegas, c0
            )
            pen_val += init_pen_val
            grad += grad_init
    

    J = -EIG_val + pen_val
    
    _elapsed = time.time() - _t0
    
    return J, grad, EIG_val, pen_val, spd_val, _elapsed
def compute_eig_for_path(m_fourier, wind_coeffs, mesh, Vh):
    """Compute EIG for a given set of Fourier path coefficients."""
    from config import (SIMULATION_TIMES, OBSERVATION_TIMES, K, TY,
                        NOISE_VARIANCE, R_MODES, GAMMA, DELTA)
    from fourier_utils import fourier_frequencies
    from wind_utils import spectral_wind_to_field
    from hippylib import BiLaplacianPrior

    omegas = fourier_frequencies(TY, K)
    from fe_setup import setup_prior
    prior = setup_prior(Vh)

    if wind_coeffs is not None:
        wind_velocity, _ = spectral_wind_to_field(mesh, wind_coeffs)
    else:
        V_vec = dl.VectorFunctionSpace(mesh, 'Lagrange', 1)
        wind_velocity = dl.Function(V_vec)

    prob, _, _ = build_problem(
        m_fourier, Vh, prior, SIMULATION_TIMES, OBSERVATION_TIMES,
        wind_velocity, K, omegas, NOISE_VARIANCE, mesh
    )

    eigsolver = CachedEigensolver()
    _, _, EIG_val = eigsolver.solve(prob, prior, R_MODES)
    return EIG_val
