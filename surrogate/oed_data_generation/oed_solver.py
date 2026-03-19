"""EIG computation and optimization for OED."""

import numpy as np
import dolfin as dl
import time
import sys

# Add paths
sys.path.append("../../../")
sys.path.append('../../')

# Import from hippylib
from hippylib import *
from hippylib.hippylib.modeling.prior import BiLaplacianPrior
# from hippylib.hippylib.modeling.misfit import Misfit

# Local imports
from oed_core import (generate_targets, boundary_penalty_dense,
                      speed_penalty_dense, obstacle_penalty_dense,
                      get_snapshot, eval_fn_and_grad_P1,
                      reset_bbt_cache, MovingSensorMisfit)



class CachedEigensolver:
    """Cache random vectors for consistent EIG computation."""
    
    def __init__(self):
        self.Omega = None
    
    def solve(self, prob, prior, r):
        """Compute eigenvalues and EIG value."""
        u_lin = prob.generate_vector(STATE)
        m_lin = prob.generate_vector(PARAMETER)
        p_lin = prob.generate_vector(ADJOINT)

        prob.solveFwd(u_lin, [u_lin, m_lin, p_lin])
        prob.solveAdj(p_lin, [u_lin, m_lin, p_lin])

        H = ReducedHessian(prob, misfit_only=True)
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
        lmbda = np.maximum(lmbda, 0.0)
        EIG = float(np.sum(np.log(1.0 + lmbda)))
        return lmbda, V, EIG
        
    def reset(self):
        self.Omega = None


def build_problem(m_fourier, Vh, prior, wind_velocity, 
                  observation_times, simulation_times,
                  K, omegas, noise_variance):
    """Build the OED problem for a given sensor path."""
    # Import here to avoid circular imports
    from model_ad_diff_bwd import TimeDependentAD
    
    targets = generate_targets(m_fourier, observation_times, K, omegas)
    misfit = MovingSensorMisfit(Vh, observation_times, targets, noise_variance)

    problem = TimeDependentAD(
        Vh.mesh(), [Vh, Vh, Vh], prior, misfit,
        simulation_times, wind_velocity, True
    )
    return problem, misfit, targets


def compute_eigendecomposition(prob, prior, r, eigsolver):
    """Wrapper for eigensolver."""
    return eigsolver.solve(prob, prior, r)


def oed_objective_and_grad(m_fourier, Vh, prior, wind_velocity,
                           observation_times, simulation_times,
                           K, omegas, noise_variance, r_modes,
                           obstacles, eigsolver,
                           zeta_bdy=1000.0, zeta_speed=500.0,
                           v_max=0.5, zeta_obs=2000.0):
    """
    Compute OED objective (negative EIG + penalties) and its gradient.
    """
    _t0 = time.time()

    # Build problem and get eigenpairs
    prob, msft, tgts = build_problem(
        m_fourier, Vh, prior, wind_velocity,
        observation_times, simulation_times,
        K, omegas, noise_variance
    )
    lmbda, V, EIG_val = compute_eigendecomposition(prob, prior, r_modes, eigsolver)

    # Forward trajectories for each eigenmode
    lmbda_thresh = 1e-2
    n_active = max(1, int(np.sum(lmbda > lmbda_thresh)))
    
    u_tilde_trajs = []
    for i in range(n_active):
        v_i = dl.Function(Vh).vector()
        v_i.axpy(1.0, V[i])
        u_t = prob.generate_vector(STATE)
        prob.solveFwd(u_t, [u_t, v_i, None])
        u_tilde_trajs.append(u_t)

    # Sensitivity S(t_j)
    n_obs = len(observation_times)
    mesh = Vh.mesh()
    S = np.zeros((n_obs, 2))
    
    for j in range(n_obs):
        cj = tgts[j]
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

    # Project to Fourier basis
    def project_to_fourier(S_mat):
        g = np.zeros(4*K + 2)
        g[0] = np.sum(S_mat[:, 0])
        g[1] = np.sum(S_mat[:, 1])
        for kk in range(K):
            cos_v = np.cos(omegas[kk] * observation_times)
            sin_v = np.sin(omegas[kk] * observation_times)
            g[2 + 4*kk] = np.dot(S_mat[:, 0], cos_v)
            g[3 + 4*kk] = np.dot(S_mat[:, 0], sin_v)
            g[4 + 4*kk] = np.dot(S_mat[:, 1], cos_v)
            g[5 + 4*kk] = np.dot(S_mat[:, 1], sin_v)
        return g
    
    grad_eig = project_to_fourier(S)

    # Penalties
    pen_val, grad_pen = boundary_penalty_dense(
        m_fourier, observation_times, K, omegas, zeta=zeta_bdy)
    spd_val, grad_spd = speed_penalty_dense(
        m_fourier, observation_times, K, omegas,
        v_max_arg=v_max, zeta_speed_arg=zeta_speed)
    obs_val, grad_obs = obstacle_penalty_dense(
        m_fourier, observation_times, K, omegas,
        obstacles, zeta_obs=zeta_obs)

    _elapsed = time.time() - _t0
    J = -EIG_val + pen_val + spd_val + obs_val
    
    return (J, -grad_eig + grad_pen + grad_spd + grad_obs), {
        'eig': EIG_val,
        'bdy': pen_val,
        'spd': spd_val,
        'obs': obs_val,
        'time': _elapsed
    }


def create_initial_path(c_init, K):
    """Create initial Fourier path centered at drone position."""
    m0 = np.zeros(4*K + 2)
    m0[0] = c_init[0]
    m0[1] = c_init[1]
    m0[2] = 0.05   # small initial amplitude
    m0[5] = 0.05
    return m0