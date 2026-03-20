import numpy as np
import dolfin as dl
import time

import logging
logging.getLogger('FFC').setLevel(logging.WARNING)
logging.getLogger('UFL').setLevel(logging.WARNING)
dl.set_log_active(False)


import sys, os
sys.path.append(os.environ.get('HIPPYLIB_BASE_DIR', "../../../", "../../"))
from hippylib import *
from model_ad_diff_bwd import TimeDependentAD
from scipy.optimize import minimize



from path import *



def generate_targets(m_fourier, t_param, K, omegas):
    xbar, coeffs = m_to_xbar_coeffs(m_fourier, K)
    targets = np.array([fourier_path(t, xbar, coeffs, omegas)
                        for t in t_param])
    eps = 1e-6
    n_clipped = np.sum(targets < eps) + np.sum(targets > 1.0 - eps)
    if n_clipped > 0:
        print(f"  WARNING: path clipped at {n_clipped} coordinates")
    targets[:, 0] = np.clip(targets[:, 0], eps, 1.0 - eps)
    targets[:, 1] = np.clip(targets[:, 1], eps, 1.0 - eps)
    return targets

# ═══════════════════════════════════════════════════════════════
# 9.  EIGENDECOMPOSITION
# ═══════════════════════════════════════════════════════════════
class CachedEigensolver:
    def __init__(self):
        self.Omega = None
    
    def solve(self, prob, prior, r):
        u_lin = prob.generate_vector(STATE)
        m_lin = prob.generate_vector(PARAMETER)
        p_lin = prob.generate_vector(ADJOINT)

        prob.solveFwd(u_lin, [u_lin, m_lin, p_lin])
        prob.solveAdj(p_lin, [u_lin, m_lin, p_lin])

        H     = ReducedHessian(prob, misfit_only=True)
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
        self.Omega = None

def compute_eigendecomposition(prob, prior, r):
    return eigsolver.solve(prob, prior, r)

eigsolver = CachedEigensolver() # where should this go?

# ═══════════════════════════════════════════════════════════════
# 8.  BUILD PROBLEM  
# ═══════════════════════════════════════════════════════════════
class MovingSensorMisfit:
    """
    Moving sensor: at observation time t_k, only target k is active.
    Matches SpaceTimePointwiseStateObservation interface exactly.
    """
    
    def __init__(self, Vh, observation_times, targets):
        self.Vh = Vh
        self.observation_times = observation_times
        self.n_obs = len(observation_times)
        self.targets = targets
        self.noise_variance = 1.0
        
        assert targets.shape[0] == self.n_obs
        
        # One B_k (1 × N_dof) per observation time
        self.B_list = []
        for k in range(self.n_obs):
            B_k = assemblePointwiseObservation(Vh, targets[k:k+1, :])
            self.B_list.append(B_k)
        
        # Also store B_list[0] as self.B so hIPPYlib internals don't break
        self.B = self.B_list[0]
        
        # Data: obs-space vectors (dim 1) at each time
        self.d = TimeDependentVector(observation_times)
        self.d.data = []
        for k in range(self.n_obs):
            v = dl.Vector()
            self.B.init_vector(v, 0)
            self.d.data.append(v)
        
        # Pre-allocate work vectors
        self.u_snapshot  = dl.Vector()
        self.Bu_snapshot = dl.Vector()
        self.d_snapshot  = dl.Vector()
        self.B.init_vector(self.u_snapshot, 1)   # N_dof
        self.B.init_vector(self.Bu_snapshot, 0)  # dim 1
        self.B.init_vector(self.d_snapshot, 0)   # dim 1
    
    def observe(self, x, d):
        for k in range(self.n_obs):
            tk = self.observation_times[k]
            x[STATE].retrieve(self.u_snapshot, tk)
            self.B_list[k].mult(self.u_snapshot, self.Bu_snapshot)
            d.store(self.Bu_snapshot, tk)
    
    def cost(self, x):
        c = 0.0
        for k in range(self.n_obs):
            tk = self.observation_times[k]
            x[STATE].retrieve(self.u_snapshot, tk)
            self.B_list[k].mult(self.u_snapshot, self.Bu_snapshot)
            self.d.retrieve(self.d_snapshot, tk)
            self.Bu_snapshot.axpy(-1.0, self.d_snapshot)
            c += self.Bu_snapshot.inner(self.Bu_snapshot)
        return 0.5 * c / self.noise_variance
    
    def grad(self, i, x, out):
        out.zero()
        if i == STATE:
            for k in range(self.n_obs):
                tk = self.observation_times[k]
                x[STATE].retrieve(self.u_snapshot, tk)
                self.B_list[k].mult(self.u_snapshot, self.Bu_snapshot)
                self.d.retrieve(self.d_snapshot, tk)
                self.Bu_snapshot.axpy(-1.0, self.d_snapshot)
                self.Bu_snapshot *= 1.0 / self.noise_variance
                self.B_list[k].transpmult(self.Bu_snapshot, self.u_snapshot)
                out.store(self.u_snapshot, tk)
        else:
            pass
    
    def setLinearizationPoint(self, x, gauss_newton_approx=False):
        pass
    
    def apply_ij(self, i, j, direction, out):
        out.zero()
        if i == STATE and j == STATE:
            for k in range(self.n_obs):
                tk = self.observation_times[k]
                direction.retrieve(self.u_snapshot, tk)
                self.B_list[k].mult(self.u_snapshot, self.Bu_snapshot)
                self.Bu_snapshot *= 1.0 / self.noise_variance
                self.B_list[k].transpmult(self.Bu_snapshot, self.u_snapshot)
                out.store(self.u_snapshot, tk)
        else:
            pass


def build_problem(m_fourier):
    targets = generate_targets(m_fourier, t_param, K, omegas)
    misfit = MovingSensorMisfit(Vh, observation_times, targets)
    misfit.noise_variance = noise_variance

    problem = TimeDependentAD(
        mesh, [Vh, Vh, Vh], prior, misfit,
        simulation_times, wind_velocity, True
    )
    return problem, misfit, targets

def run_single_oed_sample(wind_coeffs, wind_velocity, c_init,
                         bounds, K, prior, r_modes):
    """
    Runs one OED optimization and returns structured result.
    """
    global _cached_bbt, opt_history

    # Initial guess centered at drone
    m0 = np.zeros(4*K + 2)
    m0[0], m0[1] = c_init
    m0[2] = 0.05
    m0[5] = 0.05

    _cached_bbt = None
    eigsolver.reset()
    opt_history = {'eig': [], 'bdy': [], 'spd': [], 'J': [], 'time': []}

    t0 = time.time()

    result = minimize(EIG_objective_and_grad, m0,
                      jac=True, method='L-BFGS-B', bounds=bounds,
                      options={'maxiter': 80, 'disp': False,
                               'ftol': 1e-10, 'maxls': 40})

    m_opt = result.x

    # Evaluate final EIG
    prob_opt, _, _ = build_problem(m_opt)
    _, _, eig_opt = compute_eigendecomposition(prob_opt, prior, r_modes)

    elapsed = time.time() - t0

    return {
        'm_opt': m_opt,
        'eig_opt': eig_opt,
        'converged': result.success,
        'time': elapsed
    }