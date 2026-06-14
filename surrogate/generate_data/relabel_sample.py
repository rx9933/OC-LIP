"""
relabel_sample.py
Per-sample relabeling worker. Loads neighbors' labels, runs L-BFGS from each
as initial guess, keeps best-EIG result.

UPDATED: Uses objective_with_obstacles (same as data generator) so paths
respect building obstacles.
"""

import os, sys, pickle, argparse, time, warnings
warnings.filterwarnings('ignore')

import numpy as np
import dolfin as dl
from scipy.optimize import minimize

dl.set_log_level(40)

sys.path.append('generate_data/')
sys.path.append(os.path.join(os.path.dirname(__file__), 'generate_data'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from generate_data.config import *
from generate_data.fourier_utils import fourier_frequencies
from generate_data.oed_objective import (CachedEigensolver, build_problem,
                                          compute_eig_gradient)
from generate_data.fe_setup import setup_prior
from generate_data.penalties import (boundary_penalty_dense, speed_penalty_dense,
                                      acceleration_penalty_dense,
                                      initial_position_penalty_dense,
                                      obstacle_penalty_dense)


# Buildings layout (must match training_data_generator_hippylib_example.py)
BUILDINGS = [
    {'type': 'rectangle', 'lower': (0.26, 0.16), 'upper': (0.49, 0.39), 'margin': 0.03},
    {'type': 'rectangle', 'lower': (0.61, 0.61), 'upper': (0.74, 0.84), 'margin': 0.03},
]


# ===== Path-length window: IDENTICAL to production_worker_v2.py =====
L_MIN, L_MAX, ZETA_LEN = 1.945, 1.95, 15000.0
N_LEN_GRID = 200


def _path_velocity(m, t_arr, K_local, omegas_arr):
    cos_wt = np.cos(np.outer(t_arr, omegas_arr))
    sin_wt = np.sin(np.outer(t_arr, omegas_arr))
    vx = np.zeros(len(t_arr)); vy = np.zeros(len(t_arr))
    for k in range(K_local):
        w = omegas_arr[k]
        vx += w*(-m[2+4*k]*sin_wt[:,k] + m[3+4*k]*cos_wt[:,k])
        vy += w*(-m[4+4*k]*sin_wt[:,k] + m[5+4*k]*cos_wt[:,k])
    return vx, vy, cos_wt, sin_wt


def length_penalty(m, K_local, omegas_arr, n_grid=N_LEN_GRID):
    t_arr = np.linspace(0.0, T_FINAL, n_grid); dt = t_arr[1]-t_arr[0]
    vx, vy, cos_wt, sin_wt = _path_velocity(m, t_arr, K_local, omegas_arr)
    speed = np.sqrt(vx**2 + vy**2 + 1e-16)
    L = float(np.sum(speed)*dt)
    lo = max(0.0, L_MIN - L); hi = max(0.0, L - L_MAX)
    pen_val = ZETA_LEN*(lo**2 + hi**2)
    dPdL = 2.0*ZETA_LEN*(-lo + hi)
    grad = np.zeros(4*K_local + 2)
    if dPdL == 0.0:
        return pen_val, grad
    inv = 1.0/speed
    for k in range(K_local):
        w = omegas_arr[k]
        grad[2+4*k] = dPdL*dt*np.sum(vx*inv*(-w*sin_wt[:,k]))
        grad[3+4*k] = dPdL*dt*np.sum(vx*inv*( w*cos_wt[:,k]))
        grad[4+4*k] = dPdL*dt*np.sum(vy*inv*(-w*sin_wt[:,k]))
        grad[5+4*k] = dPdL*dt*np.sum(vy*inv*( w*cos_wt[:,k]))
    return pen_val, grad


def reconstruct_wind(wind_dofs, V_vec):
    w = dl.Function(V_vec)
    w.vector().set_local(wind_dofs)
    w.vector().apply('insert')
    return w


def objective_with_obstacles(m, c0, Vh, mesh, prior, wind_velocity,
                              stage_K, omegas, eigsolver):
    """OED objective with ALL penalties including obstacle avoidance.
    Mirrors training_data_generator_hippylib_example.py:228-260."""
    prob, msft, tgts = build_problem(
        m, Vh, prior, SIMULATION_TIMES, OBSERVATION_TIMES,
        wind_velocity, stage_K, omegas, NOISE_VARIANCE, mesh
    )

    EIG_val, grad_eig, lmbda, V = compute_eig_gradient(
        m, prob, prior, R_MODES, eigsolver, Vh, mesh,
        SIMULATION_TIMES, OBSERVATION_TIMES, stage_K, omegas,
        NOISE_VARIANCE, OBSERVATION_TIMES
    )

    grad = -grad_eig.copy()
    pen_val = 0.0

    bdy_val, grad_bdy = boundary_penalty_dense(m, OBSERVATION_TIMES, stage_K, omegas)
    pen_val += bdy_val; grad += grad_bdy

    spd_val, grad_spd = speed_penalty_dense(m, OBSERVATION_TIMES, stage_K, omegas)
    pen_val += spd_val; grad += grad_spd

    acc_val, grad_acc = acceleration_penalty_dense(m, OBSERVATION_TIMES, stage_K, omegas)
    pen_val += acc_val; grad += grad_acc

    ic_val, grad_ic = initial_position_penalty_dense(m, OBSERVATION_TIMES, stage_K, omegas, c0)
    pen_val += ic_val; grad += grad_ic

    obs_val, grad_obs = obstacle_penalty_dense(m, OBSERVATION_TIMES, stage_K, omegas, BUILDINGS)
    pen_val += obs_val; grad += grad_obs

    len_val, grad_len = length_penalty(m, stage_K, omegas)
    pen_val += len_val; grad += grad_len

    J = -EIG_val + pen_val
    return J, grad, EIG_val, pen_val


def compute_eig_clean(m, c_init, wind, mesh, Vh, prior):
    """Compute just EIG (no penalties), for final reported value."""
    omegas = fourier_frequencies(TY, K)
    prob, _, _ = build_problem(m, Vh, prior, SIMULATION_TIMES, OBSERVATION_TIMES,
                                wind, K, omegas, NOISE_VARIANCE, mesh)
    es = CachedEigensolver()
    _, _, eig_val = es.solve(prob, prior, R_MODES)
    return eig_val


def path_intersects_buildings(m_vec, buildings=BUILDINGS, n_check=400):
    """Check if the Fourier path ever enters a building interior.
    Uses a finer time grid than the penalty (400 pts) for safety.
    Returns True if path violates any building."""
    omegas = fourier_frequencies(TY, K)
    t_grid = np.linspace(T_1, T_FINAL, n_check)
    x_bar, y_bar = m_vec[0], m_vec[1]
    x = np.full_like(t_grid, x_bar)
    y = np.full_like(t_grid, y_bar)
    for k in range(K):
        theta, phi, psi, eta = m_vec[2+4*k : 2+4*(k+1)]
        x += theta * np.cos(omegas[k] * t_grid) + phi * np.sin(omegas[k] * t_grid)
        y += psi * np.cos(omegas[k] * t_grid) + eta * np.sin(omegas[k] * t_grid)
    for b in buildings:
        lx, ly = b['lower']
        ux, uy = b['upper']
        mask = (x >= lx) & (x <= ux) & (y >= ly) & (y <= uy)
        if mask.any():
            return True
    return False


def run_lbfgs_from_start(m_start, c_init, wind, mesh, Vh, prior,
                         maxiter=30, ftol=1e-6):
    omegas = fourier_frequencies(TY, K)
    eigsolver = CachedEigensolver()

    def objective(m):
        J, grad, _, _ = objective_with_obstacles(
            m, c_init, Vh, mesh, prior, wind, K, omegas, eigsolver
        )
        return J, grad

    try:
        result = minimize(
            objective, m_start,
            jac=True, method='L-BFGS-B', bounds=BOUNDS,
            options={'maxiter': maxiter, 'disp': False,
                     'ftol': ftol, 'maxls': OPT_MAXLS, 'maxfun': OPT_MAXFUN}
        )
        eig_final = compute_eig_clean(result.x, c_init, wind, mesh, Vh, prior)
        return result.x, eig_final, result.success
    except Exception as e:
        return m_start, -1e10, False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_idx', type=int, required=True)
    parser.add_argument('--pass_num', type=int, required=True)
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--neighbor_file', type=str, required=True)
    parser.add_argument('--current_labels_file', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--n_neighbors', type=int, default=10)
    parser.add_argument('--maxiter', type=int, default=30)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f'sample_{args.sample_idx:06d}.pkl')
    if os.path.exists(out_path):
        print(f"Already done: {out_path}")
        return

    t_start = time.time()

    data = np.load(args.input_file, allow_pickle=True, mmap_mode='r')
    x_data = data['x']
    wind_dofs_all = data['wind_dofs']

    cur = np.load(args.current_labels_file, allow_pickle=True)
    m_current = cur['m']
    eig_current = cur['eig']

    nbrs = np.load(args.neighbor_file, allow_pickle=True)
    nn_idxs = nbrs['nn_idxs']

    i = args.sample_idx
    c0 = x_data[i]

    mesh = dl.refine(dl.Mesh('generate_data/ad_20.xml'))
    Vh = dl.FunctionSpace(mesh, 'Lagrange', 1)
    V_vec = dl.VectorFunctionSpace(mesh, 'Lagrange', 2)
    prior = setup_prior(Vh)
    wind = reconstruct_wind(wind_dofs_all[i], V_vec)

    candidate_starts = [m_current[i]]
    for k in range(args.n_neighbors):
        nb_idx = nn_idxs[i, k]
        candidate_starts.append(m_current[nb_idx])

    best_m = m_current[i].copy()
    best_eig = eig_current[i]
    best_source = 'original'
    all_eigs = []
    rejected_candidates = []  # track which were rejected for path violations

    for k, m_start in enumerate(candidate_starts):
        m_opt, eig_opt, converged = run_lbfgs_from_start(
            m_start, c0, wind, mesh, Vh, prior, maxiter=args.maxiter
        )
        all_eigs.append(eig_opt)

        # Safety check: reject if path goes through buildings
        # (obstacle penalty should prevent this, but verify)
        if path_intersects_buildings(m_opt):
            rejected_candidates.append(k)
            continue

        if eig_opt > best_eig:
            best_m = m_opt
            best_eig = eig_opt
            best_source = f'neighbor_{k-1}' if k > 0 else 'self'

    elapsed = time.time() - t_start

    result = {
        'sample_idx': i,
        'pass_num': args.pass_num,
        'm_new': best_m,
        'eig_new': best_eig,
        'eig_old': eig_current[i],
        'improved': best_eig > eig_current[i],
        'improvement': best_eig - eig_current[i],
        'best_source': best_source,
        'all_eigs': all_eigs,
        'rejected_candidates': rejected_candidates,
        'elapsed': elapsed,
    }
    with open(out_path, 'wb') as f:
        pickle.dump(result, f)

    print(f"Sample {i:5d}  pass {args.pass_num}: "
          f"EIG {eig_current[i]:.3f} -> {best_eig:.3f} "
          f"({'IMPROVED' if best_eig > eig_current[i] else 'same'} "
          f"from {best_source}) [{elapsed:.1f}s] "
          f"rejected={len(rejected_candidates)}/11")


if __name__ == '__main__':
    main()
