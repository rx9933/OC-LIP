import os, sys, time, argparse, pickle
import numpy as np
import dolfin as dl

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

sys.path.append('../../')
sys.path.append('..')

from config import (K, NOISE_VARIANCE, SIMULATION_TIMES, OBSERVATION_TIMES,
                    TY, T_FINAL, T_1, LB, UB, GAMMA, DELTA)
from fourier_utils import fourier_frequencies
from training_data_generator_hippylib_example import (
    MESH_FILE,
    WIND_SPEED_LEFT_MEAN, WIND_SPEED_LEFT_STD,
    WIND_SPEED_RIGHT_MEAN, WIND_SPEED_RIGHT_STD,
    DRONE_POS_MEAN, DRONE_POS_STD, DRONE_POS_BOUNDS)

import training_data_generator_hippylib_example as gen
import oed_objective as oedobj
from training_data_generator_hippylib_example import (
    setup_buildings_mesh, setup_prior_buildings, BUILDINGS,
    sample_drone_position, sample_wind_speeds, fix_c0_in_m,
    run_multi_start_multi_refinement, run_single_stage, CachedEigensolver)

# ===== eigenvalue truncation rank =====
R_MODES_V2 = 10
gen.R_MODES = R_MODES_V2
oedobj.R_MODES = R_MODES_V2

# ===== soft path-length window (both phases) =====
L_MIN, L_MAX, ZETA_LEN = 1.945, 1.95, 15000.0
N_LEN_GRID = 200

# ===== consensus config =====
N_CONSENSUS = 10
CONSENSUS_SIGMA = 0.01

# ===== widened initial-guess radii =====
RADIUS_RANGE = (0.05, 0.25)

_omegas_K = fourier_frequencies(TY, K)


def create_initial_guess_K1_wide(c0):
    K_stage = 1
    omegas = fourier_frequencies(TY, K_stage)
    m0 = np.zeros(4 * K_stage + 2)
    radius = np.random.uniform(*RADIUS_RANGE)
    eccentricity = np.random.uniform(0, 0.5)
    a = radius * (1 + eccentricity)
    b = radius * (1 - eccentricity)
    theta = np.random.uniform(0, np.pi)
    sign = np.random.choice([-1, 1])
    m0[2] = a * np.cos(theta)
    m0[3] = -sign * b * np.sin(theta)
    m0[4] = a * np.sin(theta)
    m0[5] = sign * b * np.cos(theta)
    for j in range(2, 6):
        m0[j] = np.clip(m0[j], -0.3, 0.3)
    return fix_c0_in_m(m0, c0, K_stage, omegas)

gen.create_initial_guess_K1 = create_initial_guess_K1_wide


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


def compute_path_length(m, n_grid=500):
    vx, vy, _, _ = _path_velocity(m, np.linspace(0, T_FINAL, n_grid), K, _omegas_K)
    dt = T_FINAL / (n_grid - 1)
    return float(np.sum(np.sqrt(vx**2 + vy**2 + 1e-16)) * dt)


_orig_objective = gen.objective_with_obstacles

def objective_with_length(m, c0, Vh, mesh, prior, wind_velocity,
                          stage_K, omegas, eigsolver):
    J, grad, EIG_val, pen_val = _orig_objective(
        m, c0, Vh, mesh, prior, wind_velocity, stage_K, omegas, eigsolver)
    L_pen, L_grad = length_penalty(m, stage_K, omegas)
    return J + L_pen, grad + L_grad, EIG_val, pen_val + L_pen

gen.objective_with_obstacles = objective_with_length


def run_consensus(m_star, c0, mesh, Vh, prior, wind_velocity):
    shared_eig = CachedEigensolver()
    out = []
    for j in range(N_CONSENSUS):
        m0 = m_star.copy() if j == 0 else m_star + CONSENSUS_SIGMA*np.random.randn(len(m_star))
        m_opt, eig_opt, pen_opt, nfev = run_single_stage(
            3, m0, c0, mesh, Vh, prior, wind_velocity, shared_eig, 0, start_idx=j)
        out.append({'m': m_opt.copy(), 'eig': eig_opt, 'pen': pen_opt, 'nfev': nfev})
    return out


def main():
    parser = argparse.ArgumentParser(description='OED training data v2 (consensus)')
    parser.add_argument('--job_id', type=int, default=0)
    parser.add_argument('--n_samples', type=int, default=2)
    parser.add_argument('--n_starts', type=int, default=10)
    parser.add_argument('--output_prefix', type=str, default='hippylib_training_data_v2')
    args = parser.parse_args()

    job_id = args.job_id
    n_samples = args.n_samples
    n_starts = args.n_starts
    output_file = f"{args.output_prefix}_job{job_id}.pkl"

    print("=" * 70)
    print("  TRAINING DATA v2: exploration + consensus, R_MODES=10")
    print(f"  Job {job_id}, {n_samples} samples, {n_starts} explore starts, "
          f"{N_CONSENSUS} consensus starts (sigma={CONSENSUS_SIGMA})")
    print(f"  Window [{L_MIN},{L_MAX}] zeta={ZETA_LEN}, radii {RADIUS_RANGE}")
    print(f"  Output: {output_file}")
    print("=" * 70)
    sys.stdout.flush()

    base_seed = (int(time.time()) + job_id * 1000) % (2**32 - 1)
    print(f"  Base seed: {base_seed}")

    mesh_template = dl.refine(dl.Mesh(MESH_FILE))
    n_dofs = dl.FunctionSpace(mesh_template, "Lagrange", 1).dim()
    print(f"  Mesh DOFs: {n_dofs}")
    sys.stdout.flush()

    training_data = []
    successful = 0; failed = 0
    t_start = time.time()

    for i in range(n_samples):
        seed = base_seed + i
        np.random.seed(seed)
        speed_left, speed_right = sample_wind_speeds()
        c0 = sample_drone_position()
        print(f"\n  [{i+1:3d}/{n_samples}] seed={seed}  sl={speed_left:.3f}  "
              f"sr={speed_right:.3f}  c0=({c0[0]:.3f},{c0[1]:.3f})")
        sys.stdout.flush()
        t0 = time.time()
        try:
            mesh, Vh, wind_velocity = setup_buildings_mesh(speed_left, speed_right)
            prior = setup_prior_buildings(Vh)
            wind_dof_vector = wind_velocity.vector().get_local().copy()

            # ---- Phase 1: exploration ----
            result = run_multi_start_multi_refinement(
                c0, mesh, Vh, prior, wind_velocity, i, n_starts=n_starts)
            if result is None:
                raise RuntimeError("All multi-start attempts failed")
            m_explore = result['m_opt'].copy()
            t_explore = time.time() - t0

            # ---- Phase 2: consensus ----
            t1 = time.time()
            cons = run_consensus(m_explore, c0, mesh, Vh, prior, wind_velocity)
            t_consensus = time.time() - t1
            cons_eigs = np.array([c['eig'] for c in cons])
            best_c = int(np.argmax(cons_eigs))
            m_label = cons[best_c]['m'].copy()
            eig_label = float(cons_eigs[best_c])
            cons_spread = float(cons_eigs.max() - cons_eigs.min())
            L_label = compute_path_length(m_label)

            elapsed = time.time() - t0
            wind_params = np.array([speed_left, speed_right])

            sample = {
                'seed': seed,
                'c_init': c0.copy(),
                'speed_left': speed_left,
                'speed_right': speed_right,
                'wind_params': wind_params.copy(),
                'nn_input': np.concatenate([wind_params, c0]),
                'wind_dof_vector': wind_dof_vector,
                # LABEL = consensus best
                'm_opt': m_label,
                'eig_K3': eig_label,
                'length': L_label,
                # exploration record
                'm_init': result['m_init'].copy(),
                'm_opt_explore': m_explore,
                'eig_K0': result['eig_K0'],
                'eig_K1': result['eig_K1'],
                'eig_K2': result['eig_K2'],
                'eig_K3_explore': result['eig_K3'],
                'pen_K3': result['pen_K3'],
                'all_eigs': result.get('all_eigs', np.array([])),
                'best_start_idx': result.get('best_start_idx', 0),
                'eig_spread_explore': result.get('eig_spread', 0.0),
                # consensus diagnostics
                'consensus_eigs': cons_eigs,
                'consensus_m_opts': [c['m'].copy() for c in cons],
                'consensus_spread': cons_spread,
                'consensus_pens': np.array([c['pen'] for c in cons]),
                # cost
                'nfev_total': result['nfev_total'] + int(sum(c['nfev'] for c in cons)),
                'time': elapsed, 'time_explore': t_explore, 'time_consensus': t_consensus,
                'n_starts': n_starts,
                'R_MODES': R_MODES_V2,
            }
            training_data.append(sample)
            successful += 1
            print(f"    -> LABEL EIG: {eig_label:.2f}  L={L_label:.4f}  "
                  f"(explore spread={sample['eig_spread_explore']:.2f} -> "
                  f"consensus spread={cons_spread:.3f})  "
                  f"[{t_explore:.0f}s + {t_consensus:.0f}s]")
            sys.stdout.flush()
        except Exception as e:
            failed += 1
            print(f"    FAILED: {e}  [{time.time()-t0:.0f}s]")
            sys.stdout.flush()

        # checkpoint after every sample
        with open(output_file, 'wb') as f:
            pickle.dump({
                'samples': training_data,
                'job_id': job_id, 'n_samples': len(training_data),
                'n_starts': n_starts, 'buildings': BUILDINGS,
                'wind_prior': {'speed_left_mean': WIND_SPEED_LEFT_MEAN,
                               'speed_left_std': WIND_SPEED_LEFT_STD,
                               'speed_right_mean': WIND_SPEED_RIGHT_MEAN,
                               'speed_right_std': WIND_SPEED_RIGHT_STD},
                'drone_prior': {'mean': DRONE_POS_MEAN, 'std': DRONE_POS_STD,
                                'bounds': DRONE_POS_BOUNDS},
                'n_dofs': n_dofs, 'nn_input_dim': 4, 'nn_output_dim': 4*K + 2,
                'pipeline': 'v2_explore_plus_consensus',
                'R_MODES': R_MODES_V2,
                'L_window': (L_MIN, L_MAX), 'zeta': ZETA_LEN,
                'radius_range': RADIUS_RANGE,
                'n_consensus': N_CONSENSUS, 'consensus_sigma': CONSENSUS_SIGMA,
            }, f)

    print(f"\n  JOB {job_id} COMPLETE: {successful} ok, {failed} failed, "
          f"{(time.time()-t_start)/3600:.2f} h")


if __name__ == '__main__':
    main()
