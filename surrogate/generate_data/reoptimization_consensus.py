import os, sys, time, argparse, pickle
import numpy as np
import dolfin as dl

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

sys.path.append('../../')
sys.path.append('..')
from hippylib import BiLaplacianPrior

from config import (
    K, NOISE_VARIANCE, SIMULATION_TIMES, OBSERVATION_TIMES,
    TY, T_FINAL, T_1, LB, UB, GAMMA, DELTA,
)
from fourier_utils import fourier_frequencies

import training_data_generator_hippylib_example as gen
import oed_objective as oedobj
from training_data_generator_hippylib_example import (
    setup_buildings_mesh, CachedEigensolver, run_single_stage,
)

# ===== R_MODES = 10, same as the run being consensused =====
gen.R_MODES = 10
oedobj.R_MODES = 10

# ===== Length window: identical to the 1945 run =====
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


def compute_path_length(m, K_local, omegas_arr, n_grid=N_LEN_GRID):
    t_arr = np.linspace(0.0, T_FINAL, n_grid); dt = t_arr[1]-t_arr[0]
    vx, vy, _, _ = _path_velocity(m, t_arr, K_local, omegas_arr)
    return float(np.sum(np.sqrt(vx**2+vy**2+1e-16))*dt)


# ===== Point 2: orientation canonicalization (reusable utility) =====
def signed_area(m, K_local):
    """A = pi * sum_k k*(theta_k*eta_k - phi_k*psi_k). CCW > 0."""
    A = 0.0
    for k in range(K_local):
        th = m[2+4*k]; ph = m[3+4*k]; ps = m[4+4*k]; et = m[5+4*k]
        A += (k+1) * (th*et - ph*ps)
    return np.pi * A


def canonicalize_orientation(m, K_local):
    """If clockwise (A<0), flip t -> -t (negate sine coeffs). Same geometry."""
    if signed_area(m, K_local) < 0:
        m = m.copy()
        for k in range(K_local):
            m[3+4*k] = -m[3+4*k]
            m[5+4*k] = -m[5+4*k]
        return m, True
    return m, False


_orig_objective = gen.objective_with_obstacles

def objective_with_length(m, c0, Vh, mesh, prior, wind_velocity,
                          stage_K, omegas, eigsolver):
    J, grad, EIG_val, pen_val = _orig_objective(
        m, c0, Vh, mesh, prior, wind_velocity, stage_K, omegas, eigsolver)
    L_pen, L_grad = length_penalty(m, stage_K, omegas)
    return J + L_pen, grad + L_grad, EIG_val, pen_val + L_pen

gen.objective_with_obstacles = objective_with_length


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_pkl', type=str,
                        default='multi_refinement_length_1945_195_r10_results.pkl')
    parser.add_argument('--n_starts', type=int, default=10)
    parser.add_argument('--sigma', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', type=str, default='data/consensus_1945_195_r10')
    args = parser.parse_args()

    d = pickle.load(open(args.source_pkl, 'rb'))
    results = d['results']
    c0 = np.asarray(d['c0'])
    sl = float(d['speed_left']); sr = float(d['speed_right'])
    eigs0 = np.array([r['eig_K3'] for r in results])
    best = int(np.argmax(eigs0))
    m_star = np.asarray(results[best]['m_K3']).copy()
    spread0 = eigs0.max() - eigs0.min()

    print(f"Source: {args.source_pkl}")
    print(f"  incumbent best: {results[best]['label']}  EIG={eigs0.max():.3f}  "
          f"L={results[best]['L_K3']:.4f}  (source spread {spread0:.2f})")
    print(f"Consensus: {args.n_starts} starts from m* (start 0 exact, "
          f"others +N(0,{args.sigma}^2)), window [{L_MIN},{L_MAX}] z={ZETA_LEN}, "
          f"R_MODES=10\n")

    np.random.seed(args.seed)
    omegas_final = fourier_frequencies(TY, K)
    mesh, Vh, wind = setup_buildings_mesh(speed_left=sl, speed_right=sr)
    prior = BiLaplacianPrior(Vh, GAMMA, DELTA, robin_bc=True)
    prior.mean = dl.interpolate(dl.Constant(0.25), Vh).vector()

    shared_eig = CachedEigensolver()
    out_results = []
    t0 = time.time()
    for j in range(args.n_starts):
        m0 = m_star.copy() if j == 0 else m_star + args.sigma*np.random.randn(len(m_star))
        m0, flipped = canonicalize_orientation(m0, K)
        if flipped:
            print(f"  [start {j}] orientation flip applied (unexpected)")
        m_opt, eig_opt, pen_opt, nfev = run_single_stage(
            3, m0, c0, mesh, Vh, prior, wind, shared_eig, 0, start_idx=j)
        L = compute_path_length(m_opt, K, omegas_final)
        out_results.append({'idx': j, 'm_K3': m_opt.copy(), 'eig_K3': eig_opt,
                            'L_K3': L, 'pen_K3': pen_opt, 'nfev': nfev,
                            'perturbed': j != 0})
        print(f"    consensus start {j}: EIG={eig_opt:.3f}  L={L:.4f}  nfev={nfev}")

    eigs = np.array([r['eig_K3'] for r in out_results])
    Ls   = np.array([r['L_K3']  for r in out_results])
    print(f"\n=== CONSENSUS RESULT ===")
    print(f"  EIG range: {eigs.min():.3f} to {eigs.max():.3f}   "
          f"spread={eigs.max()-eigs.min():.4f}   (was {spread0:.2f})")
    print(f"  L range:   [{Ls.min():.4f}, {Ls.max():.4f}]")
    print(f"  wall: {time.time()-t0:.0f}s")

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out + '.pkl', 'wb') as f:
        pickle.dump({'results': out_results, 'm_star': m_star,
                     'source_pkl': args.source_pkl, 'source_best': results[best]['label'],
                     'source_spread': spread0, 'c0': c0, 'speed_left': sl,
                     'speed_right': sr, 'sigma': args.sigma,
                     'L_MIN': L_MIN, 'L_MAX': L_MAX, 'ZETA_LEN': ZETA_LEN,
                     'R_MODES': 10}, f)
    print(f"Saved: {args.out}.pkl")


if __name__ == '__main__':
    main()
