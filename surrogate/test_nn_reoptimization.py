"""
Test 2: NN-Initialized Re-Optimization
=======================================
Use the NN-predicted path as initial guess for L-BFGS-B and compare:
  - How many iterations to converge (vs cold start from circular guess)
  - Final EIG compared to PDE-optimal
  - Whether NN warm-start finds the same local optimum
"""

import sys, os
import numpy as np
import torch
import dolfin as dl
import time
from scipy.optimize import minimize

sys.path.append('../')
sys.path.append('generate_data/')

from generate_data.config import *
from generate_data.fourier_utils import fourier_frequencies, xbar_coeffs_to_m
from generate_data.wind_utils import spectral_wind_to_field
from generate_data.oed_objective import (CachedEigensolver, build_problem,
                                          oed_objective_and_grad)
from generate_data.fe_setup import setup_prior
from generate_data.training_data_generator import create_initial_guess

# Import NN architecture
sys.path.append('dinotorch_lite/src/')
from dinotorch_lite import GenericDense

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--n_samples', type=int, default=5, help='Number of test samples')
parser.add_argument('--data_dir', type=str, default='data/', help='Data directory')
parser.add_argument('--n_train', type=int, default=800, help='Training set size (for model name)')
parser.add_argument('--data_type', type=str, default='xvspectral', help='Input type')
parser.add_argument('--dQ', type=int, default=22, help='Input dimension')
parser.add_argument('--dM', type=int, default=14, help='Output dimension')
parser.add_argument('--maxiter_nn', type=int, default=80, help='Max iterations from NN init')
parser.add_argument('--maxiter_cold', type=int, default=80, help='Max iterations from cold start')
args = parser.parse_args()


def reconstruct_wind_coeffs(v_mean, v_coeff, r_wind=3):
    """Reconstruct wind coefficients dictionary from stored data."""
    n_coeff = r_wind * r_wind
    a_ij = v_coeff[:n_coeff].reshape(r_wind, r_wind)
    b_ij = v_coeff[n_coeff:].reshape(r_wind, r_wind)
    return {
        'a_ij': a_ij, 'b_ij': b_ij,
        'mean_vx': v_mean[0], 'mean_vy': v_mean[1],
        'r_wind': r_wind, 'sigma': 1.0, 'alpha': 2.0
    }


def compute_eig(m_fourier, wind_velocity, mesh, Vh, prior):
    """Compute EIG for given path and pre-built wind field."""
    omegas = fourier_frequencies(TY, K)
    prob, _, _ = build_problem(
        m_fourier, Vh, prior, SIMULATION_TIMES, OBSERVATION_TIMES,
        wind_velocity, K, omegas, NOISE_VARIANCE, mesh
    )
    eigsolver = CachedEigensolver()
    _, _, EIG_val = eigsolver.solve(prob, prior, R_MODES)
    return EIG_val


def run_optimization(m_init, c_init, wind_velocity, mesh, Vh, prior, maxiter, label=""):
    """Run L-BFGS-B optimization from given initial guess."""
    omegas = fourier_frequencies(TY, K)
    eigsolver = CachedEigensolver()

    iteration_count = [0]
    eig_history = []

    def objective(m):
        J, grad, eig_val, pen_val, spd_val, elapsed = oed_objective_and_grad(
            c_init, m, Vh, mesh, prior, SIMULATION_TIMES, OBSERVATION_TIMES,
            wind_velocity, K, omegas, R_MODES, NOISE_VARIANCE,
            OBSERVATION_TIMES, eigsolver, obstacles=None, include_penalties=True
        )
        iteration_count[0] += 1
        eig_history.append(eig_val)
        return J, grad

    t0 = time.time()
    result = minimize(
        objective, m_init,
        jac=True, method='L-BFGS-B', bounds=BOUNDS,
        options={'maxiter': maxiter, 'disp': False,
                 'ftol': OPT_FTOL, 'maxls': OPT_MAXLS, 'maxfun': OPT_MAXFUN}
    )
    elapsed = time.time() - t0

    # Compute final EIG (clean, no penalties)
    eigsolver.reset()
    _, _, eig_final, _, _, _ = oed_objective_and_grad(
        c_init, result.x, Vh, mesh, prior, SIMULATION_TIMES, OBSERVATION_TIMES,
        wind_velocity, K, omegas, R_MODES, NOISE_VARIANCE,
        OBSERVATION_TIMES, eigsolver, obstacles=None, include_penalties=False
    )

    return {
        'label': label,
        'm_opt': result.x,
        'eig_final': eig_final,
        'nit': result.nit,
        'nfev': result.nfev,
        'converged': result.success,
        'time': elapsed,
        'eig_history': eig_history,
        'message': result.message
    }


def main():
    print("=" * 70)
    print("  NN-INITIALIZED RE-OPTIMIZATION TEST")
    print("=" * 70)

    # Setup
    mesh = dl.UnitSquareMesh(NX, NY)
    Vh = dl.FunctionSpace(mesh, 'Lagrange', 1)
    prior = setup_prior(Vh)

    device = torch.device('cpu')

    # Load NN model
    model = GenericDense(input_dim=args.dQ, hidden_layer_dim=2*args.dQ, output_dim=args.dM).to(device)
    model_path = os.path.join(args.data_dir,
        f"rbno_datatype_{args.data_type}_rQ{args.dQ}_rM{args.dM}_ntrain{args.n_train}.pth")

    if not os.path.exists(model_path):
        print(f"  ERROR: Model not found at {model_path}")
        print(f"  Run training first: CUDA_VISIBLE_DEVICES='' python train_rbno_oc.py")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"  Loaded NN model from {model_path}")

    # Load data
    mq_data = np.load(os.path.join(args.data_dir, 'mq_data_reduced.npz'))
    m_data = mq_data['m']
    v_mean = mq_data['v_mean']
    v_coeff = mq_data['v_coeff']
    x_data = mq_data['x']

    # Build input data (same as training script)
    if args.data_type == 'xvspectral':
        q_data = np.concatenate((mq_data['x'], mq_data['v_mean'], mq_data['v_coeff']), axis=1)
    else:
        q_data = np.concatenate((mq_data['x'], mq_data['v']), axis=1)

    n_test = min(args.n_samples, 100)
    test_indices = np.arange(len(m_data) - n_test, len(m_data))[:args.n_samples]

    print(f"  Test samples: {len(test_indices)}")
    print(f"  Max iterations (NN init): {args.maxiter_nn}")
    print(f"  Max iterations (cold start): {args.maxiter_cold}")
    print()

    results_nn = []
    results_cold = []

    for i, sample_idx in enumerate(test_indices):
        print(f"\n{'═' * 70}")
        print(f"  Sample {i+1}/{len(test_indices)} (index {sample_idx})")
        print(f"{'═' * 70}")

        m_true = m_data[sample_idx]
        wind_coeffs = reconstruct_wind_coeffs(v_mean[sample_idx], v_coeff[sample_idx])
        c_init = x_data[sample_idx]

        # Reconstruct wind field
        wind_velocity, _ = spectral_wind_to_field(mesh, wind_coeffs)

        # Get NN prediction
        q_input = q_data[sample_idx]
        with torch.no_grad():
            q_tensor = torch.FloatTensor(q_input).unsqueeze(0).to(device)
            m_nn = model(q_tensor).cpu().numpy().flatten()

        # Compute EIG at different starting points
        eig_true = compute_eig(m_true, wind_velocity, mesh, Vh, prior)
        eig_nn = compute_eig(m_nn, wind_velocity, mesh, Vh, prior)

        print(f"  PDE-optimal EIG:  {eig_true:.4f}")
        print(f"  NN prediction EIG: {eig_nn:.4f}")
        print(f"  Path distance:     {np.linalg.norm(m_nn - m_true):.4f}")

        # --- Optimization from NN warm start ---
        print(f"\n  Running L-BFGS-B from NN prediction...")
        res_nn = run_optimization(m_nn, c_init, wind_velocity, mesh, Vh, prior,
                                   args.maxiter_nn, label="NN warm-start")
        print(f"    Final EIG: {res_nn['eig_final']:.4f} | "
              f"Iterations: {res_nn['nit']} | "
              f"Time: {res_nn['time']:.1f}s | "
              f"Converged: {res_nn['converged']}")

        # --- Optimization from cold start (circular guess) ---
        print(f"\n  Running L-BFGS-B from circular cold start...")
        m_cold = create_initial_guess(c_init, K)
        eig_cold_init = compute_eig(m_cold, wind_velocity, mesh, Vh, prior)
        print(f"    Cold start EIG: {eig_cold_init:.4f}")

        res_cold = run_optimization(m_cold, c_init, wind_velocity, mesh, Vh, prior,
                                     args.maxiter_cold, label="Cold start")
        print(f"    Final EIG: {res_cold['eig_final']:.4f} | "
              f"Iterations: {res_cold['nit']} | "
              f"Time: {res_cold['time']:.1f}s | "
              f"Converged: {res_cold['converged']}")

        # Compare
        print(f"\n  Comparison:")
        print(f"    NN warm-start EIG:   {res_nn['eig_final']:.4f} ({res_nn['nit']} iters, {res_nn['time']:.1f}s)")
        print(f"    Cold start EIG:      {res_cold['eig_final']:.4f} ({res_cold['nit']} iters, {res_cold['time']:.1f}s)")
        print(f"    Original PDE-opt:    {eig_true:.4f}")
        if res_cold['time'] > 0:
            speedup = res_cold['time'] / max(res_nn['time'], 0.1)
            print(f"    Speedup:             {speedup:.1f}x")

        results_nn.append({
            'sample_idx': sample_idx,
            'eig_pde': eig_true,
            'eig_nn_init': eig_nn,
            'eig_nn_opt': res_nn['eig_final'],
            'nit_nn': res_nn['nit'],
            'time_nn': res_nn['time'],
            'conv_nn': res_nn['converged'],
        })
        results_cold.append({
            'sample_idx': sample_idx,
            'eig_cold_init': eig_cold_init,
            'eig_cold_opt': res_cold['eig_final'],
            'nit_cold': res_cold['nit'],
            'time_cold': res_cold['time'],
            'conv_cold': res_cold['converged'],
        })

    # ==================== SUMMARY ====================
    print("\n" + "=" * 70)
    print("  SUMMARY TABLE")
    print("=" * 70)

    print(f"\n{'Sample':>8s}  {'EIG_PDE':>8s}  {'EIG_NN':>8s}  "
          f"{'NN→Opt':>8s}  {'Cold→Opt':>8s}  "
          f"{'NN iters':>8s}  {'Cold iters':>10s}  "
          f"{'Speedup':>8s}")
    print("-" * 90)

    speedups = []
    for rnn, rcold in zip(results_nn, results_cold):
        spd = rcold['time_cold'] / max(rnn['time_nn'], 0.1)
        speedups.append(spd)
        print(f"  {rnn['sample_idx']:5d}  "
              f"{rnn['eig_pde']:8.2f}  "
              f"{rnn['eig_nn_init']:8.2f}  "
              f"{rnn['eig_nn_opt']:8.2f}  "
              f"{rcold['eig_cold_opt']:8.2f}  "
              f"{rnn['nit_nn']:8d}  "
              f"{rcold['nit_cold']:10d}  "
              f"{spd:7.1f}x")

    print("-" * 90)

    # Averages
    avg_nn_iters = np.mean([r['nit_nn'] for r in results_nn])
    avg_cold_iters = np.mean([r['nit_cold'] for r in results_cold])
    avg_speedup = np.mean(speedups)
    avg_eig_gap_nn = np.mean([r['eig_pde'] - r['eig_nn_opt'] for r in results_nn])
    avg_eig_gap_cold = np.mean([r['eig_pde'] - rcold['eig_cold_opt']
                                 for rcold in results_cold])

    print(f"\n  Average NN warm-start iterations:  {avg_nn_iters:.1f}")
    print(f"  Average cold start iterations:     {avg_cold_iters:.1f}")
    print(f"  Average speedup:                   {avg_speedup:.1f}x")
    print(f"  Average EIG gap (NN→Opt vs PDE):   {avg_eig_gap_nn:.4f}")
    print(f"  Average EIG gap (Cold→Opt vs PDE): {avg_eig_gap_cold:.4f}")

    # Save results
    save_path = os.path.join(args.data_dir, 'reoptimization_results.npz')
    np.savez(save_path,
             results_nn=results_nn,
             results_cold=results_cold,
             speedups=speedups)
    print(f"\n  Results saved to {save_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
