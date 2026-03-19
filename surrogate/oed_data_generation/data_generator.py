#!/usr/bin/env python3
"""
Main data generation script for OED training data.
Samples wind fields and drone positions, then optimizes sensor paths.
"""

import numpy as np
import dolfin as dl
import pickle
import time
import argparse
from scipy.optimize import minimize
import sys
import os

# Add HIPPYLIB path
sys.path.append("../../../")

# Local imports
from config import *
from wind_sampler import sample_spectral_wind, coeffs_to_nn_input
from oed_solver import (CachedEigensolver, build_problem,
                        compute_eigendecomposition,
                        oed_objective_and_grad,
                        create_initial_path,
                        reset_bbt_cache)
from oed_core import reset_bbt_cache as reset_cache
from utils import initialize_problem


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate OED training data')
    parser.add_argument('--n-samples', type=int, default=10,
                        help='Number of wind samples per position')
    parser.add_argument('--r-wind', type=int, default=3,
                        help='Number of spectral modes for wind')
    parser.add_argument('--sigma', type=float, default=1.0,
                        help='Wind amplitude scale')
    parser.add_argument('--alpha', type=float, default=2.0,
                        help='Wind spectral decay rate')
    parser.add_argument('--mean-vx-std', type=float, default=0.5,
                        help='Std dev for mean wind sampling')
    parser.add_argument('--position-std', type=float, default=0.15,
                        help='Std dev for drone position sampling')
    parser.add_argument('--output', type=str, default='oed_training_data.pkl',
                        help='Output filename')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--maxiter', type=int, default=80,
                        help='Max optimization iterations')
    return parser.parse_args()


def sample_mean_vx(std, base_mean=0.5):
    """Sample mean wind velocity from Gaussian."""
    return np.random.normal(base_mean, std)


def sample_drone_position(std):
    """Sample drone position from Gaussian, clipped to [0.1, 0.9]."""
    pos = np.random.normal(0.5, std, size=2)
    return np.clip(pos, 0.1, 0.9)


def main():
    args = parse_args()
    np.random.seed(args.seed)
    
    print("="*60)
    print("OED TRAINING DATA GENERATION")
    print("="*60)
    print(f"Wind samples: {args.n_samples}")
    print(f"Mean vx std: {args.mean_vx_std}")
    print(f"Position std: {args.position_std}")
    print(f"Output: {args.output}")
    print("="*60)
    
    # Initialize the problem (creates Vh, prior, true IC, etc.)
    global Vh, prior, true_initial_condition, wind_velocity
    Vh, prior, true_initial_condition, wind_velocity = initialize_problem(
        nx, ny, gamma, delta)
    
    print(f"Number of dofs: {Vh.dim()}")
    
    # Initialize eigensolver
    eigsolver = CachedEigensolver()
    
    # Calculate total samples
    total_samples = args.n_samples
    training_data = []
    
    t_start_all = time.time()
    
    for sample_idx in range(args.n_samples):
        print(f"\n--- Sample {sample_idx+1}/{args.n_samples} ---")
        
        # Sample mean wind velocity
        mean_vx = sample_mean_vx(args.mean_vx_std, base_mean=0.5)
        
        # Sample drone position
        c_init = sample_drone_position(args.position_std)
        
        # Sample wind field
        wind_velocity, wind_coeffs = sample_spectral_wind(
            mesh, r_wind=args.r_wind, sigma=args.sigma,
            alpha=args.alpha, mean_vx=mean_vx, 
            mean_vy=0.0, seed=args.seed + sample_idx)
        
        # Create initial path
        m0 = create_initial_path(c_init, K)
        
        # Reset caches
        reset_cache()
        eigsolver.reset()
        
        # Optimization history (not stored)
        opt_history = {'eig': [], 'bdy': [], 'spd': [], 'J': [], 'time': []}
        
        # Objective function wrapper
        def obj_and_grad(m):
            (J, grad), stats = oed_objective_and_grad(
                m, Vh, prior, wind_velocity,
                observation_times, simulation_times,
                K, omegas, noise_variance, r_modes,
                obstacles, eigsolver,
                zeta_bdy, zeta_speed, v_max
            )
            # Store history
            for key in stats:
                if key in opt_history:
                    opt_history[key].append(stats[key])
            opt_history['J'].append(J)
            return J, grad
        
        t0 = time.time()
        try:
            result = minimize(
                obj_and_grad, m0,
                jac=True, method='L-BFGS-B', bounds=bounds,
                options={'maxiter': args.maxiter, 'disp': False,
                        'ftol': 1e-10, 'maxls': 40}
            )
            
            m_opt = result.x
            
            # Compute final EIG
            prob_opt, _, _ = build_problem(
                m_opt, Vh, prior, wind_velocity,
                observation_times, simulation_times,
                K, omegas, noise_variance
            )
            _, _, eig_opt = compute_eigendecomposition(
                prob_opt, prior, r_modes, eigsolver)
            
            elapsed = time.time() - t0
            
            # Create NN input (wind coefficients + drone position)
            nn_input = coeffs_to_nn_input(wind_coeffs, c_init)
            
            training_data.append({
                'sample_idx': sample_idx,
                'c_init': c_init.copy(),
                'mean_vx': mean_vx,
                'nn_input': nn_input.copy(),
                'nn_output': m_opt.copy(),
                'eig_opt': eig_opt,
                'converged': result.success,
                'n_iter': result.nit,
                'wind_coeffs': wind_coeffs,
                'time': elapsed,
            })
            
            print(f"  c0=({c_init[0]:.3f},{c_init[1]:.3f}) "
                  f"mean_vx={mean_vx:.3f} "
                  f"EIG={eig_opt:.2f} "
                  f"iter={result.nit} "
                  f"[{elapsed:.1f}s]")
            
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  FAILED: {e} [{elapsed:.1f}s]")
    
    # Save results
    total_time = time.time() - t_start_all
    print("\n" + "="*60)
    print("GENERATION COMPLETE")
    print(f"Samples: {len(training_data)} / {total_samples}")
    print(f"Total time: {total_time/3600:.2f} hours")
    print(f"Avg time per sample: {total_time/max(len(training_data),1):.1f}s")
    
    # Statistics
    eigs = [d['eig_opt'] for d in training_data]
    conv = sum(d['converged'] for d in training_data)
    print(f"EIG range: {min(eigs):.2f} to {max(eigs):.2f}")
    print(f"Converged: {conv}/{len(training_data)}")
    
    # Save to file
    with open(args.output, 'wb') as f:
        pickle.dump(training_data, f)
    print(f"Saved to {args.output}")
    print("="*60)


if __name__ == "__main__":
    main()