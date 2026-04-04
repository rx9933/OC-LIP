"""Training data generation for OED with varying wind and drone positions."""

import numpy as np
import dolfin as dl
import pickle
import time, os
from scipy.optimize import minimize
# Add at the VERY beginning of your script, before importing dolfin
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

# Try to disable MPI in FEniCS
os.environ['DOLFIN_NOPETSC'] = '1'  # May not work depending on compilation

from config import *
from fourier_utils import fourier_frequencies, xbar_coeffs_to_m
from wind_utils import sample_spectral_wind, coeffs_to_nn_input, nn_input_dim
from fe_utils import reset_cached_bbt
from oed_objective import CachedEigensolver, oed_objective_and_grad
from fe_setup import setup_fe_spaces, setup_prior

import warnings
warnings.filterwarnings('ignore')

# Suppress FEniCS logging
import logging
logging.getLogger('FFC').setLevel(logging.WARNING)
logging.getLogger('UFL').setLevel(logging.WARNING)
dl.set_log_active(False)

import argparse
import sys
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, ScalarFormatter

# Only parse arguments if not in Jupyter
if any('ipykernel' in arg for arg in sys.argv):
    # We're in Jupyter - use default values
    args = argparse.Namespace(
        job_id=0,
        n_samples=N_SAMPLES,
        output_prefix='oed_training_data',
        plot=False  # Default to no plotting in Jupyter unless specified
    )
else:
    # We're in command line - parse arguments
    parser = argparse.ArgumentParser(description='Generate OED training data')
    parser.add_argument('--job_id', type=int, default=0, 
                        help='Job ID for parallel runs (used as seed offset)')
    parser.add_argument('--n_samples', type=int, default=N_SAMPLES,
                        help='Number of samples to generate in this job')
    parser.add_argument('--output_prefix', type=str, default='oed_training_data',
                        help='Prefix for output file')
    parser.add_argument('--plot', action='store_true',
                        help='Plot optimization history for each sample')
    args = parser.parse_args()

def sample_mean_vx(mean=MEAN_VX_MEAN, std=MEAN_VX_STD):
    """
    Sample mean_vx from Gaussian distribution.
    
    Parameters
    ----------
    mean : float
        Mean of distribution
    std : float
        Standard deviation of distribution
        
    Returns
    -------
    float
        Sampled mean_vx
    """
    return np.random.normal(mean, std)


def sample_drone_position(mean=DRONE_POS_MEAN, std=DRONE_POS_STD, bounds=(0.1, 0.9)):
    """
    Sample drone starting position from Gaussian, clipped to bounds.
    
    Parameters
    ----------
    mean : float
        Mean of distribution (same for x and y)
    std : float
        Standard deviation
    bounds : tuple
        (lower, upper) bounds for position
        
    Returns
    -------
    np.ndarray
        (2,) array of [x, y]
    """
    x = np.random.normal(mean, std)
    y = np.random.normal(mean, std)
    # Clip to bounds
    x = np.clip(x, bounds[0], bounds[1])
    y = np.clip(y, bounds[0], bounds[1])
    return np.array([x, y])


def create_initial_guess(c_init, K, radius_mean=0.1, radius_std=0.03, seed=None):
    """
    Create initial Fourier parameters that generate a circular path around the mean.
    The radius follows a Gaussian distribution, and the path is constrained to stay
    within the domain boundaries.
    
    Parameters
    ----------
    c_init : np.ndarray
        (2,) initial drone position (mean position of circle)
    K : int
        Number of Fourier modes (only first mode used for circle)
    radius_mean : float
        Mean radius of the circle
    radius_std : float
        Standard deviation of radius distribution
    seed : int or None
        Random seed
        
    Returns
    -------
    np.ndarray
        Flat Fourier parameter vector that generates a circular path
    """
    if seed is not None:
        np.random.seed(seed)
    
    m0 = np.zeros(4*K + 2)
    m0[0] = c_init[0]  # x̄
    m0[1] = c_init[1]  # ȳ
    
    # Sample radius from Gaussian (ensuring it's positive)
    radius = abs(np.random.normal(radius_mean, radius_std))
    
    # For a perfect circle with period T, we need:
    # x(t) = x̄ + R cos(ωt)
    # y(t) = ȳ + R sin(ωt)
    # This corresponds to Fourier coefficients:
    # θ₁ = R, φ₁ = 0, ψ₁ = 0, η₁ = R
    # (with ω = 2π/T for the first mode)
    
    # First mode creates the circle
    m0[2] = radius      # θ₁ - cosine coefficient for x
    m0[3] = 0.0         # φ₁ - sine coefficient for x (zero for circle)
    m0[4] = 0.0         # ψ₁ - cosine coefficient for y (zero for circle)
    m0[5] = radius      # η₁ - sine coefficient for y
    
    # Check if the circle would go out of bounds and adjust if needed
    x_min = m0[0] - radius
    x_max = m0[0] + radius
    y_min = m0[1] - radius
    y_max = m0[1] + radius
    
    # If out of bounds, scale down the radius
    bounds = [0.1, 0.9]  # Domain boundaries with margin
    
    if x_min < bounds[0] or x_max > bounds[1] or y_min < bounds[0] or y_max > bounds[1]:
        # Calculate maximum allowed radius
        max_radius_x = min(m0[0] - bounds[0], bounds[1] - m0[0])
        max_radius_y = min(m0[1] - bounds[0], bounds[1] - m0[1])
        max_radius = min(max_radius_x, max_radius_y, radius)
        
        if max_radius < radius:
            # Rescale
            scale_factor = max_radius / radius
            m0[2] = radius * scale_factor
            m0[5] = radius * scale_factor
            print(f"  Radius scaled from {radius:.3f} to {radius*scale_factor:.3f} to fit in bounds")
    
    # Higher modes get small random perturbations (optional)
    for k in range(1, K):
        # Much smaller amplitudes for higher modes to maintain approximate circle
        scale = radius * 0.1 / (k + 1)  # Decaying with mode number
        m0[2 + 4*k] = np.random.normal(0, scale)      # θ_k
        m0[3 + 4*k] = np.random.normal(0, scale)      # φ_k
        m0[4 + 4*k] = np.random.normal(0, scale)      # ψ_k
        m0[5 + 4*k] = np.random.normal(0, scale)      # η_k
    
    return m0


def plot_optimization_history(history, sample_info, save_dir='optimization_plots'):
    """
    Plot optimization history including objective and gradient.
    
    Parameters
    ----------
    history : dict
        Dictionary containing iteration history with keys:
        'iterations', 'J', 'grad_norm', 'eig', 'penalty', 'spd'
    sample_info : dict
        Information about the sample (seed, mean_vx, etc.)
    save_dir : str
        Directory to save plots
    """
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    iterations = history['iterations']
    J_vals = history['J']
    grad_norms = history['grad_norm']
    eig_vals = history.get('eig', [])
    penalty_vals = history.get('penalty', [])
    spd_vals = history.get('spd', [])
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Objective value
    ax1 = axes[0, 0]
    ax1.semilogy(iterations, np.array(J_vals) - J_vals[-1] + 1e-8, 'b-', linewidth=2, label='J - J_min')
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Objective (log scale)', fontsize=12)
    ax1.set_title(f'Objective Convergence\n(Seed={sample_info["seed"]}, mean_vx={sample_info["mean_vx"]:.3f})', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Gradient norm
    ax2 = axes[0, 1]
    ax2.semilogy(iterations, grad_norms, 'r-', linewidth=2, label='||∇J||')
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('Gradient Norm (log scale)', fontsize=12)
    ax2.set_title('Gradient Convergence', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: EIG, Penalty, and SPD components
    ax3 = axes[1, 0]
    if eig_vals:
        ax3.plot(iterations, eig_vals, 'g-', linewidth=2, label='EIG')
    if penalty_vals:
        ax3.plot(iterations, penalty_vals, 'm-', linewidth=2, label='Penalty')
    if spd_vals:
        ax3.plot(iterations, spd_vals, 'c-', linewidth=2, label='SPD')
    ax3.set_xlabel('Iteration', fontsize=12)
    ax3.set_ylabel('Component Values', fontsize=12)
    ax3.set_title('Optimization Components', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Plot 4: Combined view - scaled components
    ax4 = axes[1, 1]
    
    # Normalize for comparison
    if J_vals:
        J_norm = (np.array(J_vals) - J_vals[0]) / (J_vals[-1] - J_vals[0] + 1e-10)
        ax4.plot(iterations, J_norm, 'b-', linewidth=2, label='J (normalized)')
    
    if grad_norms:
        grad_norm_norm = (np.array(grad_norms) - grad_norms[0]) / (grad_norms[-1] - grad_norms[0] + 1e-10)
        ax4.plot(iterations, grad_norm_norm, 'r-', linewidth=2, label='||∇J|| (normalized)')
    
    ax4.set_xlabel('Iteration', fontsize=12)
    ax4.set_ylabel('Normalized Value', fontsize=12)
    ax4.set_title('Normalized Convergence Comparison', fontsize=12)
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.tight_layout()
    
    # Save figure
    filename = f"{save_dir}/optimization_seed{sample_info['seed']}_meanvx{sample_info['mean_vx']:.3f}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"    Saved optimization plot to {filename}")


def generate_training_sample(seed, mesh, Vh, prior, simulation_times,
                             observation_times, t_param, K, omegas,
                             r_modes, noise_variance, bounds, wind_params=None,
                             plot_history=False):
    """
    Generate a single training sample.
    
    Parameters
    ----------
    seed : int
        Random seed
    mesh : dolfin Mesh
        Computational mesh
    Vh : dolfin FunctionSpace
        Finite element space
    prior : BiLaplacianPrior
        Prior distribution
    simulation_times : np.ndarray
        Times for simulation
    observation_times : np.ndarray
        Times for observations
    t_param : np.ndarray
        Observation times (usually same as observation_times)
    K : int
        Number of Fourier modes
    omegas : np.ndarray
        Angular frequencies
    r_modes : int
        Number of eigenvalues to keep
    noise_variance : float
        Noise variance for observations
    bounds : list
        List of (lb, ub) tuples for optimization
    wind_params : dict or None
        Wind parameters (if None, use defaults)
    plot_history : bool
        Whether to plot optimization history
        
    Returns
    -------
    tuple
        (sample_dict, success_bool)
    """
    if wind_params is None:
        wind_params = {
            'r_wind': WIND_R,
            'sigma': WIND_SIGMA,
            'alpha': WIND_ALPHA,
            'mean_vx': sample_mean_vx(),
            'mean_vy': WIND_MEAN_VY
        }
    
    # Sample drone position
    c_init = sample_drone_position()
    
    # Sample wind field
    wind_velocity, wind_coeffs = sample_spectral_wind(
        mesh, 
        r_wind=wind_params['r_wind'],
        sigma=wind_params['sigma'],
        alpha=wind_params['alpha'],
        mean_vx=wind_params['mean_vx'],
        mean_vy=wind_params['mean_vy'],
        seed=seed
    )
    
    # Create initial guess
    m0 = create_initial_guess(c_init, K)
    
    # Reset solver state
    reset_cached_bbt()
    eigsolver = CachedEigensolver()
    
    t0 = time.time()
    
    # Store optimization history
    history = {
        'iterations': [],
        'J': [],
        'grad_norm': [],
        'eig': [],
        'penalty': [],
        'spd': []
    }
    
    try:
        # ===== COMPUTE INITIAL EIG =====
        print(f"    Computing initial EIG...")
        # Compute initial EIG (with penalties to match optimization)
        J0, grad0, eig_init, pen_init, spd_init, _ = oed_objective_and_grad(c_init,
            m0, Vh, mesh, prior, simulation_times, observation_times,
            wind_velocity, K, omegas, r_modes, noise_variance, t_param,
            eigsolver, obstacles=None, include_penalties=False 
        )
        print(f"    Initial EIG = {eig_init:.2f}")
        print(f"    Initial J = {J0:.6f}, |grad| = {np.linalg.norm(grad0):.6f}")
        print(f"    Initial penalty = {pen_init:.6f}, SPD penalty = {spd_init:.6f}")
        print(f"    First 5 gradient components: {grad0[:5]}")
        sys.stdout.flush()
        
        # ===== SET UP OPTIMIZATION WITH LOGGING =====
        # Create counter for iterations (use list for mutable state)
        iter_counter = [0]
        
        def objective(m):
            iter_counter[0] += 1
            
            J, grad, eig_val, pen_val, spd_val, elapsed = oed_objective_and_grad(c_init,
                m, Vh, mesh, prior, simulation_times, observation_times,
                wind_velocity, K, omegas, r_modes, noise_variance, t_param,
                eigsolver, obstacles=None, include_penalties=True
            )
            
            grad_norm = np.linalg.norm(grad)
            param_norm = np.linalg.norm(m)
            
            # Store history
            history['iterations'].append(iter_counter[0])
            history['J'].append(J)
            history['grad_norm'].append(grad_norm)
            history['eig'].append(eig_val)
            history['penalty'].append(pen_val)
            history['spd'].append(spd_val)
            
            # Print every iteration
            print(f"    [Iter {iter_counter[0]:3d}] J = {J:12.8f} | "
                  f"|grad| = {grad_norm:10.6f} | "
                  f"eig={eig_val:8.2f} | "
                  f"pen={pen_val:8.4f} | "
                  f"spd={spd_val:8.4f} | "
                  f"|m|={param_norm:8.4f}")
            
            # Print gradient components occasionally to see pattern
            if iter_counter[0] % 10 == 0:  # Every 10 iterations
                print(f"      grad[0:5] = {grad[:5]}")
                # Also show gradient norm reduction rate
                if len(history['grad_norm']) > 10:
                    reduction = history['grad_norm'][-10] / grad_norm
                    print(f"      Gradient reduction (last 10 iters): {reduction:.3f}x")
            
            sys.stdout.flush()
            
            return J, grad
        
        # Define callback for additional monitoring
        def callback(xk):
            print(f"      → Callback: iter {iter_counter[0]}, param norm={np.linalg.norm(xk):.4f}")
            sys.stdout.flush()
        
        # ===== RUN OPTIMIZATION =====
        print(f"    Starting optimization with L-BFGS-B...")
        sys.stdout.flush()
        
        result = minimize(
            objective, m0,
            jac=True, method='L-BFGS-B', bounds=bounds,
            callback=callback,
            options={'maxiter': OPT_MAXITER, 'disp': True,
                     'ftol': OPT_FTOL, 'maxls': OPT_MAXLS, 'maxfun': OPT_MAXFUN}
        )
        
        m_opt = result.x
        
        # ===== OPTIMIZATION SUMMARY =====
        print(f"    Optimization complete:")
        print(f"      Success: {result.success}")
        print(f"      Iterations: {result.nit}")
        print(f"      Function evaluations: {result.nfev}")
        print(f"      Final J: {result.fun:.12f}")
        
        # Compute final gradient
        J_final, grad_final, eig_final, pen_final, spd_final, _ = oed_objective_and_grad(c_init,
            m_opt, Vh, mesh, prior, simulation_times, observation_times,
            wind_velocity, K, omegas, r_modes, noise_variance, t_param,
            eigsolver, obstacles=None, include_penalties=True
        )
        final_grad_norm = np.linalg.norm(grad_final)
        print(f"      Final gradient norm: {final_grad_norm:.10f}")
        
        # Show improvement
        J_improvement = J0 - result.fun
        eig_improvement = eig_final - eig_init
        print(f"      Improvement: J: {J_improvement:.6f} ({(J_improvement/abs(J0)*100):.2f}%)")
        print(f"      EIG improvement: {eig_improvement:.2f} ({(eig_improvement/abs(eig_init)*100):.2f}%)")
        
        # Show gradient reduction
        if history['grad_norm']:
            initial_grad_norm = history['grad_norm'][0]
            grad_reduction = initial_grad_norm / final_grad_norm if final_grad_norm > 0 else np.inf
            print(f"      Gradient reduction: {grad_reduction:.2f}x")
        
        sys.stdout.flush()
        
        # Create plot if requested
        if plot_history and history['iterations']:
            sample_info = {
                'seed': seed,
                'mean_vx': wind_params['mean_vx'],
                'eig_init': eig_init,
                'eig_final': eig_final,
                'J0': J0,
                'J_final': result.fun
            }
            plot_optimization_history(history, sample_info)
        
        # Compute final EIG without penalties for the sample
        eigsolver.reset()
        _, _, eig_opt_no_pen, _, _, _ = oed_objective_and_grad(c_init,
            m_opt, Vh, mesh, prior, simulation_times, observation_times,
            wind_velocity, K, omegas, r_modes, noise_variance, t_param,
            eigsolver, obstacles=None, include_penalties=False
        )
        
        elapsed = time.time() - t0
        
        # NN input and output
        nn_input = coeffs_to_nn_input(wind_coeffs, c_init)
        nn_output = m_opt.copy()
        
        # Sample dictionary with all information
        sample = {
            'seed': seed,
            'c_init': c_init.copy(),
            'mean_vx': wind_params['mean_vx'],
            'nn_input': nn_input,
            'nn_output': nn_output,
            'eig_init': eig_init,
            'eig_opt': eig_opt_no_pen,
            'eig_gain': eig_opt_no_pen - eig_init,
            'converged': result.success,
            'wind_coeffs': wind_coeffs,
            'time': elapsed,
            'nit': result.nit,
            'nfev': result.nfev,
            'J0': J0,
            'J_final': result.fun,
            'grad_norm_initial': np.linalg.norm(grad0),
            'grad_norm_final': final_grad_norm,
            'penalty_initial': pen_init,
            'penalty_final': pen_final,
            'spd_initial': spd_init,
            'spd_final': spd_final,
            'optimization_history': history if plot_history else None,  # Store history if plotting
        }
        
        return sample, result.success
        
    except Exception as e:
        elapsed = time.time() - t0
        print(f"    FAILED at iteration {iter_counter[0] if 'iter_counter' in locals() else 0}: {e}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        return None, False

def generate_training_data(n_samples=args.n_samples, 
                          output_file=f"{args.output_prefix}_job{args.job_id}.pkl",
                          plot_optimization=args.plot):
    """
    Generate training data for OED.
    """
    print("="*60)
    print("  OED TRAINING DATA GENERATION")
    print("="*60)
    print(f"  Job ID: {args.job_id}")
    print(f"  Number of samples: {n_samples}")
    print(f"  Plot optimization: {plot_optimization}")
    print(f"  Mean_vx distribution: N({MEAN_VX_MEAN}, {MEAN_VX_STD}²)")
    print(f"  Drone position distribution: N({DRONE_POS_MEAN}, {DRONE_POS_STD}²)")
    print("="*60)
    
    # Use job_id as part of the base seed to ensure unique seeds across jobs
    base_seed = int(time.time()) + args.job_id * 1000000
    print(f"Using base seed: {base_seed} (based on current time + job_id offset)")
    max_attempts = n_samples * 3

    # Setup
    print("Calling setup_fe_spaces()...")
    sys.stdout.flush()
    
    mesh, Vh, wind_velocity = setup_fe_spaces()
    print("setup_fe_spaces() completed successfully")
    sys.stdout.flush()
    
    print("Calling setup_prior()...")
    sys.stdout.flush()
    prior = setup_prior(Vh)
    print("setup_prior() completed successfully")
    sys.stdout.flush()
    
    # Compute frequencies
    print("Computing frequencies...")
    omegas = fourier_frequencies(TY, K)
    print(f"Frequencies computed: {omegas}")
    sys.stdout.flush()

    print(f"\n  Mesh DOFs: {Vh.dim()}")
    print(f"  Fourier modes: {K} (parameter dim: {4*K + 2})")
    print(f"  NN input dim: {nn_input_dim(WIND_R)}")
    sys.stdout.flush()

    training_data = []
    total = n_samples
    count = 0
    successful = 0
    t_start_all = time.time()
    total_attempts = 0
    
    while successful < n_samples and total_attempts < max_attempts:
        total_attempts += 1
        
        # Generate seed for this attempt
        seed = base_seed + total_attempts
        np.random.seed(seed)
        count += 1
        
        # Sample mean_vx
        mean_vx = sample_mean_vx()
        
        wind_params = {
            'r_wind': WIND_R,
            'sigma': WIND_SIGMA,
            'alpha': WIND_ALPHA,
            'mean_vx': mean_vx,
            'mean_vy': WIND_MEAN_VY
        }
        
        print(f"\n{'='*60}")
        print(f"  SAMPLE {count}/{total} | seed={seed} | mean_vx={mean_vx:.3f}")
        print(f"{'='*60}")
        sys.stdout.flush()

        sample, success = generate_training_sample(
            seed, mesh, Vh, prior, SIMULATION_TIMES, OBSERVATION_TIMES,
            OBSERVATION_TIMES, K, omegas, R_MODES, NOISE_VARIANCE,
            BOUNDS, wind_params,
            plot_history=plot_optimization  # Pass plot flag
        )
        
        if success and sample is not None:
            training_data.append(sample)
            successful += 1
            print(f"\n  ✓ SAMPLE {successful}/{total} COMPLETED")
            print(f"    → EIG: {sample['eig_init']:.2f} → {sample['eig_opt']:.2f} "
                  f"(gain={sample['eig_gain']:.2f})")
            print(f"    → J: {sample['J0']:.4f} → {sample['J_final']:.4f}")
            print(f"    → Gradient: {sample['grad_norm_initial']:.4f} → {sample['grad_norm_final']:.4f}")
            print(f"    → Converged: {sample['converged']}")
            print(f"    → Time: {sample['time']:.1f}s")
            print(f"    → Iterations: {sample['nit']}, F-evals: {sample['nfev']}")
            sys.stdout.flush()
            
            # Save incrementally
            with open(output_file, 'wb') as f:
                pickle.dump(training_data, f)
            print(f"  Saved to {output_file}")
        else:
            print(f"  ✗ SAMPLE {count} FAILED")
            sys.stdout.flush()

    total_time = time.time() - t_start_all
    
    print("\n" + "="*60)
    print(f"  JOB {args.job_id} COMPLETE")
    print("="*60)
    print(f"  Successful samples: {successful} / {total}")
    print(f"  Success rate: {100*successful/total:.1f}%")
    print(f"  Total time: {total_time/3600:.1f} hours")
    print(f"  Avg time per sample: {total_time/max(successful,1):.1f}s")
    print("="*60)
    
    # Quick stats
    if successful > 0:
        eigs = [d['eig_opt'] for d in training_data]
        gains = [d['eig_gain'] for d in training_data]
        times = [d['time'] for d in training_data]
        converged = sum(d['converged'] for d in training_data)
        
        print(f"\n  Statistics:")
        print(f"    EIG range: {min(eigs):.2f} to {max(eigs):.2f}")
        print(f"    EIG mean: {np.mean(eigs):.2f} ± {np.std(eigs):.2f}")
        print(f"    Gain mean: {np.mean(gains):.2f} ± {np.std(gains):.2f}")
        print(f"    Converged: {converged}/{successful}")
        print(f"    Avg time: {np.mean(times):.1f}s ± {np.std(times):.1f}s")
        
        # Plot summary of all samples if requested
        if plot_optimization and successful > 1:
            plot_summary_statistics(training_data)
    
    sys.stdout.flush()
    
    return training_data


def plot_summary_statistics(training_data):
    """
    Plot summary statistics across all successful samples.
    
    Parameters
    ----------
    training_data : list
        List of sample dictionaries
    """
    # Create summary figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Extract data
    seeds = [d['seed'] for d in training_data]
    eig_init = [d['eig_init'] for d in training_data]
    eig_opt = [d['eig_opt'] for d in training_data]
    eig_gain = [d['eig_gain'] for d in training_data]
    times = [d['time'] for d in training_data]
    nits = [d['nit'] for d in training_data]
    grad_init = [d['grad_norm_initial'] for d in training_data]
    grad_final = [d['grad_norm_final'] for d in training_data]
    mean_vx = [d['mean_vx'] for d in training_data]
    
    # Plot 1: EIG before and after
    ax1 = axes[0, 0]
    x = np.arange(len(training_data))
    width = 0.35
    ax1.bar(x - width/2, eig_init, width, label='Initial', alpha=0.7)
    ax1.bar(x + width/2, eig_opt, width, label='Optimized', alpha=0.7)
    ax1.set_xlabel('Sample Index')
    ax1.set_ylabel('EIG')
    ax1.set_title('EIG Before and After Optimization')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: EIG gain vs mean_vx
    ax2 = axes[0, 1]
    scatter = ax2.scatter(mean_vx, eig_gain, c=times, s=50, cmap='viridis', alpha=0.6)
    ax2.set_xlabel('Mean Wind Velocity (vx)')
    ax2.set_ylabel('EIG Gain')
    ax2.set_title('EIG Gain vs Wind Velocity')
    ax2.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('Time (s)')
    
    # Plot 3: Gradient reduction
    ax3 = axes[1, 0]
    grad_reduction = np.array(grad_init) / np.array(grad_final)
    ax3.semilogy(range(len(training_data)), grad_init, 'b-o', label='Initial', markersize=4)
    ax3.semilogy(range(len(training_data)), grad_final, 'r-o', label='Final', markersize=4)
    ax3.set_xlabel('Sample Index')
    ax3.set_ylabel('Gradient Norm (log scale)')
    ax3.set_title('Gradient Norm Reduction')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Optimization time vs iterations
    ax4 = axes[1, 1]
    ax4.scatter(nits, times, c=eig_gain, s=50, cmap='plasma', alpha=0.6)
    ax4.set_xlabel('Number of Iterations')
    ax4.set_ylabel('Time (s)')
    ax4.set_title('Optimization Time vs Iterations')
    ax4.grid(True, alpha=0.3)
    cbar2 = plt.colorbar(scatter, ax=ax4)
    cbar2.set_label('EIG Gain')
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs('optimization_plots', exist_ok=True)
    plt.savefig('optimization_plots/summary_statistics.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n  Saved summary statistics plot to optimization_plots/summary_statistics.png")


# Main execution
if __name__ == "__main__":
    generate_training_data(plot_optimization=args.plot)