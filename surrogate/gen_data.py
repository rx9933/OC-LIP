# ═══════════════════════════════════════════════════════════════
# TRAINING DATA GENERATION
# ═══════════════════════════════════════════════════════════════
import pickle
import time

# Fixed parameters for all samples
r_wind_train = 3
sigma_train = 1.0
alpha_train = 2.0
n_samples = 200

# Drone starting positions to vary
start_positions = [
    [0.35, 0.70],  # near plume
    [0.50, 0.50],  # center
    [0.20, 0.50],  # left
    [0.70, 0.50],  # right
]

# Bounds
lb = np.zeros(4*K + 2)
ub = np.zeros(4*K + 2)
lb[0] = 0.1;  ub[0] = 0.9
lb[1] = 0.1;  ub[1] = 0.9
for kk in range(K):
    for j in range(4):
        lb[2 + 4*kk + j] = -0.3
        ub[2 + 4*kk + j] =  0.3
bounds_train = list(zip(lb, ub))

# No obstacles for training data
_obstacles_backup = obstacles
obstacles = []

training_data = []
total = n_samples * len(start_positions)
count = 0
t_start_all = time.time()

for seed in range(n_samples):
    # Sample one wind field (reused across starting positions)
    wind_velocity, wind_coeffs = sample_spectral_wind(
        mesh, r_wind=r_wind_train, sigma=sigma_train,
        alpha=alpha_train, mean_vx=0.5, seed=seed)

    for c_init in start_positions:
        count += 1

        # Initial guess centered at drone position
        m0 = np.zeros(4*K + 2)
        m0[0] = c_init[0]
        m0[1] = c_init[1]
        m0[2] = 0.05
        m0[5] = 0.05

        # Reset solver state
        _cached_bbt = None
        eigsolver.reset()
        opt_history = {'eig': [], 'bdy': [], 'spd': [], 'J': [], 'time': []}

        t0 = time.time()
        try:
            result = minimize(EIG_objective_and_grad, m0,
                              jac=True, method='L-BFGS-B', bounds=bounds_train,
                              options={'maxiter': 80, 'disp': False,
                                       'ftol': 1e-10, 'maxls': 40})

            m_opt = result.x
            prob_opt, _, _ = build_problem(m_opt)
            _, _, eig_opt = compute_eigendecomposition(prob_opt, prior, r_modes)

            elapsed = time.time() - t0

            # NN input and output
            nn_input = coeffs_to_nn_input(wind_coeffs, np.array(c_init))
            nn_output = m_opt.copy()

            training_data.append({
                'seed': seed,
                'c_init': np.array(c_init),
                'nn_input': nn_input,
                'nn_output': nn_output,
                'eig_opt': eig_opt,
                'converged': result.success,
                'wind_coeffs': wind_coeffs,
                'time': elapsed,
            })

            print(f"  [{count}/{total}] seed={seed:3d} c0=({c_init[0]:.2f},{c_init[1]:.2f}) "
                  f"EIG={eig_opt:.2f} conv={result.success} [{elapsed:.1f}s]")

        except Exception as e:
            elapsed = time.time() - t0
            print(f"  [{count}/{total}] seed={seed:3d} c0=({c_init[0]:.2f},{c_init[1]:.2f}) "
                  f"FAILED: {e} [{elapsed:.1f}s]")

# Restore obstacles
obstacles = _obstacles_backup

# Save
total_time = time.time() - t_start_all
print(f"\n{'='*60}")
print(f"  Training data generation complete")
print(f"  Samples: {len(training_data)} / {total}")
print(f"  Total time: {total_time/3600:.1f} hours")
print(f"  Avg time per sample: {total_time/max(len(training_data),1):.1f}s")
print(f"{'='*60}")

with open('oed_training_data.pkl', 'wb') as f:
    pickle.dump(training_data, f)
print(f"Saved to oed_training_data.pkl")

# Quick stats
eigs = [d['eig_opt'] for d in training_data]
conv = sum(d['converged'] for d in training_data)
print(f"\n  EIG range: {min(eigs):.2f} to {max(eigs):.2f}")
print(f"  Converged: {conv}/{len(training_data)}")