import numpy as np
from gaussian_samplers import *
from solvers import *
from nn_helper import *

K, prior, r_modes

def generate_training_sample(seed,
                             r_wind,
                             sigma_wind,
                             alpha_wind,
                             mean_vx_mu,
                             mean_vx_sigma,
                             drone_mu,
                             drone_sigma,
                             bounds):
    """
    Generate ONE training sample.
    """
    np.random.seed(seed)

    # --- Sample stochastic inputs ---
    mean_vx = sample_mean_vx(mean_vx_mu, mean_vx_sigma)
    c_init  = sample_drone_position(drone_mu, drone_sigma)

    # --- Sample wind field ---
    wind_velocity, wind_coeffs = sample_spectral_wind(
        mesh,
        r_wind=r_wind,
        sigma=sigma_wind,
        alpha=alpha_wind,
        mean_vx=mean_vx,
        seed=seed
    )

    # --- Run OED ---
    result = run_single_oed_sample(
        wind_coeffs, wind_velocity, c_init,
        bounds, K, prior, r_modes
    )

    # --- NN input/output ---
    nn_input  = coeffs_to_nn_input(wind_coeffs, c_init)
    nn_output = result['m_opt']

    return {
        'seed': seed,
        'c_init': c_init,
        'mean_vx': mean_vx,
        'nn_input': nn_input,
        'nn_output': nn_output,
        'eig_opt': result['eig_opt'],
        'converged': result['converged'],
        'wind_coeffs': wind_coeffs,
        'time': result['time']
    }
def generate_training_dataset(n_samples,
                             r_wind=3,
                             sigma_wind=1.0,
                             alpha_wind=2.0,
                             mean_vx_mu=0.5,
                             mean_vx_sigma=0.2,
                             drone_mu=np.array([0.5, 0.5]),
                             drone_sigma=0.2,
                             save_path="oed_training_data.pkl"):
    """
    Generate full dataset with Gaussian sampling.
    """
    global obstacles

    # Disable obstacles for training
    _obstacles_backup = obstacles
    obstacles = []

    # Bounds
    lb = np.zeros(4*K + 2)
    ub = np.zeros(4*K + 2)
    lb[0:2] = 0.1
    ub[0:2] = 0.9
    for kk in range(K):
        for j in range(4):
            lb[2 + 4*kk + j] = -0.3
            ub[2 + 4*kk + j] =  0.3
    bounds = list(zip(lb, ub))

    training_data = []
    t_start = time.time()

    for i in range(n_samples):
        try:
            sample = generate_training_sample(
                seed=i,
                r_wind=r_wind,
                sigma_wind=sigma_wind,
                alpha_wind=alpha_wind,
                mean_vx_mu=mean_vx_mu,
                mean_vx_sigma=mean_vx_sigma,
                drone_mu=drone_mu,
                drone_sigma=drone_sigma,
                bounds=bounds
            )

            training_data.append(sample)

            print(f"[{i+1}/{n_samples}] "
                  f"vx={sample['mean_vx']:.2f} "
                  f"c0=({sample['c_init'][0]:.2f},{sample['c_init'][1]:.2f}) "
                  f"EIG={sample['eig_opt']:.2f} "
                  f"conv={sample['converged']} "
                  f"[{sample['time']:.1f}s]")

        except Exception as e:
            print(f"[{i+1}/{n_samples}] FAILED: {e}")

    # Restore obstacles
    obstacles = _obstacles_backup

    # Save
    total_time = time.time() - t_start
    with open(save_path, 'wb') as f:
        pickle.dump(training_data, f)

    print("\n" + "="*60)
    print(f"Dataset complete: {len(training_data)}/{n_samples}")
    print(f"Total time: {total_time/3600:.2f} hours")
    print("="*60)

    return training_data

data = generate_training_dataset(
    n_samples=200,
    mean_vx_mu=0.5,
    mean_vx_sigma=0.3,         
    drone_mu=np.array([0.5, 0.5]),
    drone_sigma=0.25           
)