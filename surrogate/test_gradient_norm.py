"""Check gradient norms at stored optimal points."""
import sys, os
import numpy as np
import dolfin as dl

sys.path.append('../')
sys.path.append('generate_data/')

from generate_data.config import *
from generate_data.fourier_utils import fourier_frequencies
from generate_data.wind_utils import spectral_wind_to_field
from generate_data.oed_objective import CachedEigensolver, oed_objective_and_grad
from generate_data.fe_setup import setup_prior

mesh = dl.UnitSquareMesh(NX, NY)
Vh = dl.FunctionSpace(mesh, 'Lagrange', 1)
prior = setup_prior(Vh)
omegas = fourier_frequencies(TY, K)

mq_data = np.load('data/mq_data_reduced.npz')
m_data = mq_data['m']
v_mean = mq_data['v_mean']
v_coeff = mq_data['v_coeff']
x_data = mq_data['x']

r_wind = 3
for idx in [942, 943, 944]:
    m_opt = m_data[idx]
    c_init = x_data[idx]
    
    a_ij = v_coeff[idx, :9].reshape(3, 3)
    b_ij = v_coeff[idx, 9:].reshape(3, 3)
    wind_coeffs = {
        'a_ij': a_ij, 'b_ij': b_ij,
        'mean_vx': v_mean[idx, 0], 'mean_vy': v_mean[idx, 1],
        'r_wind': r_wind, 'sigma': 1.0, 'alpha': 2.0
    }
    wind_velocity, _ = spectral_wind_to_field(mesh, wind_coeffs)
    
    eigsolver = CachedEigensolver()
    J, grad, eig_val, pen_val, spd_val, _ = oed_objective_and_grad(
        c_init, m_opt, Vh, mesh, prior, SIMULATION_TIMES, OBSERVATION_TIMES,
        wind_velocity, K, omegas, R_MODES, NOISE_VARIANCE,
        OBSERVATION_TIMES, eigsolver, include_penalties=True
    )
    
    print(f"Sample {idx}: EIG={eig_val:.4f}, ||grad||={np.linalg.norm(grad):.4e}, "
          f"max|grad|={np.max(np.abs(grad)):.4e}, J={J:.4f}")
