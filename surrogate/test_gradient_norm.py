import sys, os, pickle
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

# Load new sample
with open('generate_data/oed_training_data_job999.pkl', 'rb') as f:
    data = pickle.load(f)

for i, sample in enumerate(data):
    m_opt = sample['nn_output']
    c_init = sample['c_init']
    wind_coeffs = sample['wind_coeffs']
    wind_velocity, _ = spectral_wind_to_field(mesh, wind_coeffs)

    eigsolver = CachedEigensolver()
    J, grad, eig_val, pen_val, spd_val, _ = oed_objective_and_grad(
        c_init, m_opt, Vh, mesh, prior, SIMULATION_TIMES, OBSERVATION_TIMES,
        wind_velocity, K, omegas, R_MODES, NOISE_VARIANCE,
        OBSERVATION_TIMES, eigsolver, include_penalties=True
    )

    print(f"Sample {i}: EIG={eig_val:.4f}, ||grad||={np.linalg.norm(grad):.4e}, "
          f"max|grad|={np.max(np.abs(grad)):.4e}, nit={sample['nit']}, "
          f"conv={sample['converged']}")
