"""Configuration parameters for OED data generation."""

import numpy as np
import dolfin as dl

# ============================================================================
# MESH AND FEM
# ============================================================================
nx, ny = 60, 60
mesh = dl.UnitSquareMesh(nx, ny)

# ============================================================================
# TIME PARAMETERS
# ============================================================================
t_init = 0.0
t_final = 4.0
t_1 = 1.0
dt = 0.1
observation_dt = 0.2

simulation_times = np.arange(t_init, t_final + 0.5*dt, dt)
observation_times = np.arange(t_1, t_final + 0.5*dt, observation_dt)

# ============================================================================
# FOURIER PATH PARAMETERS
# ============================================================================
K = 3  # Fourier modes
Ty = (t_1, t_final)
omegas = np.array([2.0 * np.pi * (k + 1) / (Ty[1] - Ty[0]) for k in range(K)])

# ============================================================================
# PRIOR PARAMETERS
# ============================================================================
gamma = 1.0
delta = 8.0
r_modes = 20  # eigenvalues kept for EIG computation

# ============================================================================
# OED PARAMETERS
# ============================================================================
noise_variance = 1e-4

# Penalty parameters
zeta_bdy = 1000.0      # boundary penalty weight
zeta_speed = 500.0     # speed penalty weight
v_max = 0.5            # max sensor speed

# ============================================================================
# OPTIMIZATION BOUNDS
# ============================================================================
lb = np.zeros(4*K + 2)
ub = np.zeros(4*K + 2)
lb[0] = 0.1;  ub[0] = 0.9   # x̄
lb[1] = 0.1;  ub[1] = 0.9   # ȳ
for kk in range(K):
    for j in range(4):
        lb[2 + 4*kk + j] = -0.3
        ub[2 + 4*kk + j] =  0.3
bounds = list(zip(lb, ub))

# ============================================================================
# OBSTACLES (empty for data generation)
# ============================================================================
obstacles = []

# ============================================================================
# TRUE INITIAL CONDITION
# ============================================================================
# This will be set after Vh is created
true_initial_condition = None
Vh = None
prior = None
wind_velocity = None