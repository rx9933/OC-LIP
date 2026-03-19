"""Configuration parameters for OED training data generation."""

import numpy as np

# =============================================================================
# MESH AND FE SPACE PARAMETERS
# =============================================================================
NX, NY = 60, 60

# =============================================================================
# TIME PARAMETERS
# =============================================================================
T_INIT = 0.0
T_FINAL = 4.0
T_1 = 1.0
DT = 0.1
OBSERVATION_DT = 0.2

SIMULATION_TIMES = np.arange(T_INIT, T_FINAL + 0.5*DT, DT)
OBSERVATION_TIMES = np.arange(T_1, T_FINAL + 0.5*DT, OBSERVATION_DT)

# =============================================================================
# FOURIER PATH PARAMETERS
# =============================================================================
K = 3  # Number of Fourier modes
TY = (T_1, T_FINAL)  # Time window for frequencies
T_WINDOW = T_FINAL - T_1

# =============================================================================
# PRIOR PARAMETERS
# =============================================================================
GAMMA = 1.0
DELTA = 8.0

# =============================================================================
# OED PARAMETERS
# =============================================================================
R_MODES = 5  # Eigenvalues kept
NOISE_VARIANCE = 1e-4
REL_NOISE = 0.01

# =============================================================================
# PENALTY PARAMETERS
# =============================================================================
ZETA_BDY = 1000.0      # Boundary penalty weight
ZETA_SPEED = 500.0     # Speed penalty weight
V_MAX = 0.5            # Max sensor speed (domain units / time unit)

# =============================================================================
# OPTIMIZATION PARAMETERS
# =============================================================================
OPT_MAXITER = 80
OPT_FTOL = 1e-10
OPT_MAXLS = 40

# =============================================================================
# SPECTRAL WIND PARAMETERS
# =============================================================================
WIND_R = 3              # Number of spectral modes per direction
WIND_SIGMA = 1.0        # Overall amplitude scale
WIND_ALPHA = 2.0        # Spectral decay rate
WIND_MEAN_VX = 0.5      # Default mean horizontal wind
WIND_MEAN_VY = 0.0      # Default mean vertical wind

# =============================================================================
# TRAINING DATA PARAMETERS
# =============================================================================
N_SAMPLES = 1         # Number of wind samples
MEAN_VX_MEAN = 0.5       # Mean of mean_vx distribution
MEAN_VX_STD = 0.2        # Standard deviation of mean_vx distribution
DRONE_POS_STD = 0.15     # Standard deviation for drone position (clipped to [0.1, 0.9])
DRONE_POS_MEAN = 0.5     # Mean for drone position (center of domain)

# =============================================================================
# BOUNDS FOR OPTIMIZATION
# =============================================================================
LB = np.zeros(4*K + 2)
UB = np.zeros(4*K + 2)
LB[0] = 0.1;  UB[0] = 0.9   # x̄
LB[1] = 0.1;  UB[1] = 0.9   # ȳ
for kk in range(K):
    for j in range(4):
        LB[2 + 4*kk + j] = -0.3
        UB[2 + 4*kk + j] = 0.3
BOUNDS = list(zip(LB, UB))

# =============================================================================
# OUTPUT
# =============================================================================
OUTPUT_FILE = 'oed_training_data.pkl'