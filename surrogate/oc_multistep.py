import os, sys
import torch
from utils import *
from torch.optim.lr_scheduler import ReduceLROnPlateau

import numpy as np
from torch.utils.data import DataLoader
import matplotlib.patches as mpatches

sys.path.append('../../')
sys.path.append('generate_data/')
sys.path.append('plotting/')
sys.path.append('/workspace/arushi/hippylib')
sys.path.append('/workspace/arushi/hippyflow')
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'generate_data'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from generate_data.fe_setup import setup_fe_spaces
from generate_data.config import *
from fourier_utils import fourier_frequencies, generate_targets
from generate_data.wind_utils import spectral_wind_to_field

from dinotorch_lite.src.dinotorch_lite import *
from plotting.plot_trains import *; from generate_data.config import *
import dolfin as dl
from loss_utils import compute_path_from_m, path_mse_loss, combined_loss_with_components, check_path_violations

import numpy as np
def reconstruct_wind_from_dofs(mesh, wind_dof_vector):
    """Reconstruct wind velocity field from DOF vector."""
    # Create vector function space
    Xh = dl.VectorFunctionSpace(mesh, 'Lagrange', 2)
    wind_velocity = dl.Function(Xh)
    wind_velocity.vector().set_local(wind_dof_vector)
    wind_velocity.vector().apply("")
    # Allow extrapolation for visualization
    wind_velocity.set_allow_extrapolation(True)
    return wind_velocity


sys.path.append('../../')
sys.path.append('generate_data/')
sys.path.append('plotting/')
sys.path.append('/workspace/arushi/hippylib')
sys.path.append('/workspace/arushi/hippyflow')
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'generate_data'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from generate_data.fe_setup import setup_fe_spaces
from generate_data.config import *
from fourier_utils import fourier_frequencies, generate_targets


from dinotorch_lite.src.dinotorch_lite import *
from plotting.plot_trains import *; from generate_data.config import *
import dolfin as dl
from loss_utils import compute_path_from_m, path_mse_loss, combined_loss_with_components, check_path_violations

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-rQ', '--rQ', type=int, default=22, help="rQ")
parser.add_argument('-dQ', '--dQ', type=int, default=22, help="dQ")
parser.add_argument('-rM', '--rM', type=int, default=12, help="rM")
parser.add_argument('-dM', '--dM', type=int, default=14, help="dM")
parser.add_argument('-data_type', '--data_type', type=str, default='xv', help="xv or xvspectral")
parser.add_argument('-n_train', '--n_train', type=int, default=7500, help="Number of training data")
parser.add_argument('-n_test', '--n_test', type=int, default=200, help="Number of test data")
parser.add_argument('-n_data', '--n_data', type=int, default=7945, help="Max number of total data")
parser.add_argument('-plot_samples', '--plot_samples', type=int, default=20, help="Number of test samples to plot")
parser.add_argument('-data_dir', '--data_dir', type=str, default='data/', help="data directory")
parser.add_argument('-save_dir', '--save_dir', type=str, default='./models/flow/', help="Directory to save models")
parser.add_argument('-epochs', '--epochs', type=int, default=3000, help="epochs")
args = parser.parse_args()

batch_size = 32

assert args.n_train <= 8000 and args.n_train > 0

observation_times = np.linspace(T_1, T_FINAL, 200)
omegas = fourier_frequencies(TY, K)

# ================================================================
# BUILDINGS CONFIGURATION
# ================================================================
BUILDINGS = [
    {'lower': (0.26, 0.16), 'upper': (0.49, 0.39), 'margin': 0.03},
    {'lower': (0.61, 0.61), 'upper': (0.74, 0.84), 'margin': 0.03},
]

def create_building_features(buildings):
    """Flatten building coordinates into a feature vector."""
    features = []
    for b in buildings:
        xmin, ymin = b['lower']
        xmax, ymax = b['upper']
        margin = b['margin']
        features.extend([xmin, ymin, xmax, ymax, margin])
    return np.array(features)

def draw_buildings(ax):
    """Draw building rectangles and margins on a plot axis."""
    for b in BUILDINGS:
        xmin, ymin = b['lower']
        xmax, ymax = b['upper']
        w = xmax - xmin
        h = ymax - ymin
        m = b['margin']
        rect = mpatches.Rectangle((xmin, ymin), w, h,
                                   color='black', alpha=0.8, zorder=4)
        ax.add_patch(rect)
        rect_m = mpatches.Rectangle((xmin-m, ymin-m), w+2*m, h+2*m,
                                     color='black', alpha=0.2, linestyle='--',
                                     fill=True, zorder=3)
        ax.add_patch(rect_m)

# ================================================================
# IC ENFORCEMENT BY CONSTRUCTION
# ================================================================
class PathNetwork(torch.nn.Module):
    """
    Wraps a base NN to enforce the initial condition exactly.
    """
    def __init__(self, base_model, K=3, omegas=None, t0=1.0):
        super().__init__()
        self.base_model = base_model
        self.K = K
        self.omegas = omegas
        self.t0 = t0

        self.cos_vals = [float(np.cos(omegas[k] * t0)) for k in range(K)]
        self.sin_vals = [float(np.sin(omegas[k] * t0)) for k in range(K)]

    def forward(self, q, expected=None):
        c0 = q[:, :2]
        fourier_coeffs = self.base_model(q, expected=expected[:, 2:] if expected is not None else None)
        
        shift_x = torch.zeros(q.shape[0], device=q.device)
        shift_y = torch.zeros(q.shape[0], device=q.device)

        for k in range(self.K):
            shift_x = shift_x + fourier_coeffs[:, 4*k] * self.cos_vals[k] + fourier_coeffs[:, 4*k+1] * self.sin_vals[k]
            shift_y = shift_y + fourier_coeffs[:, 4*k+2] * self.cos_vals[k] + fourier_coeffs[:, 4*k+3] * self.sin_vals[k]

        x_bar = c0[:, 0] - shift_x
        y_bar = c0[:, 1] - shift_y

        m_full = torch.cat([x_bar.unsqueeze(1), y_bar.unsqueeze(1), fourier_coeffs], dim=1)
        return m_full

# ================================================================
# MULTISTEP DENOISING FUNCTION
# ================================================================
def multistep_denoise_rk4(model, q, nsteps=20):
    """
    Multistep denoising using RK4 integration in latent space.
    More accurate than Euler for the probability flow ODE.
    """
    model.eval()
    
    with torch.no_grad():
        batch_size = q.size(0)
        device = q.device
        
        # Initialize Fourier coefficients with noise
        m_fourier = torch.randn(batch_size, 12, device=device)
        
        # Time steps from 0 to 1
        ts = np.linspace(0, 1, nsteps + 1)
        dt = 1.0 / nsteps
        
        def velocity(m, t):
            """Compute the velocity field dm/dt at time t"""
            m_clean = model.base_model(q, expected=m)
            # Probability flow ODE: dm/dt = (m_clean - m) / (1 - t)
            # Avoid division by zero at t=1
            denom = max(1 - t, 1e-8)
            return (m_clean - m) / denom
        
        for i in range(nsteps):
            t = ts[i]
            
            # RK4 steps
            k1 = velocity(m_fourier, t)
            k2 = velocity(m_fourier + dt/2 * k1, t + dt/2)
            k3 = velocity(m_fourier + dt/2 * k2, t + dt/2)
            k4 = velocity(m_fourier + dt * k3, t + dt)
            
            # Update using RK4
            m_fourier = m_fourier + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        
        # Reconstruct full m (IC enforcement)
        c0 = q[:, :2]
        
        shift_x = torch.zeros(batch_size, device=device)
        shift_y = torch.zeros(batch_size, device=device)
        
        for k in range(K):
            shift_x += (m_fourier[:, 4*k] * np.cos(omegas[k]*T_1) +
                       m_fourier[:, 4*k+1] * np.sin(omegas[k]*T_1))
            shift_y += (m_fourier[:, 4*k+2] * np.cos(omegas[k]*T_1) +
                       m_fourier[:, 4*k+3] * np.sin(omegas[k]*T_1))
        
        x_bar = c0[:, 0] - shift_x
        y_bar = c0[:, 1] - shift_y
        
        m_full = torch.cat([x_bar.unsqueeze(1), y_bar.unsqueeze(1), m_fourier], dim=1)
    
    return m_full

def multistep_denoise(model, q, nsteps=20, alpha=0.2):
    """
    Multistep denoising for better predictions.
    Uses iterative refinement with noise injection.
    """
    model.eval()
    
    with torch.no_grad():
        batch_size = q.size(0)
        device = q.device
        
        # Initialize Fourier coefficients with noise
        m_fourier = torch.randn(batch_size, 12, device=device)
        
        
        ts = np.linspace(0, 1, nsteps+1)[1:]  # Time steps for noise injection
        dt = 1/nsteps
        for t in ts:
            m_clean = model.base_model(q, expected=m_fourier)
            v = (m_clean-m_fourier)/1-t
            m_fourier = m_fourier + dt * v  # Move towards cleaner output

        
        '''

        for i in range(nsteps):
            m_clean = model.base_model(q, expected=m_fourier)
            # Relaxation/interpolation step
            m_fourier = (1 - alpha) * m_fourier + alpha * m_clean
        '''

        # Reconstruct full m (IC enforcement)
        c0 = q[:, :2]
        
        shift_x = torch.zeros(batch_size, device=device)
        shift_y = torch.zeros(batch_size, device=device)
        
        for k in range(K):
            shift_x += (m_fourier[:, 4*k] * np.cos(omegas[k]*T_1) +
                       m_fourier[:, 4*k+1] * np.sin(omegas[k]*T_1))
            shift_y += (m_fourier[:, 4*k+2] * np.cos(omegas[k]*T_1) +
                       m_fourier[:, 4*k+3] * np.sin(omegas[k]*T_1))
        
        x_bar = c0[:, 0] - shift_x
        y_bar = c0[:, 1] - shift_y
        
        m_full = torch.cat([x_bar.unsqueeze(1), y_bar.unsqueeze(1), m_fourier], dim=1)
    
    return m_full

# ================================================================
# LOAD DATA
# ================================================================
mq_data_dict = np.load(args.data_dir + 'mq_data_relabeled_pass2.npz', allow_pickle=True)

m_data = mq_data_dict['m']
# q is POD reduced velocity (20 modes ) + 2 initial position of drone = 22 total

if args.data_type == 'xv':
    v_pod = mq_data_dict['v'][:, :20]
    q_data = np.concatenate((mq_data_dict['x'], v_pod), axis=1)
elif args.data_type == 'xvspectral':
    q_data = np.concatenate((mq_data_dict['x'], mq_data_dict['v_mean'], mq_data_dict['v_coeff']), axis=1)

# Add building features to every sample
building_features = create_building_features(BUILDINGS)
building_features_repeated = np.tile(building_features, (q_data.shape[0], 1))
q_data = np.concatenate([q_data, building_features_repeated], axis=1)

print(f"Data shapes: q={q_data.shape}, m={m_data.shape}")
print(f"Input dim (dQ): {q_data.shape[1]}")
print(f"Output dim (dM): {m_data.shape[1]}")
print(f"First 2 cols of q are drone position (c0)")

m_train = torch.Tensor(m_data[:args.n_train])
q_train = torch.Tensor(q_data[:args.n_train])

m_val = torch.Tensor(m_data[args.n_train:-args.n_test])
q_val = torch.Tensor(q_data[args.n_train:-args.n_test])

m_test = torch.Tensor(m_data[-args.n_test:])
q_test = torch.Tensor(q_data[-args.n_test:])

l2invtrain = L2Dataset(q_train, m_train)
l2invval = L2Dataset(q_val, m_val)
l2invtest = L2Dataset(q_test, m_test)

train_invloader = DataLoader(l2invtrain, batch_size=batch_size, shuffle=True)
validation_invloader = DataLoader(l2invval, batch_size=batch_size, shuffle=False)
test_invloader = DataLoader(l2invtest, batch_size=batch_size, shuffle=False)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

################################################################################
# MODEL
################################################################################
args.dQ = q_data.shape[1] 
base_model = FlowGenericDenseSkipLearn(input_dim=args.dQ, hidden_layer_dim=4*args.dQ, output_dim=12).to(device)
model = PathNetwork(base_model, K=K, omegas=omegas, t0=T_1).to(device)

# Load trained model
model_save_name = f"rbno_datatype_{args.data_type}_rQ{args.dQ}_rM{args.dM}_ntrain{args.n_train}.pth"
model_path = os.path.join(args.data_dir, model_save_name)

if os.path.exists(model_path):
    print(f"\nLoading pre-trained model from {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Model loaded successfully!")
else:
    print(f"\nModel not found at {model_path}. Please train the model first.")
    sys.exit(1)

# ================================================================
# PLOT TEST EXAMPLES WITH EIG COMPARISON
# ================================================================
print("\n" + "="*60)
print("PLOTTING TEST EXAMPLES WITH EIG COMPARISON")
print("="*60)

# Use the same mesh as in data generation
MESH_FILE = 'generate_data/ad_20.xml'
try:
    mesh = dl.refine(dl.Mesh(MESH_FILE))
    print(f"Loaded mesh from {MESH_FILE} with {mesh.num_cells()} cells")
except:
    print(f"Warning: Could not load {MESH_FILE}, using UnitSquareMesh({NX}, {NY})")
    mesh = dl.UnitSquareMesh(NX, NY)

Vh_scalar = dl.FunctionSpace(mesh, 'Lagrange', 1)
V_vec_deg2 = dl.VectorFunctionSpace(mesh, 'Lagrange', 2)

n_plot = min(args.plot_samples, len(m_test))
test_indices = np.arange(1, n_plot)

print(f"Plotting {n_plot} test examples...")

fig = plt.figure(figsize=(25, 5*n_plot))
plot_idx = 1

def plot_wind_field_on_grid(ax, wind_velocity, resolution=30):
    """Plot wind field on a regular grid."""
    x = np.linspace(0, 1, resolution)
    y = np.linspace(0, 1, resolution)
    X, Y = np.meshgrid(x, y)
    
    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    
    for i in range(resolution):
        for j in range(resolution):
            point = np.array([X[i,j], Y[i,j]])
            try:
                val = wind_velocity(point)
                U[i,j] = val[0]
                V[i,j] = val[1]
            except:
                closest_point = np.clip(point, 0.001, 0.999)
                try:
                    val = wind_velocity(closest_point)
                    U[i,j] = val[0]
                    V[i,j] = val[1]
                except:
                    U[i,j] = 0
                    V[i,j] = 0
    
    speed = np.sqrt(U**2 + V**2)
    quiver = ax.quiver(X, Y, U, V, speed, 
                       cmap='coolwarm', alpha=0.7, 
                       scale=15, width=0.003)
    return quiver

# Statistics for EIG values
eig_pde_list = []
eig_direct_list = []
eig_multistep_list = []
margin_violations_direct = 0
margin_violations_multistep = 0
building_violations_direct = 0
building_violations_multistep = 0
direct_rmse_list = []
multistep_rmse_list = []

for sample_idx in test_indices:
    m_true = m_test[sample_idx].cpu().numpy()
    q_input = q_test[sample_idx].cpu().numpy()

    clean_idx = -args.n_test + sample_idx
    if clean_idx < 0:
        clean_idx = clean_idx % len(m_data)

    if args.data_type == 'xvspectral':
        x_init = q_input[:2]
        v_mean = q_input[2:4]
        v_coeff = q_input[4:]

        r_wind = 3
        n_coeff_per_mode = r_wind * r_wind
        a_ij = v_coeff[:n_coeff_per_mode].reshape(r_wind, r_wind)
        b_ij = v_coeff[n_coeff_per_mode:].reshape(r_wind, r_wind)

        wind_coeffs = {
            'a_ij': a_ij, 'b_ij': b_ij,
            'mean_vx': v_mean[0], 'mean_vy': v_mean[1],
            'r_wind': r_wind, 'sigma': 1.0, 'alpha': 2.0
        }

        wind_field, _ = spectral_wind_to_field(mesh, wind_coeffs)
        eig_true = mq_data_dict['eig_opt'][clean_idx]
        eig_init = mq_data_dict['eig_init'][clean_idx]

    else:  # args.data_type == 'xv'
        x_init = q_input[:2]
        wind_coeffs = None
        
        # Get stored wind DOFs
        wind_dofs = mq_data_dict['wind_dofs'][clean_idx]
        
        # Reconstruct wind field
        try:
            
            wind_field = reconstruct_wind_from_dofs(mesh, wind_dofs)
            print(f"  Successfully reconstructed wind field (DOFs: {len(wind_dofs)})")
        except Exception as e:
            print(f"  Warning: Could not reconstruct wind field: {e}")
            wind_field = None
        
        eig_true = mq_data_dict['eig_K3'][clean_idx]
        eig_init = mq_data_dict['eig_K0'][clean_idx]

    # Predict with NN - Direct (one-step)
    model.eval()
    with torch.no_grad():
        q_input_tensor = torch.FloatTensor(q_input).unsqueeze(0).to(device)
        m_pred_direct = model(q_input_tensor).cpu().numpy().flatten()
    
    # Predict with NN - Multistep denoising
    with torch.no_grad():
        # m_pred_multistep = multistep_denoise(model, q_input_tensor, nsteps=3, alpha=0.2).cpu().numpy().flatten()
        m_pred_multistep = multistep_denoise_rk4(model, q_input_tensor, nsteps=200).cpu().numpy().flatten()

    # Generate paths
    true_path = generate_targets(m_true, observation_times, K, omegas)
    direct_path = generate_targets(m_pred_direct, observation_times, K, omegas)
    multistep_path = generate_targets(m_pred_multistep, observation_times, K, omegas)
    
    # Calculate RMSE
    direct_rmse = np.sqrt(np.mean(np.sum((true_path - direct_path)**2, axis=1)))
    multistep_rmse = np.sqrt(np.mean(np.sum((true_path - multistep_path)**2, axis=1)))
    direct_rmse_list.append(direct_rmse)
    multistep_rmse_list.append(multistep_rmse)

    # Compute EIG for both predictions
    eig_direct = None
    eig_multistep = None
    
    try:
        from oed_objective import compute_eig_for_path
        
        # EIG for direct prediction
        eig_direct = compute_eig_for_path(m_pred_direct, wind_coeffs, mesh, Vh_scalar)
        if eig_direct is not None:
            eig_direct_list.append(eig_direct)
            
            # Check violations for direct path
            if check_path_violations(direct_path, BUILDINGS, margin_violation=True):
                margin_violations_direct += 1
            if check_path_violations(direct_path, BUILDINGS, margin_violation=False):
                building_violations_direct += 1
        
        # EIG for multistep prediction
        eig_multistep = compute_eig_for_path(m_pred_multistep, wind_coeffs, mesh, Vh_scalar)
        if eig_multistep is not None:
            eig_multistep_list.append(eig_multistep)
            
            # Check violations for multistep path
            if check_path_violations(multistep_path, BUILDINGS, margin_violation=True):
                margin_violations_multistep += 1
            if check_path_violations(multistep_path, BUILDINGS, margin_violation=False):
                building_violations_multistep += 1
        
        eig_pde_list.append(eig_true)
        
        # Format output strings safely
        eig_direct_str = f"{eig_direct:.4f}" if eig_direct is not None else "N/A"
        eig_multistep_str = f"{eig_multistep:.4f}" if eig_multistep is not None else "N/A"
        
        print(f"\n  Sample {sample_idx}:")
        print(f"    PDE Optimal EIG: {eig_true:.4f}")
        print(f"    Direct NN EIG: {eig_direct_str}")
        print(f"    Multistep NN EIG: {eig_multistep_str}")
        print(f"    Direct RMSE: {direct_rmse:.4f}, Multistep RMSE: {multistep_rmse:.4f}")
        
    except Exception as e:
        print(f"  WARNING: compute_eig_for_path failed for sample {sample_idx}: {e}")
        eig_pde_list.append(eig_true)

    # ---- Subplot 1: Wind field ----
    ax1 = plt.subplot(n_plot, 5, plot_idx)
    plot_idx += 1
    
    # Draw buildings
    for b in BUILDINGS:
        xmin, ymin = b['lower']
        xmax, ymax = b['upper']
        w = xmax - xmin
        h = ymax - ymin
        m = b['margin']
        rect = mpatches.Rectangle((xmin, ymin), w, h,
                                   color='black', alpha=0.8, zorder=4)
        ax1.add_patch(rect)
        rect_m = mpatches.Rectangle((xmin-m, ymin-m), w+2*m, h+2*m,
                                     color='black', alpha=0.2, linestyle='--',
                                     fill=True, zorder=3)
        ax1.add_patch(rect_m)
    
    # Plot wind field
    if wind_field is not None:
        try:
            quiver = plot_wind_field_on_grid(ax1, wind_field, resolution=25)
            plt.colorbar(quiver, ax=ax1, fraction=0.046, pad=0.04, label='Wind speed')
        except Exception as e:
            print(f"  Warning: Failed to plot wind field: {e}")
            ax1.text(0.5, 0.5, f'Wind field plot error:\n{str(e)[:50]}',
                    ha='center', va='center', transform=ax1.transAxes, fontsize=8)
    else:
        ax1.text(0.5, 0.5, 'Wind field unavailable',
                ha='center', va='center', transform=ax1.transAxes, fontsize=10)

    # Mark start position
    ax1.plot(x_init[0], x_init[1], 'go', markersize=8, label='Start', zorder=10)
    
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    ax1.set_aspect('equal')
    ax1.set_title(f'Sample {sample_idx}: Wind Field')
    ax1.grid(True, alpha=0.3)

    # ---- Subplot 2: Direct Prediction vs True ----
    ax2 = plt.subplot(n_plot, 5, plot_idx)
    plot_idx += 1
    draw_buildings(ax2)
    ax2.plot(true_path[:, 0], true_path[:, 1], 'b-', linewidth=2, label='True (PDE)')
    ax2.plot(direct_path[:, 0], direct_path[:, 1], 'r--', linewidth=2, label='Direct NN')
    ax2.plot(x_init[0], x_init[1], 'go', markersize=10, label='Start', zorder=10)
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])
    ax2.set_aspect('equal')
    ax2.set_title(f'Direct NN (RMSE: {direct_rmse:.4f})')
    ax2.legend(fontsize=7, loc='upper left')
    ax2.grid(True, alpha=0.3)

    # ---- Subplot 3: Multistep Denoising vs True ----
    ax3 = plt.subplot(n_plot, 5, plot_idx)
    plot_idx += 1
    draw_buildings(ax3)
    ax3.plot(true_path[:, 0], true_path[:, 1], 'b-', linewidth=2, label='True (PDE)')
    ax3.plot(multistep_path[:, 0], multistep_path[:, 1], 'g--', linewidth=2, label='Multistep NN')
    ax3.plot(x_init[0], x_init[1], 'go', markersize=10, label='Start', zorder=10)
    ax3.set_xlim([0, 1])
    ax3.set_ylim([0, 1])
    ax3.set_aspect('equal')
    ax3.set_title(f'Multistep Denoise (RMSE: {multistep_rmse:.4f})')
    ax3.legend(fontsize=7, loc='upper left')
    ax3.grid(True, alpha=0.3)

    # ---- Subplot 4: Path Error Comparison ----
    ax4 = plt.subplot(n_plot, 5, plot_idx)
    plot_idx += 1
    path_error_direct = np.sqrt(np.sum((true_path - direct_path)**2, axis=1))
    path_error_multistep = np.sqrt(np.sum((true_path - multistep_path)**2, axis=1))
    times = observation_times - T_1
    ax4.plot(times, path_error_direct, 'r-', linewidth=2, label='Direct NN')
    ax4.plot(times, path_error_multistep, 'g-', linewidth=2, label='Multistep')
    ax4.fill_between(times, 0, path_error_direct, alpha=0.2, color='red')
    ax4.fill_between(times, 0, path_error_multistep, alpha=0.2, color='green')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Position Error')
    ax4.set_title('Path Error Comparison')
    ax4.legend(fontsize=7)
    ax4.grid(True, alpha=0.3)

    # ---- Subplot 5: EIG Comparison (PDE vs Direct vs Multistep) ----
    ax5 = plt.subplot(n_plot, 5, plot_idx)
    plot_idx += 1

    # Prepare data for bar chart - only include valid values
    labels = []
    values = []
    colors_bar = []
    
    # Add PDE EIG if available
    if eig_true is not None:
        labels.append('PDE\nOptimal')
        values.append(eig_true)
        colors_bar.append('blue')
    
    # Add Direct NN EIG if available
    if eig_direct is not None:
        labels.append('Direct\nNN')
        values.append(eig_direct)
        colors_bar.append('red')
    
    # Add Multistep NN EIG if available
    if eig_multistep is not None:
        labels.append('Multistep\nNN')
        values.append(eig_multistep)
        colors_bar.append('green')
    
    if len(values) > 0:
        bars = ax5.bar(labels, values, color=colors_bar, alpha=0.7)
        ax5.set_ylabel('EIG Value')
        ax5.set_title(f'Sample {sample_idx}: EIG Comparison')
        ax5.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on top of bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Add percentage difference if both PDE and NN EIG are available
        if eig_true is not None and eig_direct is not None:
            pct_diff_direct = ((eig_direct - eig_true) / eig_true) * 100
            ax5.text(0.5, -0.15, f'Direct Δ: {pct_diff_direct:+.1f}%', 
                    transform=ax5.transAxes, ha='center', fontsize=8, color='red')
        
        if eig_true is not None and eig_multistep is not None:
            pct_diff_multistep = ((eig_multistep - eig_true) / eig_true) * 100
            ax5.text(0.5, -0.25, f'Multistep Δ: {pct_diff_multistep:+.1f}%', 
                    transform=ax5.transAxes, ha='center', fontsize=8, color='green')
    else:
        ax5.text(0.5, 0.5, 'EIG values unavailable', ha='center', va='center')
        ax5.set_title(f'Sample {sample_idx}: EIG Comparison')
        ax5.set_xlim([0, 1])
        ax5.set_ylim([0, 1])
    
    ic_error = np.linalg.norm(direct_path[0] - x_init)
    print(f"    IC error={ic_error:.2e}")

# Calculate statistics - filter out None values
eig_pde_list_filtered = [e for e in eig_pde_list if e is not None]
eig_direct_list_filtered = [e for e in eig_direct_list if e is not None]
eig_multistep_list_filtered = [e for e in eig_multistep_list if e is not None]

mean_eig_pde = np.mean(eig_pde_list_filtered) if eig_pde_list_filtered else 0
mean_eig_direct = np.mean(eig_direct_list_filtered) if eig_direct_list_filtered else 0
mean_eig_multistep = np.mean(eig_multistep_list_filtered) if eig_multistep_list_filtered else 0
mean_direct_rmse = np.mean(direct_rmse_list) if direct_rmse_list else 0
mean_multistep_rmse = np.mean(multistep_rmse_list) if multistep_rmse_list else 0
improvement = (mean_direct_rmse - mean_multistep_rmse) / mean_direct_rmse * 100 if mean_direct_rmse > 0 else 0

print("\n" + "="*60)
print("EIG COMPARISON SUMMARY")
print("="*60)
print(f"Number of samples with valid EIG:")
print(f"  PDE: {len(eig_pde_list_filtered)}/{len(test_indices)}")
print(f"  Direct NN: {len(eig_direct_list_filtered)}/{len(test_indices)}")
print(f"  Multistep NN: {len(eig_multistep_list_filtered)}/{len(test_indices)}")
print(f"\nAverage EIG Values:")
print(f"  PDE Optimal: {mean_eig_pde:.6f}")
if mean_eig_direct > 0:
    print(f"  Direct NN: {mean_eig_direct:.6f}")
    print(f"  Direct vs PDE difference: {((mean_eig_direct - mean_eig_pde)/mean_eig_pde*100):.2f}%")
else:
    print(f"  Direct NN: N/A")
if mean_eig_multistep > 0:
    print(f"  Multistep NN: {mean_eig_multistep:.6f}")
    print(f"  Multistep vs PDE difference: {((mean_eig_multistep - mean_eig_pde)/mean_eig_pde*100):.2f}%")
    if mean_eig_direct > 0:
        print(f"  Multistep improvement over Direct: {((mean_eig_direct - mean_eig_multistep)/mean_eig_direct*100):.2f}%")
else:
    print(f"  Multistep NN: N/A")

print(f"\nPath RMSE Comparison:")
print(f"Average Direct RMSE: {mean_direct_rmse:.6f}")
print(f"Average Multistep RMSE: {mean_multistep_rmse:.6f}")
print(f"Improvement: {improvement:.2f}%")

print(f"\nPath Violations:")
print(f"Direct - Margin: {margin_violations_direct}/{len(test_indices)} ({100*margin_violations_direct/len(test_indices):.1f}%), Building: {building_violations_direct}/{len(test_indices)} ({100*building_violations_direct/len(test_indices):.1f}%)")
print(f"Multistep - Margin: {margin_violations_multistep}/{len(test_indices)} ({100*margin_violations_multistep/len(test_indices):.1f}%), Building: {building_violations_multistep}/{len(test_indices)} ({100*building_violations_multistep/len(test_indices):.1f}%)")

# Add to suptitle
fig.suptitle(f'Multi-Step Denoising vs Direct Prediction\n'
             f'PDE EIG: {mean_eig_pde:.4f} | Direct EIG: {mean_eig_direct:.4f} | MultiStep EIG: {mean_eig_multistep:.4f}\n'
             f'Direct RMSE: {mean_direct_rmse:.4f} | MultiStep RMSE: {mean_multistep_rmse:.4f} | Improvement: {improvement:.1f}%\n'
             f'Margin Violations - Direct: {margin_violations_direct}/{len(test_indices)} | MultiStep: {margin_violations_multistep}/{len(test_indices)}\n'
             f'Building Violations - Direct: {building_violations_direct}/{len(test_indices)} | MultiStep: {building_violations_multistep}/{len(test_indices)}', 
             fontsize=10, fontweight='bold', y = 1.02)

plt.tight_layout()

plot_save_name = f"rk4_eig_comparison_{args.data_type}_ntrain{args.n_train}.png"
plot_save_path = os.path.join(args.data_dir, plot_save_name)
plt.savefig(plot_save_path, dpi=150, bbox_inches='tight')
print(f"\nTest samples plot saved to {plot_save_path}")
plt.close()

# Save EIG comparison to file
eig_comparison_file = os.path.join(args.data_dir, f"eig_comparison_{args.data_type}_ntrain{args.n_train}.txt")
with open(eig_comparison_file, 'w') as f:
    f.write("EIG COMPARISON RESULTS\n")
    f.write("="*60 + "\n")
    f.write(f"Data type: {args.data_type}\n")
    f.write(f"Number of training samples: {args.n_train}\n")
    f.write(f"Number of test samples: {len(test_indices)}\n\n")
    f.write("Sample Counts:\n")
    f.write(f"  PDE EIG valid: {len(eig_pde_list_filtered)}/{len(test_indices)}\n")
    f.write(f"  Direct NN EIG valid: {len(eig_direct_list_filtered)}/{len(test_indices)}\n")
    f.write(f"  Multistep NN EIG valid: {len(eig_multistep_list_filtered)}/{len(test_indices)}\n\n")
    f.write("Average EIG Values:\n")
    f.write(f"  PDE Optimal: {mean_eig_pde:.8f}\n")
    if mean_eig_direct > 0:
        f.write(f"  Direct NN: {mean_eig_direct:.8f}\n")
    else:
        f.write(f"  Direct NN: N/A\n")
    if mean_eig_multistep > 0:
        f.write(f"  Multistep NN: {mean_eig_multistep:.8f}\n")
    else:
        f.write(f"  Multistep NN: N/A\n\n")
    if mean_eig_direct > 0 and mean_eig_pde > 0:
        f.write("EIG Differences:\n")
        f.write(f"  Direct vs PDE: {(mean_eig_direct - mean_eig_pde)/mean_eig_pde*100:.2f}%\n")
    if mean_eig_multistep > 0 and mean_eig_pde > 0:
        f.write(f"  Multistep vs PDE: {(mean_eig_multistep - mean_eig_pde)/mean_eig_pde*100:.2f}%\n")
    if mean_eig_multistep > 0 and mean_eig_direct > 0:
        f.write(f"  Multistep vs Direct: {(mean_eig_multistep - mean_eig_direct)/mean_eig_direct*100:.2f}%\n\n")
    f.write("RMSE Values:\n")
    f.write(f"  Direct RMSE: {mean_direct_rmse:.8f}\n")
    f.write(f"  Multistep RMSE: {mean_multistep_rmse:.8f}\n")
    f.write(f"  Improvement: {improvement:.2f}%\n\n")
    f.write("Path Violations:\n")
    f.write(f"  Direct - Margin: {margin_violations_direct}/{len(test_indices)} ({100*margin_violations_direct/len(test_indices):.1f}%)\n")
    f.write(f"  Direct - Building: {building_violations_direct}/{len(test_indices)} ({100*building_violations_direct/len(test_indices):.1f}%)\n")
    f.write(f"  Multistep - Margin: {margin_violations_multistep}/{len(test_indices)} ({100*margin_violations_multistep/len(test_indices):.1f}%)\n")
    f.write(f"  Multistep - Building: {building_violations_multistep}/{len(test_indices)} ({100*building_violations_multistep/len(test_indices):.1f}%)\n")

print(f"\nEIG comparison saved to {eig_comparison_file}")
print("\n" + "="*60)
print("PLOTTING COMPLETE")
print("="*60)