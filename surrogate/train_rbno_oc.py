import os, sys
import torch
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

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-rQ', '--rQ', type=int, default=22, help="rQ")
parser.add_argument('-dQ', '--dQ', type=int, default=22, help="dQ")
parser.add_argument('-rM', '--rM', type=int, default=12, help="rM")
parser.add_argument('-dM', '--dM', type=int, default=14, help="dM")
parser.add_argument('-data_type', '--data_type', type=str, default='xv', help="xv or xvspectral")
parser.add_argument('-n_train', '--n_train', type=int, default=800, help="Number of training data")
parser.add_argument('-n_test', '--n_test', type=int, default=100, help="Number of test data")
parser.add_argument('-n_data', '--n_data', type=int, default=945, help="Max number of total data")
parser.add_argument('-plot_samples', '--plot_samples', type=int, default=4, help="Number of test samples to plot")
parser.add_argument('-data_dir', '--data_dir', type=str, default='data/', help="data directory")
parser.add_argument('-save_dir', '--save_dir', type=str, default='./models/rbno/', help="Directory to save models")
parser.add_argument('-epochs', '--epochs', type=int, default=1000, help="epochs")
args = parser.parse_args()

batch_size = 32

assert args.n_train <= 8000 and args.n_train > 0

observation_times = np.linspace(T_1, T_FINAL, 50)
omegas = fourier_frequencies(TY, K)

# ================================================================
# BUILDINGS CONFIGURATION
# ================================================================
BUILDINGS = [
    {'lower': (0.26, 0.16), 'upper': (0.49, 0.39), 'margin': 0.03},
    {'lower': (0.61, 0.61), 'upper': (0.74, 0.84), 'margin': 0.03},
]

def draw_buildings(ax):
    """Draw building rectangles and margins on a plot axis."""
    for b in BUILDINGS:
        xmin, ymin = b['lower']
        xmax, ymax = b['upper']
        w = xmax - xmin
        h = ymax - ymin
        m = b['margin']
        rect = mpatches.Rectangle((xmin, ymin), w, h,
                                   color='gray', alpha=0.8, zorder=4)
        ax.add_patch(rect)
        rect_m = mpatches.Rectangle((xmin-m, ymin-m), w+2*m, h+2*m,
                                     color='gray', alpha=0.2, linestyle='--',
                                     fill=True, zorder=3)
        ax.add_patch(rect_m)


# ================================================================
# IC ENFORCEMENT BY CONSTRUCTION
# ================================================================
class PathNetwork(torch.nn.Module):
    """
    Wraps a base NN to enforce the initial condition exactly.

    The base model outputs 12 Fourier amplitudes (theta_k, phi_k, psi_k, eta_k for k=1,2,3).
    This wrapper computes x_bar and y_bar deterministically from c0 and the predicted
    Fourier coefficients, so the path always starts at c0 exactly.

    No IC penalty needed in the loss function.
    """
    def __init__(self, base_model, K=3, omegas=None, t0=1.0):
        super().__init__()
        self.base_model = base_model
        self.K = K
        self.omegas = omegas
        self.t0 = t0

        self.cos_vals = [float(np.cos(omegas[k] * t0)) for k in range(K)]
        self.sin_vals = [float(np.sin(omegas[k] * t0)) for k in range(K)]

    def forward(self, q):
        c0 = q[:, :2]
        fourier_coeffs = self.base_model(q)

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
# LOAD DATA
# ================================================================
mq_data_dict = np.load(args.data_dir + 'mq_data_reduced.npz', allow_pickle=True)

m_data = mq_data_dict['m']

if args.data_type == 'xv':
    v_pod = mq_data_dict['v'][:, :20]
    q_data = np.concatenate((mq_data_dict['x'], v_pod), axis=1)
elif args.data_type == 'xvspectral':
    q_data = np.concatenate((mq_data_dict['x'], mq_data_dict['v_mean'], mq_data_dict['v_coeff']), axis=1)

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
validation_invloader = DataLoader(l2invval, batch_size=batch_size, shuffle=True)
test_invloader = DataLoader(l2invtest, batch_size=batch_size, shuffle=True)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

################################################################################
# MODEL
################################################################################
base_model = GenericDense(input_dim=args.dQ, hidden_layer_dim=2*args.dQ, output_dim=12).to(device)
model = PathNetwork(base_model, K=K, omegas=omegas, t0=T_1).to(device)

print(f"\nModel architecture:")
print(f"  Base model: {args.dQ}D -> {2*args.dQ}D -> 12D (Fourier amplitudes)")
print(f"  PathNetwork: adds x_bar, y_bar from c0 -> 14D output")
print(f"  IC enforced by construction (no penalty needed)")

n_epochs = args.epochs
loss_func = normalized_f_mse
lr_scheduler = None

optimizer = torch.optim.Adam(model.parameters())

network, history = l2_training(model, loss_func, train_invloader, validation_invloader,
                     optimizer, lr_scheduler=lr_scheduler, n_epochs=n_epochs, verbose=True, ic_penalty=False)

rel_error_test = evaluate_l2_error(model, test_invloader)
print('L2 relative error = ', rel_error_test)

model_save_name = f"rbno_datatype_{args.data_type}_rQ{args.dQ}_rM{args.dM}_ntrain{args.n_train}.pth"
torch.save(model.state_dict(), os.path.join(args.data_dir, model_save_name))

plot_training_history('RBNO', history, args.n_train, args.data_dir, args)

# Calculate MSE loss
model.eval()
mse_loss_fn = torch.nn.MSELoss()
mse_total = 0.0
n_batches = 0

with torch.no_grad():
    for q_batch, m_batch in test_invloader:
        q_batch = q_batch.to(device)
        m_batch = m_batch.to(device)

        m_pred = model(q_batch)
        batch_mse = mse_loss_fn(m_pred, m_batch)
        mse_total += batch_mse.item() * q_batch.size(0)
        n_batches += q_batch.size(0)

mse = mse_total / n_batches
print(f'MSE loss = {mse:.6f}')

# ================================================================
# IC VERIFICATION
# ================================================================
print("\n" + "="*60)
print("IC ENFORCEMENT VERIFICATION")
print("="*60)
model.eval()
with torch.no_grad():
    for i in range(min(5, len(q_test))):
        q_input = q_test[i].unsqueeze(0).to(device)
        m_pred = model(q_input).cpu().numpy().flatten()
        c0_input = q_test[i][:2].numpy()

        pred_path = generate_targets(m_pred, np.array([observation_times[0]]), K, omegas)
        pred_start = pred_path[0]

        ic_error = np.linalg.norm(pred_start - c0_input)
        print(f"  Sample {i}: c0=({c0_input[0]:.4f}, {c0_input[1]:.4f}), "
              f"path_start=({pred_start[0]:.4f}, {pred_start[1]:.4f}), "
              f"IC error={ic_error:.2e}")

with open(os.path.join(args.data_dir, "test_errors.txt"), "a") as f:
    f.write(f"data_type={args.data_type}, n_train={args.n_train}, rel_error={rel_error_test}, mse={mse}\n")

# ================================================================
# PLOT TEST EXAMPLES
# ================================================================
print("\n" + "="*60)
print("PLOTTING TEST EXAMPLES")
print("="*60)

mesh = dl.UnitSquareMesh(NX, NY)
Vh_scalar = dl.FunctionSpace(mesh, 'Lagrange', 1)
V_vec = dl.VectorFunctionSpace(mesh, 'Lagrange', 1)

n_plot = min(args.plot_samples, len(m_test))
test_indices = np.arange(1, n_plot)

print(f"Plotting {n_plot} test examples...")

fig = plt.figure(figsize=(20, 5*n_plot))
plot_idx = 1

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

    else:
        x_init = q_input[:2]
        v_coeff = q_input[2:]
        wind_coeffs = None
        wind_field = None
        eig_true = mq_data_dict['eig_K3'][clean_idx]
        eig_init = mq_data_dict['eig_K0'][clean_idx]

    # Predict with NN
    model.eval()
    with torch.no_grad():
        q_input_tensor = torch.FloatTensor(q_input).unsqueeze(0).to(device)
        m_pred = model(q_input_tensor).cpu().numpy().flatten()

    # Generate paths
    true_path = generate_targets(m_true, observation_times, K, omegas)
    pred_path = generate_targets(m_pred, observation_times, K, omegas)

    # Compute EIG for NN path
    eig_pred = None
    eig_true_recomputed = eig_true
    try:
        from oed_objective import compute_eig_for_path
        eig_pred = compute_eig_for_path(m_pred, wind_coeffs, mesh, Vh_scalar)
        if eig_pred is not None:
            from penalties import boundary_penalty_dense, speed_penalty_dense
            bdy_val, _ = boundary_penalty_dense(m_pred, observation_times, K, omegas)
            spd_val, _ = speed_penalty_dense(m_pred, observation_times, K, omegas)
            print(f"  NN penalties: boundary={bdy_val:.2f}, speed={spd_val:.2f}")
            print(f"  NN raw EIG={eig_pred:.4f}, NN penalized EIG={eig_pred - bdy_val - spd_val:.4f}")
            eig_true_recomputed = compute_eig_for_path(m_true, wind_coeffs, mesh, Vh_scalar)
            print(f"  Stored PDE EIG={eig_true:.4f}, Recomputed PDE EIG={eig_true_recomputed:.4f}, NN EIG={eig_pred:.4f}")
    except Exception as e:
        print(f"  WARNING: compute_eig_for_path failed: {e}")

    # ---- Subplot 1: Wind field ----
    ax1 = plt.subplot(n_plot, 4, plot_idx)
    plot_idx += 1
    draw_buildings(ax1)

    if wind_field is not None:
        vx_func, vy_func = wind_field.split(deepcopy=True)
        coords = mesh.coordinates()
        vx_vals = vx_func.vector().get_local()
        vy_vals = vy_func.vector().get_local()

        nx, ny = 40, 40
        xi = np.linspace(0, 1, nx)
        yi = np.linspace(0, 1, ny)
        X, Y = np.meshgrid(xi, yi)

        from scipy.interpolate import griddata
        Vx_grid = griddata((coords[:, 0], coords[:, 1]), vx_vals, (X, Y), method='linear')
        Vy_grid = griddata((coords[:, 0], coords[:, 1]), vy_vals, (X, Y), method='linear')

        speed = np.sqrt(Vx_grid**2 + Vy_grid**2)
        strm = ax1.streamplot(X, Y, Vx_grid, Vy_grid, color=speed,
                             cmap='viridis', linewidth=1, density=1.2)
        plt.colorbar(strm.lines, ax=ax1, label='Speed')
    else:
        ax1.text(0.5, 0.5, 'Wind field (POD input)',
                ha='center', va='center', transform=ax1.transAxes, fontsize=10)

    ax1.plot(x_init[0], x_init[1], 'go', markersize=8, label='Start')
    ax1.set_xlim([0, 1]); ax1.set_ylim([0, 1])
    ax1.set_aspect('equal')
    ax1.set_title(f'Sample {sample_idx}: Wind Field')
    ax1.grid(True, alpha=0.3)

    # ---- Subplot 2: True vs Predicted Path ----
    ax2 = plt.subplot(n_plot, 4, plot_idx)
    plot_idx += 1
    draw_buildings(ax2)
    ax2.plot(true_path[:, 0], true_path[:, 1], 'b-', linewidth=2, label='True (PDE)')
    ax2.plot(pred_path[:, 0], pred_path[:, 1], 'r--', linewidth=2, label='NN Prediction')
    ax2.plot(x_init[0], x_init[1], 'go', markersize=10, label='Start (c0)', zorder=10)
    ax2.plot(true_path[0, 0], true_path[0, 1], 'b^', markersize=8, label='True Start', zorder=9)
    ax2.plot(pred_path[0, 0], pred_path[0, 1], 'r^', markersize=8, label='NN Start', zorder=9)

    # Add time-colored sensor dots on both paths
    n_obs = len(observation_times)
    true_sensors = generate_targets(m_true, observation_times, K, omegas)
    pred_sensors = generate_targets(m_pred, observation_times, K, omegas)
    ax2.scatter(true_sensors[:, 0], true_sensors[:, 1], c=range(n_obs),
                cmap='Blues', s=15, alpha=0.6, edgecolors='blue', linewidths=0.3, zorder=6)
    ax2.scatter(pred_sensors[:, 0], pred_sensors[:, 1], c=range(n_obs),
                cmap='Reds', s=15, alpha=0.6, edgecolors='red', linewidths=0.3, zorder=6)

    ax2.set_xlim([0, 1]); ax2.set_ylim([0, 1])
    ax2.set_aspect('equal')
    ax2.set_title(f'Sample {sample_idx}: Path Comparison')
    ax2.legend(fontsize=7, loc='upper left')
    ax2.grid(True, alpha=0.3)

    # ---- Subplot 3: Path Error ----
    ax3 = plt.subplot(n_plot, 4, plot_idx)
    plot_idx += 1
    path_error = np.sqrt(np.sum((true_path - pred_path)**2, axis=1))
    times = observation_times - T_1
    ax3.plot(times, path_error, 'r-', linewidth=2)
    ax3.fill_between(times, 0, path_error, alpha=0.3, color='red')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Position Error')
    ax3.set_title(f'Sample {sample_idx}: Path Error (RMSE: {np.mean(path_error):.4f})')
    ax3.grid(True, alpha=0.3)

    # ---- Subplot 4: EIG Comparison ----
    ax4 = plt.subplot(n_plot, 4, plot_idx)
    plot_idx += 1

    labels = ['Initial', 'PDE Optimal']
    values = [eig_init, eig_true_recomputed]
    colors_bar = ['orange', 'blue']

    if eig_pred is not None:
        labels.append('NN Prediction')
        values.append(eig_pred)
        colors_bar.append('red')

    bars = ax4.bar(labels, values, color=colors_bar, alpha=0.7)
    ax4.set_ylabel('EIG Value')
    ax4.set_title(f'Sample {sample_idx}: EIG Comparison')
    ax4.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}', ha='center', va='bottom', fontsize=8)

    ic_error = np.linalg.norm(pred_path[0] - x_init)
    print(f"  Sample {sample_idx}: True EIG={eig_true:.4f}, IC error={ic_error:.2e}, Path RMSE={np.mean(path_error):.4f}")

plt.tight_layout()

plot_save_name = f"test_samples_{args.data_type}_ntrain{args.n_train}.png"
plot_save_path = os.path.join(args.data_dir, plot_save_name)
plt.savefig(plot_save_path, dpi=150, bbox_inches='tight')
print(f"\nTest samples plot saved to {plot_save_path}")
plt.close()

print("\n" + "="*60)
print("PLOTTING COMPLETE")
print("="*60)
