"""
Tuned RBNO training script for OED drone path prediction.

Improvements over train_rbno_oc.py:
  1. Deeper network (3 hidden layers: 128-128-64)
  2. Learning rate scheduler (ReduceLROnPlateau)
  3. Input/output normalization
  4. Early stopping
  5. PathNetwork IC enforcement (same as before)
  6. Building visualization on plots
"""

import os, sys
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.append('../../')
sys.path.append('generate_data/')
sys.path.append('plotting/')
sys.path.append('/workspace/arushi/hippylib')
sys.path.append('/workspace/arushi/hippyflow')
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'generate_data'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from generate_data.config import *
from fourier_utils import fourier_frequencies, generate_targets

from dinotorch_lite.src.dinotorch_lite import *
import dolfin as dl

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-dQ', '--dQ', type=int, default=22, help="input dim")
parser.add_argument('-dM', '--dM', type=int, default=14, help="output dim")
parser.add_argument('-n_pod', '--n_pod', type=int, default=20, help="number of POD modes to use")
parser.add_argument('-data_type', '--data_type', type=str, default='xv', help="xv or xvspectral")
parser.add_argument('-n_train', '--n_train', type=int, default=1600, help="Number of training data")
parser.add_argument('-n_test', '--n_test', type=int, default=200, help="Number of test data")
parser.add_argument('-plot_samples', '--plot_samples', type=int, default=6, help="Number of test samples to plot")
parser.add_argument('-data_dir', '--data_dir', type=str, default='data/', help="data directory")
parser.add_argument('-epochs', '--epochs', type=int, default=3000, help="max epochs")
parser.add_argument('-patience', '--patience', type=int, default=200, help="early stopping patience")
parser.add_argument('-lr', '--lr', type=float, default=1e-3, help="initial learning rate")
parser.add_argument('-hidden', '--hidden', type=str, default='128,128,64', help="hidden layer dims, comma separated")
args = parser.parse_args()

batch_size = 32
observation_times = np.linspace(T_1, T_FINAL, 50)
omegas = fourier_frequencies(TY, K)

# ================================================================
# BUILDINGS
# ================================================================
BUILDINGS = [
    {'lower': (0.26, 0.16), 'upper': (0.49, 0.39), 'margin': 0.03},
    {'lower': (0.61, 0.61), 'upper': (0.74, 0.84), 'margin': 0.03},
]

def draw_buildings(ax):
    for b in BUILDINGS:
        xmin, ymin = b['lower']
        xmax, ymax = b['upper']
        w = xmax - xmin
        h = ymax - ymin
        m = b['margin']
        rect = mpatches.Rectangle((xmin, ymin), w, h, color='gray', alpha=0.8, zorder=4)
        ax.add_patch(rect)
        rect_m = mpatches.Rectangle((xmin-m, ymin-m), w+2*m, h+2*m,
                                     color='gray', alpha=0.2, linestyle='--', fill=True, zorder=3)
        ax.add_patch(rect_m)


# ================================================================
# DEEP MLP
# ================================================================
class DeepMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[128, 128, 64]):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.GELU())
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ================================================================
# PATH NETWORK (IC ENFORCEMENT)
# ================================================================
class PathNetwork(nn.Module):
    def __init__(self, base_model, K=3, omegas=None, t0=1.0):
        super().__init__()
        self.base_model = base_model
        self.K = K
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

        return torch.cat([x_bar.unsqueeze(1), y_bar.unsqueeze(1), fourier_coeffs], dim=1)


# ================================================================
# NORMALIZER
# ================================================================
class Normalizer:
    def __init__(self, data, eps=1e-8):
        self.mean = data.mean(axis=0)
        self.std = data.std(axis=0) + eps

    def normalize(self, data):
        return (data - self.mean) / self.std

    def denormalize(self, data):
        return data * self.std + self.mean


# ================================================================
# LOAD DATA
# ================================================================
print("=" * 70)
print("  TUNED RBNO TRAINING")
print("=" * 70)

mq_data_dict = np.load(args.data_dir + 'mq_data_reduced.npz', allow_pickle=True)

m_data = mq_data_dict['m']  # [n_samples, 14]

n_pod = args.n_pod
if args.data_type == 'xv':
    v_pod = mq_data_dict['v'][:, :n_pod]
    q_data = np.concatenate((mq_data_dict['x'], v_pod), axis=1)
elif args.data_type == 'xvspectral':
    q_data = np.concatenate((mq_data_dict['x'], mq_data_dict['v_mean'], mq_data_dict['v_coeff']), axis=1)

# Only use the 12 Fourier amplitudes as targets (x_bar, y_bar computed by PathNetwork)
m_fourier = m_data[:, 2:]  # [n_samples, 12]

actual_dQ = q_data.shape[1]
print(f"  Data: {q_data.shape[0]} samples")
print(f"  Input dim: {actual_dQ} (2 position + {n_pod} POD)")
print(f"  Target dim: 12 (Fourier amplitudes, IC computed by PathNetwork)")

# Check POD explained variance
ev = mq_data_dict['pod_explained_variance']
print(f"  POD {n_pod} modes explains {ev[n_pod-1]*100:.2f}% of variance")

# Split data
n_total = q_data.shape[0]
n_train = min(args.n_train, n_total - args.n_test)
n_test = args.n_test
n_val = n_total - n_train - n_test

print(f"  Split: {n_train} train, {n_val} val, {n_test} test")

q_train_np = q_data[:n_train]
m_train_np = m_fourier[:n_train]

q_val_np = q_data[n_train:n_train+n_val]
m_val_np = m_fourier[n_train:n_train+n_val]

q_test_np = q_data[-n_test:]
m_test_np = m_fourier[-n_test:]
m_test_full_np = m_data[-n_test:]  # Full 14D for plotting

# Normalize inputs
q_norm = Normalizer(q_train_np)
q_train_normed = q_norm.normalize(q_train_np)
q_val_normed = q_norm.normalize(q_val_np)
q_test_normed = q_norm.normalize(q_test_np)

# Normalize targets (12 Fourier amplitudes)
m_norm = Normalizer(m_train_np)
m_train_normed = m_norm.normalize(m_train_np)
m_val_normed = m_norm.normalize(m_val_np)
m_test_normed = m_norm.normalize(m_test_np)

print(f"  Input stats: mean=[{q_norm.mean[:3]}...], std=[{q_norm.std[:3]}...]")
print(f"  Target stats: mean=[{m_norm.mean[:3]}...], std=[{m_norm.std[:3]}...]")

# Create dataloaders
train_ds = TensorDataset(torch.FloatTensor(q_train_normed), torch.FloatTensor(m_train_normed))
val_ds = TensorDataset(torch.FloatTensor(q_val_normed), torch.FloatTensor(m_val_normed))
test_ds = TensorDataset(torch.FloatTensor(q_test_normed), torch.FloatTensor(m_test_normed))

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"  Device: {device}")

# ================================================================
# MODEL
# ================================================================
hidden_dims = [int(x) for x in args.hidden.split(',')]
base_model = DeepMLP(input_dim=actual_dQ, output_dim=12, hidden_dims=hidden_dims).to(device)

n_params = sum(p.numel() for p in base_model.parameters())
print(f"\n  Network: {actual_dQ} -> {' -> '.join(map(str, hidden_dims))} -> 12")
print(f"  Parameters: {n_params:,}")
print(f"  Activation: GELU")
print("=" * 70)

# ================================================================
# TRAINING
# ================================================================
optimizer = torch.optim.Adam(base_model.parameters(), lr=args.lr, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=100, factor=0.5, min_lr=1e-6
)
loss_fn = nn.MSELoss()

best_val_loss = float('inf')
best_model_state = None
no_improve = 0

history = {
    'train_loss': [],
    'val_loss': [],
    'lr': [],
}

print(f"\nTraining for up to {args.epochs} epochs (patience={args.patience})...")
print(f"  LR: {args.lr}, scheduler: ReduceLROnPlateau(patience=100, factor=0.5)")

for epoch in range(args.epochs):
    # Train
    base_model.train()
    train_loss = 0.0
    for q_batch, m_batch in train_loader:
        q_batch, m_batch = q_batch.to(device), m_batch.to(device)
        m_pred = base_model(q_batch)
        loss = loss_fn(m_pred, m_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * q_batch.size(0)
    train_loss /= len(train_loader.dataset)

    # Validate
    base_model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for q_batch, m_batch in val_loader:
            q_batch, m_batch = q_batch.to(device), m_batch.to(device)
            m_pred = base_model(q_batch)
            loss = loss_fn(m_pred, m_batch)
            val_loss += loss.item() * q_batch.size(0)
    val_loss /= len(val_loader.dataset)

    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]['lr']

    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['lr'].append(current_lr)

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = {k: v.clone() for k, v in base_model.state_dict().items()}
        no_improve = 0
    else:
        no_improve += 1

    if epoch % 50 == 0:
        print(f"  Epoch {epoch:4d}  train={train_loss:.6e}  val={val_loss:.6e}  "
              f"lr={current_lr:.2e}  no_improve={no_improve}")
        sys.stdout.flush()

    if no_improve >= args.patience:
        print(f"\n  Early stopping at epoch {epoch} (patience={args.patience})")
        break

# Restore best model
if best_model_state is not None:
    base_model.load_state_dict(best_model_state)
    print(f"  Restored best model (val_loss={best_val_loss:.6e})")

# ================================================================
# WRAP WITH PATHNETWORK FOR EVALUATION
# ================================================================
def predict_full_path(q_raw_np):
    """Given raw (unnormalized) input, predict full 14D path coefficients."""
    q_normed = q_norm.normalize(q_raw_np)
    q_tensor = torch.FloatTensor(q_normed).to(device)
    if q_tensor.dim() == 1:
        q_tensor = q_tensor.unsqueeze(0)

    base_model.eval()
    with torch.no_grad():
        m_pred_normed = base_model(q_tensor).cpu().numpy()

    # Denormalize
    m_pred_fourier = m_norm.denormalize(m_pred_normed)

    # Add x_bar, y_bar from IC constraint
    c0 = q_raw_np[:2] if q_raw_np.ndim == 1 else q_raw_np[:, :2]
    results = []
    for i in range(m_pred_fourier.shape[0]):
        m_full = np.zeros(14)
        m_full[2:] = m_pred_fourier[i]
        # fix_c0_in_m
        shift_x = 0.0
        shift_y = 0.0
        t0 = T_1
        for k in range(K):
            cos_kt = np.cos(omegas[k] * t0)
            sin_kt = np.sin(omegas[k] * t0)
            shift_x += m_full[2 + 4*k] * cos_kt + m_full[3 + 4*k] * sin_kt
            shift_y += m_full[4 + 4*k] * cos_kt + m_full[5 + 4*k] * sin_kt
        c0_i = c0 if c0.ndim == 1 else c0[i]
        m_full[0] = c0_i[0] - shift_x
        m_full[1] = c0_i[1] - shift_y
        results.append(m_full)

    return np.array(results)


# ================================================================
# TEST EVALUATION
# ================================================================
print("\n" + "=" * 70)
print("  TEST EVALUATION")
print("=" * 70)

# Compute test loss on normalized data
base_model.eval()
test_loss = 0.0
with torch.no_grad():
    for q_batch, m_batch in test_loader:
        q_batch, m_batch = q_batch.to(device), m_batch.to(device)
        m_pred = base_model(q_batch)
        loss = loss_fn(m_pred, m_batch)
        test_loss += loss.item() * q_batch.size(0)
test_loss /= len(test_loader.dataset)
print(f"  Test MSE (normalized): {test_loss:.6e}")

# Compute relative error on raw data
total_rel_error = 0.0
total_samples = 0
for i in range(n_test):
    q_raw = q_test_np[i]
    m_pred_full = predict_full_path(q_raw)[0]
    m_true_full = m_test_full_np[i]
    rel_err = np.linalg.norm(m_pred_full - m_true_full) / (np.linalg.norm(m_true_full) + 1e-8)
    total_rel_error += rel_err
    total_samples += 1

avg_rel_error = total_rel_error / total_samples
print(f"  Test relative error (raw): {avg_rel_error:.4f}")

# IC verification
print("\n  IC Verification:")
for i in range(min(5, n_test)):
    q_raw = q_test_np[i]
    m_pred_full = predict_full_path(q_raw)[0]
    c0_input = q_raw[:2]
    pred_path = generate_targets(m_pred_full, np.array([observation_times[0]]), K, omegas)
    ic_error = np.linalg.norm(pred_path[0] - c0_input)
    print(f"    Sample {i}: c0=({c0_input[0]:.4f}, {c0_input[1]:.4f}), "
          f"start=({pred_path[0][0]:.4f}, {pred_path[0][1]:.4f}), IC err={ic_error:.2e}")

# Save model
model_save_name = f"tuned_rbno_{args.data_type}_npod{n_pod}_ntrain{n_train}.pth"
torch.save({
    'model_state': base_model.state_dict(),
    'q_norm_mean': q_norm.mean,
    'q_norm_std': q_norm.std,
    'm_norm_mean': m_norm.mean,
    'm_norm_std': m_norm.std,
    'hidden_dims': hidden_dims,
    'n_pod': n_pod,
    'args': vars(args),
}, os.path.join(args.data_dir, model_save_name))
print(f"\n  Model saved: {model_save_name}")

# Save errors
with open(os.path.join(args.data_dir, "test_errors.txt"), "a") as f:
    f.write(f"tuned_rbno, data_type={args.data_type}, n_pod={n_pod}, n_train={n_train}, "
            f"hidden={args.hidden}, test_mse={test_loss:.6e}, rel_error={avg_rel_error:.4f}\n")

# ================================================================
# PLOT TRAINING CURVES
# ================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

epochs_arr = range(1, len(history['train_loss']) + 1)
ax1.semilogy(epochs_arr, history['train_loss'], 'b-', label='Train', linewidth=1.5)
ax1.semilogy(epochs_arr, history['val_loss'], 'r-', label='Validation', linewidth=1.5)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('MSE Loss')
ax1.set_title(f'Training Curves (n_train={n_train})')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(epochs_arr, history['lr'], 'g-', linewidth=1.5)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Learning Rate')
ax2.set_title('Learning Rate Schedule')
ax2.set_yscale('log')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(args.data_dir, f'tuned_training_curves_ntrain{n_train}.png'), dpi=150)
plt.close()
print(f"  Training curves saved")

# ================================================================
# PLOT TEST EXAMPLES
# ================================================================
print("\n" + "=" * 70)
print("  PLOTTING TEST EXAMPLES")
print("=" * 70)

n_plot = min(args.plot_samples, n_test)
test_indices = np.linspace(0, n_test-1, n_plot, dtype=int)

fig = plt.figure(figsize=(20, 5 * n_plot))
plot_idx = 1

for plot_i, sample_idx in enumerate(test_indices):
    q_raw = q_test_np[sample_idx]
    m_true_full = m_test_full_np[sample_idx]
    m_pred_full = predict_full_path(q_raw)[0]

    x_init = q_raw[:2]

    true_path = generate_targets(m_true_full, observation_times, K, omegas)
    pred_path = generate_targets(m_pred_full, observation_times, K, omegas)

    # EIG values from data
    clean_idx = -n_test + sample_idx
    if clean_idx < 0:
        clean_idx = clean_idx % m_data.shape[0]
    eig_true = mq_data_dict['eig_K3'][clean_idx]
    eig_init = mq_data_dict['eig_K0'][clean_idx]

    # Compute EIG for NN path
    eig_pred = None
    try:
        from oed_objective import compute_eig_for_path
        mesh_for_eig = dl.UnitSquareMesh(NX, NY)
        Vh_for_eig = dl.FunctionSpace(mesh_for_eig, 'Lagrange', 1)
        eig_pred = compute_eig_for_path(m_pred_full, None, mesh_for_eig, Vh_for_eig)
    except:
        pass

    # ---- Subplot 1: Buildings + start position ----
    ax1 = plt.subplot(n_plot, 4, plot_idx); plot_idx += 1
    draw_buildings(ax1)
    ax1.plot(x_init[0], x_init[1], 'go', markersize=10, label='Start (c0)', zorder=10)
    ax1.set_xlim([0, 1]); ax1.set_ylim([0, 1])
    ax1.set_aspect('equal')
    ax1.set_title(f'Sample {sample_idx}: Start Position')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # ---- Subplot 2: Path comparison ----
    ax2 = plt.subplot(n_plot, 4, plot_idx); plot_idx += 1
    draw_buildings(ax2)
    ax2.plot(true_path[:, 0], true_path[:, 1], 'b-', linewidth=2, label='True (PDE)')
    ax2.plot(pred_path[:, 0], pred_path[:, 1], 'r--', linewidth=2, label='NN Prediction')
    ax2.plot(x_init[0], x_init[1], 'go', markersize=10, zorder=10)
    ax2.plot(true_path[0, 0], true_path[0, 1], 'b^', markersize=8, zorder=9)
    ax2.plot(pred_path[0, 0], pred_path[0, 1], 'r^', markersize=8, zorder=9)

    true_sensors = generate_targets(m_true_full, observation_times, K, omegas)
    pred_sensors = generate_targets(m_pred_full, observation_times, K, omegas)
    n_obs = len(observation_times)
    ax2.scatter(true_sensors[:, 0], true_sensors[:, 1], c=range(n_obs),
                cmap='Blues', s=15, alpha=0.6, edgecolors='blue', linewidths=0.3, zorder=6)
    ax2.scatter(pred_sensors[:, 0], pred_sensors[:, 1], c=range(n_obs),
                cmap='Reds', s=15, alpha=0.6, edgecolors='red', linewidths=0.3, zorder=6)

    ax2.set_xlim([0, 1]); ax2.set_ylim([0, 1])
    ax2.set_aspect('equal')
    ax2.set_title(f'Sample {sample_idx}: Path Comparison')
    ax2.legend(fontsize=7, loc='upper left')
    ax2.grid(True, alpha=0.3)

    # ---- Subplot 3: Path error ----
    ax3 = plt.subplot(n_plot, 4, plot_idx); plot_idx += 1
    path_error = np.sqrt(np.sum((true_path - pred_path)**2, axis=1))
    times = observation_times - T_1
    ax3.plot(times, path_error, 'r-', linewidth=2)
    ax3.fill_between(times, 0, path_error, alpha=0.3, color='red')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Position Error')
    rmse = np.mean(path_error)
    ax3.set_title(f'Path Error (RMSE: {rmse:.4f})')
    ax3.grid(True, alpha=0.3)

    # ---- Subplot 4: EIG comparison ----
    ax4 = plt.subplot(n_plot, 4, plot_idx); plot_idx += 1
    labels = ['Initial', 'PDE Optimal']
    values = [eig_init, eig_true]
    colors_bar = ['orange', 'blue']
    if eig_pred is not None:
        labels.append('NN Pred')
        values.append(eig_pred)
        colors_bar.append('red')
    bars = ax4.bar(labels, values, color=colors_bar, alpha=0.7)
    ax4.set_ylabel('EIG Value')
    ax4.set_title(f'EIG Comparison')
    ax4.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, values):
        ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{val:.2f}', ha='center', va='bottom', fontsize=8)

    ic_error = np.linalg.norm(pred_path[0] - x_init)
    print(f"  Sample {sample_idx}: EIG_true={eig_true:.2f}, RMSE={rmse:.4f}, IC_err={ic_error:.2e}")

plt.tight_layout()
plot_path = os.path.join(args.data_dir, f'tuned_test_samples_ntrain{n_train}.png')
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\n  Test samples plot saved: {plot_path}")

print("\n" + "=" * 70)
print("  DONE")
print("=" * 70)
