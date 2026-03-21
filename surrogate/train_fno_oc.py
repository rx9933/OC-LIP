import os, sys
import torch
import numpy as np
from torch.utils.data import DataLoader
from fno import *

sys.path.append('../../')
sys.path.append('generate_data/')
sys.path.append('plotting/')
sys.path.append('workspace/arushi/hippylib')
sys.path.append('/workspace/arushi/hippyflow')

from dinotorch_lite.src.dinotorch_lite import * 
from plotting.plot_trains import *

import argparse
import matplotlib.pyplot as plt
import dolfin as dl

parser = argparse.ArgumentParser()
parser.add_argument('-n_train', '--n_train', type=int, default=800, help="Number of training data")
parser.add_argument('-n_test', '--n_test', type=int, default=100, help="Number of test data")
parser.add_argument('-n_data', '--n_data', type=int, default=800, help="Max number of total data")
parser.add_argument('-data_dir', '--data_dir', type=str, default='data/', help="data directory")
parser.add_argument('-epochs', '--epochs', type=int, default=1000, help="epochs")
parser.add_argument('-plot_samples', '--plot_samples', type=int, default=4, help="Number of test samples to plot")
parser.add_argument('-r_wind', '--r_wind', type=int, default=3, help="Number of wind modes in each dimension (from config)")
parser.add_argument('-width', '--width', type=int, default=64, help="FNO channel width")
parser.add_argument('-lr', '--lr', type=float, default=1e-3, help="Learning rate")
parser.add_argument('-save_dir', '--save_dir', type=str, default='./models/fno/', help="Directory to save models")
args = parser.parse_args()

batch_size = 32
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

os.makedirs(args.save_dir, exist_ok=True)

data_dir = args.data_dir
mq_data_dict = np.load(data_dir + 'mq_data_reduced.npz')

# m_data: path coefficients (output) - shape: (n_samples, 4*K + 2)
m_data = mq_data_dict['m']
print(f"m_data shape: {m_data.shape}")

# Extract all three input components
x_data = mq_data_dict['x']           # drone positions - shape: (n_samples, 2)
v_mean_data = mq_data_dict['v_mean'] # shape: (n_samples, 2) - v_x, v_y
v_coeff_data = mq_data_dict['v_coeff'] # shape: (n_samples, 18) for r_wind=3

print(f"x_data shape: {x_data.shape} (drone positions)")
print(f"v_mean_data shape: {v_mean_data.shape} (mean flow)")
print(f"v_coeff_data shape: {v_coeff_data.shape} (wind spectral coefficients)")

# Get r_wind from data
r_wind = int(np.sqrt(v_coeff_data.shape[1] // 2))  # Since total coefficients = 2 * r_wind^2
print(f"Detected r_wind = {r_wind} from v_coeff_data")

# Parse v_coeff_data into a_coeff and b_coeff
if len(v_coeff_data.shape) == 2:
    # Flattened format: (n_samples, 2 * r_wind^2)
    print(f"Reshaping flattened coefficients (r_wind={r_wind})...")
    n_coeff_per_mode = r_wind * r_wind
    a_coeff = v_coeff_data[:, :n_coeff_per_mode].reshape(-1, r_wind, r_wind)
    b_coeff = v_coeff_data[:, n_coeff_per_mode:2*n_coeff_per_mode].reshape(-1, r_wind, r_wind)
elif len(v_coeff_data.shape) == 4 and v_coeff_data.shape[1] == 2:
    # Already in grid format (batch, 2, r, r)
    a_coeff = v_coeff_data[:, 0]  # first channel
    b_coeff = v_coeff_data[:, 1]  # second channel
else:
    raise ValueError(f"Unexpected v_coeff_data shape: {v_coeff_data.shape}")

print(f"a_coeff shape: {a_coeff.shape}")
print(f"b_coeff shape: {b_coeff.shape}")

# Convert to tensors
m_data = torch.Tensor(m_data)
x_data = torch.Tensor(x_data)
v_mean_data = torch.Tensor(v_mean_data)
a_coeff = torch.Tensor(a_coeff)
b_coeff = torch.Tensor(b_coeff)

# Split into train/test
n_samples = len(m_data)
n_train = min(args.n_train, n_samples - args.n_test)
n_test = min(args.n_test, n_samples - n_train)

print(f"Using {n_train} training samples and {n_test} test samples")

# Split data: train, validation, test (like RBNO script)
m_train = m_data[:n_train]
m_val = m_data[n_train:-n_test] if n_test > 0 else m_data[n_train:]
m_test = m_data[-n_test:] if n_test > 0 else m_data[n_train:]

x_train = x_data[:n_train]
x_val = x_data[n_train:-n_test] if n_test > 0 else x_data[n_train:]
x_test = x_data[-n_test:] if n_test > 0 else x_data[n_train:]

v_mean_train = v_mean_data[:n_train]
v_mean_val = v_mean_data[n_train:-n_test] if n_test > 0 else v_mean_data[n_train:]
v_mean_test = v_mean_data[-n_test:] if n_test > 0 else v_mean_data[n_train:]

a_coeff_train = a_coeff[:n_train]
a_coeff_val = a_coeff[n_train:-n_test] if n_test > 0 else a_coeff[n_train:]
a_coeff_test = a_coeff[-n_test:] if n_test > 0 else a_coeff[n_train:]

b_coeff_train = b_coeff[:n_train]
b_coeff_val = b_coeff[n_train:-n_test] if n_test > 0 else b_coeff[n_train:]
b_coeff_test = b_coeff[-n_test:] if n_test > 0 else b_coeff[n_train:]

print(f"\nTrain/Val/Test split: {len(m_train)}/{len(m_val)}/{len(m_test)}")

# Create dataset
class WindToPathDataset(torch.utils.data.Dataset):
    def __init__(self, x_positions, v_mean, a_coeff, b_coeff, path_coeffs):
        self.x_positions = x_positions
        self.v_mean = v_mean
        self.a_coeff = a_coeff
        self.b_coeff = b_coeff
        self.path_coeffs = path_coeffs
        
    def __len__(self):
        return len(self.path_coeffs)
    
    def __getitem__(self, idx):
        return {
            'x_positions': self.x_positions[idx],
            'v_mean': self.v_mean[idx],
            'a_coeff': self.a_coeff[idx],
            'b_coeff': self.b_coeff[idx],
            'path_coeffs': self.path_coeffs[idx]
        }

train_dataset = WindToPathDataset(x_train, v_mean_train, a_coeff_train, b_coeff_train, m_train)
val_dataset = WindToPathDataset(x_val, v_mean_val, a_coeff_val, b_coeff_val, m_val)
test_dataset = WindToPathDataset(x_test, v_mean_test, a_coeff_test, b_coeff_test, m_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ----------------------------
# Initialize the enhanced FNO model
# ----------------------------
print("\n" + "="*60)
print("Initializing FullInputSpectralFNO model")
print("="*60)

# Determine K from m_data shape
K = (m_data.shape[1] - 2) // 4
print(f"Detected K = {K} from m_data shape")

# Import the model (make sure fno.py has the FullInputSpectralFNO class)
try:
    from fno import FullInputSpectralFNO
    model = FullInputSpectralFNO(
        r=args.r_wind,  # Use r_wind, not r_modes
        K=K,
        width=args.width,
        modes1=min(8, args.r_wind),
        modes2=min(8, args.r_wind),
        n_layers=4
    ).to(device)
except ImportError:
    # If FullInputSpectralFNO doesn't exist, use a simpler version
    print("Warning: FullInputSpectralFNO not found, using SimpleFNO")
    
    class SimpleFNO(torch.nn.Module):
        def __init__(self, r_wind, K, width=64, modes=8):
            super().__init__()
            self.r_wind = r_wind
            self.K = K
            
            # Input: x(2) + v_mean(2) + a_coeff(r,r) + b_coeff(r,r) = 2 + 2 + 2*r*r
            input_dim = 4 + 2 * r_wind * r_wind
            self.fc1 = torch.nn.Linear(input_dim, width)
            self.fc2 = torch.nn.Linear(width, width)
            self.fc3 = torch.nn.Linear(width, 4*K+2)
            
        def forward(self, x_positions, v_mean, a_coeff, b_coeff):
            # Flatten inputs
            batch_size = x_positions.shape[0]
            x_flat = x_positions.view(batch_size, -1)
            v_mean_flat = v_mean.view(batch_size, -1)
            a_flat = a_coeff.view(batch_size, -1)
            b_flat = b_coeff.view(batch_size, -1)
            
            # Concatenate
            combined = torch.cat([x_flat, v_mean_flat, a_flat, b_flat], dim=1)
            
            # Forward pass
            x = torch.relu(self.fc1(combined))
            x = torch.relu(self.fc2(x))
            output = self.fc3(x)
            
            return output
    
    model = SimpleFNO(
        r_wind=args.r_wind,
        K=K,
        width=args.width
    ).to(device)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# ----------------------------
# Training setup
# ----------------------------
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=20, min_lr=1e-6
)

# Loss function - MSE for path coefficients
criterion = torch.nn.MSELoss()

# Training history
history = {
    'train_loss': [],
    'val_loss': [],
    'lr': []
}

# ----------------------------
# Training loop
# ----------------------------
print("\n" + "="*60)
print("Starting FNO training with all three inputs:")
print("  - Drone positions (x_data)")
print("  - Mean flow (v_mean_data)")
print(f"  - Wind spectral coefficients (a_coeff, b_coeff) with r_wind={args.r_wind}")
print("="*60)

best_val_loss = float('inf')
best_model_path = os.path.join(args.save_dir, f'fno_model_r{args.r_wind}_ntrain{args.n_train}.pth')

for epoch in range(args.epochs):
    # Training
    model.train()
    train_loss = 0
    for batch in train_loader:
        x_positions = batch['x_positions'].to(device)
        v_mean = batch['v_mean'].to(device)
        a_coeff = batch['a_coeff'].to(device)
        b_coeff = batch['b_coeff'].to(device)
        path_coeffs = batch['path_coeffs'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass with all three inputs
        pred_path = model(x_positions, v_mean, a_coeff, b_coeff)
        
        # Compute loss
        loss = criterion(pred_path, path_coeffs)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * len(x_positions)
    
    train_loss /= len(train_dataset)
    
    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            x_positions = batch['x_positions'].to(device)
            v_mean = batch['v_mean'].to(device)
            a_coeff = batch['a_coeff'].to(device)
            b_coeff = batch['b_coeff'].to(device)
            path_coeffs = batch['path_coeffs'].to(device)
            
            pred_path = model(x_positions, v_mean, a_coeff, b_coeff)
            loss = criterion(pred_path, path_coeffs)
            val_loss += loss.item() * len(x_positions)
    
    val_loss /= len(val_dataset)
    
    # Update learning rate
    current_lr = optimizer.param_groups[0]['lr']
    scheduler.step(val_loss)
    new_lr = optimizer.param_groups[0]['lr']
    
    # Save history
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)

    history['lr'].append(current_lr)
    
    # Print progress
    if epoch % 50 == 0 or epoch == args.epochs - 1:
        print(f"Epoch {epoch:4d}: train_loss = {train_loss:.6e}, val_loss = {val_loss:.6e}, lr = {current_lr:.2e}")
    
    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, best_model_path)
        if epoch % 10 == 0:
            print(f"  → New best model saved (val_loss = {val_loss:.6e})")

# ----------------------------
# Plot training history
# ----------------------------
plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
plt.semilogy(history['train_loss'], label='Train')
plt.semilogy(history['val_loss'], label='Validation')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('FNO Training History (All Inputs)')
plt.legend()
plt.grid(True)

plt.subplot(1,2,2)
plt.plot(history['lr'])
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Schedule')
plt.grid(True)

plt.tight_layout()
history_path = os.path.join(args.save_dir, f'fno_training_history_ntrain{args.n_train}.png')
plt.savefig(history_path, dpi=200)
plt.close()
print(f"\nSaved training history plot to: {history_path}")

# ----------------------------
# Load best model and evaluate
# ----------------------------
print("\n" + "="*60)
print("Evaluating best model on test set")
print("="*60)

checkpoint = torch.load(best_model_path)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Compute detailed metrics on test set
test_predictions = []
test_targets = []

with torch.no_grad():
    for batch in test_loader:
        x_positions = batch['x_positions'].to(device)
        v_mean = batch['v_mean'].to(device)
        a_coeff = batch['a_coeff'].to(device)
        b_coeff = batch['b_coeff'].to(device)
        path_coeffs = batch['path_coeffs'].to(device)
        
        pred_path = model(x_positions, v_mean, a_coeff, b_coeff)
        
        test_predictions.append(pred_path.cpu().numpy())
        test_targets.append(path_coeffs.cpu().numpy())

test_predictions = np.concatenate(test_predictions, axis=0)
test_targets = np.concatenate(test_targets, axis=0)

# Compute relative L2 error
relative_errors = np.linalg.norm(test_predictions - test_targets, axis=1) / (np.linalg.norm(test_targets, axis=1) + 1e-8)
mse =  np.mean((test_predictions - test_targets)**2)
mean_rel_error = np.mean(relative_errors)
std_rel_error = np.std(relative_errors)

print(f"Test set performance:")
print(f"  Best validation loss: {best_val_loss:.6e}")
print(f"  Mean relative L2 error: {mean_rel_error:.6e}")
print(f"  Std relative L2 error: {std_rel_error:.6e}")

# Save test error to file
with open(os.path.join(args.save_dir, "test_errors.txt"), "a") as f: 
    f.write(f"model=FNO, r_wind={args.r_wind}, n_train={args.n_train}, rel_error={mean_rel_error:.6e}\n")

# ----------------------------
# Save final model and results
# ----------------------------
final_model_path = os.path.join(args.save_dir, 'final_fno_model.pth')
torch.save({
    'model_state_dict': model.state_dict(),
    'history': history,
    'args': args,
    'test_metrics': {
        'best_val_loss': best_val_loss,
        'mean_rel_error': mean_rel_error,
        'std_rel_error': std_rel_error
    }
}, final_model_path)

# Save predictions for analysis
np.savez(os.path.join(args.save_dir, 'test_predictions.npz'),
         predictions=test_predictions,
         targets=test_targets,
         relative_errors=relative_errors)

print(f"\nSaved final model to: {final_model_path}")
print(f"Saved test predictions to: {os.path.join(args.save_dir, 'test_predictions.npz')}")

# ============================================================================
# PLOT TEST EXAMPLES (like RBNO script)
# ============================================================================
print("\n" + "="*60)
print("PLOTTING TEST EXAMPLES")
print("="*60)

# Setup finite element spaces for plotting
from generate_data.config import *
from fourier_utils import fourier_frequencies, generate_targets
from generate_data.wind_utils import spectral_wind_to_field

# Setup mesh and function spaces
mesh = dl.UnitSquareMesh(NX, NY)
Vh_scalar = dl.FunctionSpace(mesh, 'Lagrange', 1)  # For plotting
V_vec = dl.VectorFunctionSpace(mesh, 'Lagrange', 1)  # For wind fields

# Compute frequencies for path generation
omegas = fourier_frequencies(TY, K)

# Get a few test samples
n_plot = min(args.plot_samples, len(m_test))
test_indices = np.arange(min(args.plot_samples, len(m_test)))

print(f"Plotting {n_plot} test examples...")

# Create a figure with multiple subplots for all test cases
fig = plt.figure(figsize=(20, 5*n_plot))
plot_idx = 1

for sample_idx in test_indices:
    # Get test data
    x_positions = x_test[sample_idx].cpu().numpy()
    v_mean = v_mean_test[sample_idx].cpu().numpy()
    a_coeff_val = a_coeff_test[sample_idx].cpu().numpy()
    b_coeff_val = b_coeff_test[sample_idx].cpu().numpy()
    m_true = m_test[sample_idx].cpu().numpy()
    
    # Get corresponding clean data from original dict
    clean_idx = -n_test + sample_idx if n_test > 0 else sample_idx
    if clean_idx < 0:
        clean_idx = clean_idx % len(m_data)
    
    # Reconstruct wind coefficients dictionary
    wind_coeffs = {
        'a_ij': a_coeff_val,
        'b_ij': b_coeff_val,
        'mean_vx': v_mean[0],
        'mean_vy': v_mean[1],
        'r_wind': args.r_wind,
        'sigma': 1.0,
        'alpha': 2.0
    }
    
    # Reconstruct wind field
    wind_field, _ = spectral_wind_to_field(mesh, wind_coeffs)
    
    # Get true EIG values
    eig_true = mq_data_dict['eig_opt'][clean_idx]
    eig_init = mq_data_dict['eig_init'][clean_idx]
    
    # Predict with FNO model
    model.eval()
    with torch.no_grad():
        x_positions_tensor = torch.FloatTensor(x_positions).unsqueeze(0).to(device)
        v_mean_tensor = torch.FloatTensor(v_mean).unsqueeze(0).to(device)
        a_coeff_tensor = torch.FloatTensor(a_coeff_val).unsqueeze(0).to(device)
        b_coeff_tensor = torch.FloatTensor(b_coeff_val).unsqueeze(0).to(device)
        
        m_pred = model(x_positions_tensor, v_mean_tensor, a_coeff_tensor, b_coeff_tensor).cpu().numpy().flatten()
    
    # Generate paths
    observation_times = np.linspace(T_1, T_FINAL, 50)  # Smooth path
    true_path = generate_targets(m_true, observation_times, K, omegas)
    pred_path = generate_targets(m_pred, observation_times, K, omegas)
    
    # Compute path error metrics
    path_error = np.sqrt(np.sum((true_path - pred_path)**2, axis=1))
    rmse_path = np.mean(path_error)
    
    # Subplot 1: Wind field with streamlines
    ax1 = plt.subplot(n_plot, 4, plot_idx)
    plot_idx += 1
    
    # Extract velocity components for plotting
    vx_func, vy_func = wind_field.split(deepcopy=True)
    coords = mesh.coordinates()
    vx_vals = vx_func.vector().get_local()
    vy_vals = vy_func.vector().get_local()
    
    # Create grid for plotting
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
    plt.colorbar(strm.lines, ax=ax1, label='Speed (m/s)')
    ax1.plot(x_positions[0], x_positions[1], 'go', markersize=8, label='Start')
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    ax1.set_aspect('equal')
    ax1.set_title(f'Sample {sample_idx}: Wind Field (vx={v_mean[0]:.2f}, vy={v_mean[1]:.2f})')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Subplot 2: True vs Predicted Path
    ax2 = plt.subplot(n_plot, 4, plot_idx)
    plot_idx += 1
    ax2.plot(true_path[:, 0], true_path[:, 1], 'b-', linewidth=2, label='True (PDE)')
    ax2.plot(pred_path[:, 0], pred_path[:, 1], 'r--', linewidth=2, label='FNO Prediction')
    ax2.plot(x_positions[0], x_positions[1], 'go', markersize=8, label='Start')
    ax2.plot(true_path[-1, 0], true_path[-1, 1], 'bs', markersize=8, label='True End')
    ax2.plot(pred_path[-1, 0], pred_path[-1, 1], 'rs', markersize=8, label='FNO End')
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])
    ax2.set_aspect('equal')
    ax2.set_title(f'Sample {sample_idx}: Path Comparison')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # Subplot 3: Path Error over time
    ax3 = plt.subplot(n_plot, 4, plot_idx)
    plot_idx += 1
    times = observation_times - T_1
    ax3.plot(times, path_error, 'r-', linewidth=2)
    ax3.fill_between(times, 0, path_error, alpha=0.3, color='red')
    ax3.axhline(y=rmse_path, color='k', linestyle='--', alpha=0.5, label=f'RMSE: {rmse_path:.4f}')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Position Error')
    ax3.set_title(f'Sample {sample_idx}: Path Error (RMSE: {rmse_path:.4f})')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # Subplot 4: EIG Comparison
    ax4 = plt.subplot(n_plot, 4, plot_idx)
    plot_idx += 1
    
    labels = ['Initial', 'PDE Optimal']
    values = [eig_init, eig_true]
    colors_bar = ['orange', 'blue']
    
    bars = ax4.bar(labels, values, color=colors_bar, alpha=0.7)
    ax4.set_ylabel('EIG Value')
    ax4.set_title(f'Sample {sample_idx}: EIG Comparison')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}', ha='center', va='bottom', fontsize=8)
    
    print(f"  Sample {sample_idx}: True EIG={eig_true:.4f}, Initial EIG={eig_init:.4f}, Path RMSE={rmse_path:.4f}")

plt.tight_layout()

# Save the figure
plot_save_name = f"fno_test_samples_r{args.r_wind}_ntrain{args.n_train}.png"
plot_save_path = os.path.join(args.save_dir, plot_save_name)
plt.savefig(plot_save_path, dpi=150, bbox_inches='tight')
print(f"\nTest samples plot saved to {plot_save_path}")
plt.close()


# Update the file writing to include MSE
with open(os.path.join(args.data_dir, "test_errors.txt"), "a") as f: 
    f.write(f"data_type=fno, n_train={args.n_train}, rel_error={mean_rel_error}, mse={mse}\n")
# ----------------------------
# Summary
# ----------------------------
print("\n" + "="*60)
print("Training Complete!")
print("="*60)
print(f"\nInputs used:")
print(f"  - Drone positions: {x_data.shape}")
print(f"  - Mean flow: {v_mean_data.shape}")
print(f"  - Wind coefficients: {v_coeff_data.shape}")
print(f"  - r_wind: {args.r_wind} (detected from data)")
print(f"\nResults saved to: {args.save_dir}")
print(f"  - Best model: {best_model_path}")
print(f"  - Final model: {final_model_path}")
print(f"  - Training history: {history_path}")
print(f"  - Test predictions: {os.path.join(args.save_dir, 'test_predictions.npz')}")
print(f"  - Test samples plot: {plot_save_path}")
print("="*60)