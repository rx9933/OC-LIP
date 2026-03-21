import os, sys
import torch
import numpy as np
from torch.utils.data import DataLoader
from fno import *

sys.path.append('../../')
sys.path.append('/workspace/arushi/hippylib')
sys.path.append('/workspace/arushi/hippyflow')

from dinotorch_lite.src.dinotorch_lite import * 
from plotting.plot_trains import *

import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('-n_train', '--n_train', type=int, default=800, help="Number of training data")
parser.add_argument('-n_test', '--n_test', type=int, default=200, help="Number of test data")
parser.add_argument('-n_data', '--n_data', type=int, default=800, help="Max number of total data")
parser.add_argument('-data_dir', '--data_dir', type=str, default='data/', help="data directory")
parser.add_argument('-epochs', '--epochs', type=int, default=1000, help="epochs")
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

m_train = m_data[:n_train]
m_test = m_data[-n_test:]

x_train = x_data[:n_train]
x_test = x_data[-n_test:]

v_mean_train = v_mean_data[:n_train]
v_mean_test = v_mean_data[-n_test:]

a_coeff_train = a_coeff[:n_train]
a_coeff_test = a_coeff[-n_test:]

b_coeff_train = b_coeff[:n_train]
b_coeff_test = b_coeff[-n_test:]

print(f"\nTrain/test split: {n_train}/{n_test}")

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
test_dataset = WindToPathDataset(x_test, v_mean_test, a_coeff_test, b_coeff_test, m_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
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
    'test_loss': [],
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

best_test_loss = float('inf')
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
    test_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            x_positions = batch['x_positions'].to(device)
            v_mean = batch['v_mean'].to(device)
            a_coeff = batch['a_coeff'].to(device)
            b_coeff = batch['b_coeff'].to(device)
            path_coeffs = batch['path_coeffs'].to(device)
            
            pred_path = model(x_positions, v_mean, a_coeff, b_coeff)
            loss = criterion(pred_path, path_coeffs)
            test_loss += loss.item() * len(x_positions)
    
    test_loss /= len(test_dataset)
    
    # Update learning rate
    current_lr = optimizer.param_groups[0]['lr']
    scheduler.step(test_loss)
    new_lr = optimizer.param_groups[0]['lr']
    
    # Save history
    history['train_loss'].append(train_loss)
    history['test_loss'].append(test_loss)
    history['lr'].append(current_lr)
    
    # Print progress
    if epoch % 50 == 0 or epoch == args.epochs - 1:
        print(f"Epoch {epoch:4d}: train_loss = {train_loss:.6e}, test_loss = {test_loss:.6e}, lr = {current_lr:.2e}")
    
    # Save best model
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'test_loss': test_loss,
        }, best_model_path)
        if epoch % 10 == 0:
            print(f"  → New best model saved (test_loss = {test_loss:.6e})")

# ----------------------------
# Plot training history
# ----------------------------
plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
plt.semilogy(history['train_loss'], label='Train')
plt.semilogy(history['test_loss'], label='Test')
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
history_path = os.path.join(args.save_dir, 'fno_training_history.png')
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

# Compute detailed metrics
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
mean_rel_error = np.mean(relative_errors)
std_rel_error = np.std(relative_errors)

print(f"Test set performance:")
print(f"  Best test loss: {best_test_loss:.6e}")
print(f"  Mean relative L2 error: {mean_rel_error:.6e}")
print(f"  Std relative L2 error: {std_rel_error:.6e}")

# ----------------------------
# Save final model and results
# ----------------------------
final_model_path = os.path.join(args.save_dir, 'final_fno_model.pth')
torch.save({
    'model_state_dict': model.state_dict(),
    'history': history,
    'args': args,
    'test_metrics': {
        'best_test_loss': best_test_loss,
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
print("="*60)