import torch
from torch import nn 
from dinotorch_lite.architectures.fno_utils import * 

# ----------------------------
# Create dataset and dataloader for FNO
# ----------------------------
class WindToPathDataset(torch.utils.data.Dataset):
    """Dataset for drone positions + wind coefficients to path coefficients mapping"""
    def __init__(self, x_positions, v_mean, a_coeff, b_coeff, m_data):
        self.x_positions = x_positions  # drone positions
        self.v_mean = v_mean            # mean flow components
        self.a_coeff = a_coeff          # cos-cos coefficients
        self.b_coeff = b_coeff          # sin-cos coefficients
        self.m_data = m_data            # path coefficients
        
    def __len__(self):
        return len(self.m_data)
    
    def __getitem__(self, idx):
        return {
            'x_positions': self.x_positions[idx],  # drone positions
            'v_mean': self.v_mean[idx],             # mean flow
            'a_coeff': self.a_coeff[idx],           # wind mode coefficients (u)
            'b_coeff': self.b_coeff[idx],           # wind mode coefficients (v)
            'path_coeffs': self.m_data[idx]         # target path coefficients
        }


class EnhancedSpectralFNO(nn.Module):
    """
    Enhanced version with attention over modes and frequency-adaptive processing.
    """
    def __init__(self, r=10, K=10, width=64, modes1=8, modes2=8, n_layers=4):
        super().__init__()
        
        self.r = r
        self.K = K
        self.width = width
        
        # Mean flow projection
        self.mean_flow_proj = nn.Linear(2, width)
        
        # Create frequency embeddings for the modes
        # This helps the network understand the physical meaning of (i,j) indices
        i_indices = torch.arange(1, r+1).view(1, -1, 1).float()
        j_indices = torch.arange(1, r+1).view(1, 1, -1).float()
        self.register_buffer('i_grid', i_indices)
        self.register_buffer('j_grid', j_indices)
        
        # Learnable embeddings for frequency
        self.freq_embed = nn.Embedding(r, width//4)  # rough embedding for i
        # We'll combine these later
        
        # Initial projection with frequency awareness
        self.grid_proj = nn.Sequential(
            nn.Conv2d(2, width, 1),  # initial projection
            # Could add frequency-dependent scaling here
        )
        
        # Fourier layers
        self.fourier_layers = nn.ModuleList()
        self.conv_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for _ in range(n_layers):
            self.fourier_layers.append(
                SpectralConv2d(width, width, modes1, modes2)
            )
            self.conv_layers.append(
                nn.Conv2d(width, width, 1)
            )
            self.norms.append(nn.LayerNorm([width, r, r]))
        
        # Path coefficient decoder with structured output
        # We'll output mean (x̄, ȳ) and then K groups of 4 coefficients
        self.mean_decoder = nn.Linear(width + width, 2)
        
        # Separate decoder for each frequency? 
        # Could share weights or have separate
        self.freq_decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(width + width, 64),
                nn.GELU(),
                nn.Linear(64, 4)  # θ_k, φ_k, ψ_k, η_k
            ) for _ in range(K)
        ])
        
    def forward(self, v_mean, a_coeff, b_coeff):
        batch_size = v_mean.shape[0]
        
        # Mean flow features
        mean_features = self.mean_flow_proj(v_mean) 
        
        # Combine coefficients
        mode_grid = torch.stack([a_coeff, b_coeff], dim=1)  # (batch, 2, r, r)
        
        # Add frequency positional encodings
        # This helps the network know which (i,j) it's looking at
        i_embed = self.freq_embed((self.i_grid - 1).long()).squeeze(0)  # (r, 1, width//4)
        j_embed = self.freq_embed((self.j_grid - 1).long()).squeeze(0)  # (1, r, width//4)
        
        # Project and add positional info
        x = self.grid_proj(mode_grid)
        
        # Fourier layers
        for fourier, conv, norm in zip(self.fourier_layers, self.conv_layers, self.norms):
            x_fourier = fourier(x)
            x_conv = conv(x)
            x = x + x_fourier + x_conv
            x = norm(x)
            x = F.gelu(x)
        
        # Global pooling
        grid_features = x.mean(dim=[-2, -1])  # (batch, width)
        
        # Combine features
        combined = torch.cat([grid_features, mean_features], dim=-1)
        
        # Decode mean position
        mean_pos = self.mean_decoder(combined)  # (batch, 2)
        
        # Decode each frequency's coefficients
        freq_coeffs = []
        for decoder in self.freq_decoders:
            coeff = decoder(combined)  # (batch, 4)
            freq_coeffs.append(coeff)
        
        # Concatenate all coefficients
        # Shape: (batch, 2 + K*4)
        path_coeffs = torch.cat([mean_pos] + freq_coeffs, dim=-1)
        
        return path_coeffs

# ----------------------------
# Enhanced FNO model that accepts all three inputs
# ----------------------------
class FullInputSpectralFNO(EnhancedSpectralFNO):
    """
    Enhanced version of SpectralFNO that also accepts drone positions as input.
    """
    def __init__(self, r=10, K=10, width=64, modes1=8, modes2=8, n_layers=4):
        super().__init__(r, K, width, modes1, modes2, n_layers)
        
        
        # Projection for drone positions
        self.drone_pos_proj = nn.Linear( 2, width)
        
        # Modify the final projection to include drone features
        # Original final_proj from EnhancedSpectralFNO expects width + width
        # Now we'll have width (mean flow) + width (drone positions) + width (grid features)
        self.final_proj_modified = nn.Sequential(
            nn.Linear(3 * width, 256),  # grid + mean flow + drone positions
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 4*K + 2)
        )
        
        # Replace the mean_decoder and freq_decoders to use all features
        self.mean_decoder = nn.Linear(3 * width, 2)
        self.freq_decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(3 * width, 64),
                nn.GELU(),
                nn.Linear(64, 4)
            ) for _ in range(K)
        ])
        
    def forward(self, x_positions, v_mean, a_coeff, b_coeff):
        """
        Args:
            x_positions: (batch, 2) - drone positions (x,y) for each drone
            v_mean: (batch, 2) - v_x, v_y mean flow components
            a_coeff: (batch, r, r) - coefficients for u (cos-cos modes)
            b_coeff: (batch, r, r) - coefficients for v (sin-cos modes)
        
        Returns:
            path_coeffs: (batch, 4*K + 2) - path Fourier coefficients
        """
        batch_size = v_mean.shape[0]
        
        # Process drone positions
        drone_features = self.drone_pos_proj(x_positions)  # (batch, width)
        
        # Process mean flow (original)
        mean_features = self.mean_flow_proj(v_mean)  # (batch, width)
        
        # Process mode grid (original)
        mode_grid = torch.stack([a_coeff, b_coeff], dim=1)  # (batch, 2, r, r)
        
        # Add frequency positional encodings
        i_embed = self.freq_embed((self.i_grid - 1).long()).squeeze(0)  # (r, 1, width//4)
        j_embed = self.freq_embed((self.j_grid - 1).long()).squeeze(0)  # (1, r, width//4)
        
        # Project and add positional info
        x = self.grid_proj(mode_grid)
        
        # Fourier layers
        for fourier, conv, norm in zip(self.fourier_layers, self.conv_layers, self.norms):
            x_fourier = fourier(x)
            x_conv = conv(x)
            x = x + x_fourier + x_conv
            x = norm(x)
            x = F.gelu(x)
        
        # Global pooling over mode grid
        grid_features = x.mean(dim=[-2, -1])  # (batch, width)
        
        # Combine all three feature types
        combined = torch.cat([grid_features, mean_features, drone_features], dim=-1)  # (batch, 3*width)
        
        # Option 1: Use separate decoders for structured output
        mean_pos = self.mean_decoder(combined)  # (batch, 2)
        
        freq_coeffs = []
        for decoder in self.freq_decoders:
            coeff = decoder(combined)  # (batch, 4)
            freq_coeffs.append(coeff)
        
        path_coeffs = torch.cat([mean_pos] + freq_coeffs, dim=-1)
        
        return path_coeffs