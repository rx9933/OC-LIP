# MIT License
# Copyright (c) 2025
#
# This is part of the dino_tutorial package
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
# For additional questions contact Thomas O'Leary-Roseberry
import torch
from torch import nn
from torch.utils.data import Dataset
import numpy as np
from scipy.sparse import csr_matrix

# ---------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------
class QtoMTranslatorH1(nn.Module):
    def __init__(self, input_dim, output_dim, latent_dim, width):
        super(QtoMTranslatorH1, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        self.neural_network = nn.Sequential(
            nn.Linear(latent_dim, width),
            nn.GELU(),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        z_transformed = self.neural_network(z)
        x_recon = self.decoder(z_transformed)
        return x_recon


class QtoMTranslator(nn.Module):
    def __init__(self, input_dim, output_dim, latent_dim):
        super(QtoMTranslator, self).__init__()
        
        # Encoder (q -> latent space)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        
        # Decoder (latent space -> m)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )
    
    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon
class QtoMTranslatorH1(nn.Module):
    def __init__(self, input_dim, output_dim, latent_dim, width):
        super(QtoMTranslatorH1, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        self.neural_network = nn.Sequential(
            nn.Linear(latent_dim, width),
            nn.GELU(),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        z_transformed = self.neural_network(z)
        x_recon = self.decoder(z_transformed)
        return x_recon

class MtoQTranslator(nn.Module):
    def __init__(self, input_dim, output_dim, width):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.neural_network = nn.Sequential(
            nn.Linear(128, width),
            nn.GELU(),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, 128),
        )
        self.decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        zt = self.neural_network(z)
        return self.decoder(zt)





class Encoder(nn.Module):
    def __init__(self, input_dim, output_dim, width):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.neural_network = nn.Sequential(
            nn.Linear(128, width),
            nn.GELU(),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, 128),
        )
        self.decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        zt = self.neural_network(z)
        return self.decoder(zt)



class Encoder2NN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim = 200):
        super().__init__()
        self.encoder= nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim) #latent dim
        )
        

        self.neural_network = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )
        
    def forward(self, x):
        z = self.encoder(x)
        zt = self.neural_network(z)
        return zt


class LatentToOutput(nn.Module):
    """
    Wrapper that takes a loaded Encoder model and uses only neural_network + decoder.
    This gives you φθ1(z) - the forward map from latent to output.
    """
    def __init__(self, full_encoder_model):
        super().__init__()
        # Extract the relevant parts from the loaded model
        self.neural_network = full_encoder_model.neural_network
        self.decoder = full_encoder_model.decoder
        
        # Optionally freeze them
        for param in self.neural_network.parameters():
            param.requires_grad = False
        for param in self.decoder.parameters():
            param.requires_grad = False
            
    def forward(self, z):
        """
        Args:
            z: latent vector of shape (batch_size, 128) or (128,)
        """
        # Handle both batched and single inputs
        if z.dim() == 1:
            z = z.unsqueeze(0)
            
        zt = self.neural_network(z)  # Process latent
        q = self.decoder(zt)          # Map to output
        
        # Return in same shape as input
        if z.size(0) == 1 and z.dim() == 2:
            q = q.squeeze(0)
            
        return q



# ----------------------------
# Decoder2 Network - maps latent back to parameter space
# ----------------------------
class Decoder2(nn.Module):
    """Decoder2 network: latent (128) -> M (parameter space)"""
    def __init__(self, latent_dim, output_dim, hidden_dim=200):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, z):
        return self.net(z)

# ----------------------------
# Autoencoder for training Decoder2 with frozen encoder
# ----------------------------
class LatentAutoencoder(nn.Module):
    """Autoencoder: m -> encoder (frozen) -> z -> decoder2 (trainable) -> m_hat"""
    def __init__(self, encoder, decoder2):
        super().__init__()
        self.encoder = encoder
        self.decoder2 = decoder2
        
        # Freeze the encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        
    def forward(self, m):
        z = self.encoder(m)
        m_hat = self.decoder2(z)
        return m_hat, z


class GenericDense(nn.Module):
    def __init__(self,  input_dim=50, hidden_layer_dim = 256, output_dim=20):
        super().__init__()

        self.hidden1 = nn.Linear(input_dim, hidden_layer_dim)
        self.act1 = nn.GELU()
        self.hidden2 = nn.Linear(hidden_layer_dim, hidden_layer_dim)
        self.act2 = nn.GELU()
        self.hidden3 = nn.Linear(hidden_layer_dim, hidden_layer_dim)
        self.act3 = nn.GELU()
        self.hidden4 = nn.Linear(hidden_layer_dim, hidden_layer_dim)
        self.act4 = nn.GELU()
        self.output = nn.Linear(hidden_layer_dim, output_dim)

    def forward(self, x):
        x = self.act1(self.hidden1(x))
        x = self.act2(self.hidden2(x))
        x = self.act3(self.hidden3(x))
        x = self.act4(self.hidden4(x))
        x = self.output(x)
        return x

class GenericDenseSkip(nn.Module):
    def __init__(self, input_dim=50, hidden_layer_dim=256, output_dim=20):
        super().__init__()
        
        # Store dimensions for skip connections
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Main layers
        self.hidden1 = nn.Linear(input_dim, hidden_layer_dim)
        self.act1 = nn.GELU()
        self.hidden2 = nn.Linear(hidden_layer_dim, hidden_layer_dim)
        self.act2 = nn.GELU()
        self.hidden3 = nn.Linear(hidden_layer_dim, hidden_layer_dim)
        self.act3 = nn.GELU()
        self.hidden4 = nn.Linear(hidden_layer_dim, hidden_layer_dim)
        self.act4 = nn.GELU()
        self.output = nn.Linear(hidden_layer_dim, output_dim)
        
        # Skip connection adapters to match dimensions
        # These project the first two input features to the output dimension
        self.skip1_adapter = nn.Linear(1, output_dim)  # For first input point
        self.skip2_adapter = nn.Linear(1, output_dim)  # For second input point
        
    def forward(self, x):
        # Main path
        main = self.act1(self.hidden1(x))
        main = self.act2(self.hidden2(main))
        main = self.act3(self.hidden3(main))
        main = self.act4(self.hidden4(main))
        main = self.output(main)
        
        # Skip connections from first two input points to output
        # Extract first two features from input
        x1 = x[:, 0:1]  # First input point (feature)
        x2 = x[:, 1:2]  # Second input point (feature)
        
        # Project them to output dimension
        skip1 = self.skip1_adapter(x1)
        skip2 = self.skip2_adapter(x2)
        
        # Combine main path with skip connections
        output = main + skip1 + skip2
        
        return output
def squared_f_norm(A):
    return torch.sum(torch.square(A))

def squared_f_error(A_pred, A_true):
    return squared_f_norm(A_true - A_pred)

def f_mse(A_pred_batched, A_true_batched):
    return torch.mean(torch.vmap(squared_f_error, in_dims=(0, 0), out_dims=0)(A_pred_batched, A_true_batched), axis=0)

def normalized_f_mse(A_pred_batched, A_true_batched):
    err = f_mse(A_pred_batched, A_true_batched)
    normalization = torch.mean(torch.vmap(squared_f_norm)(A_true_batched), axis=0)
    return err / normalization



class L2Dataset(Dataset):
    """
    L2NO dataset
    Each sample is a pair of (m, u) where m is the parameter and u is the state
    """
    def __init__(self, m_data: torch.Tensor, u_data: torch.Tensor):
        """
        Initialize the dataset

        Input:
        - m_data: torch.Tensor, shape (n_data, m_dim)
        - u_data: torch.Tensor, shape (n_data, u_dim)
        """
        assert m_data.shape[0] == u_data.shape[0], "m_data and u_data must have the same number of samples"

        self.m_data = m_data
        self.u_data = u_data


    def __len__(self):
        return self.m_data.shape[0]


    def __getitem__(self, idx):
        return self.m_data[idx], self.u_data[idx]

class DINODataset(Dataset):
    """
    DINO dataset
    Each sample is a triplet of (m, u, J) where m is the parameter, u is the state and j is the jacobian
    """
    def __init__(self, m_data: torch.Tensor, u_data: torch.Tensor, J_data: torch.Tensor):
        """
        Initialize the dataset

        Input:
        - m_data: torch.Tensor, shape (n_data, m_dim)
        - u_data: torch.Tensor, shape (n_data, u_dim)
        - J_data: torch.Tensor, shape (n_data, u_dim, m_dim)
        """
        assert m_data.shape[0] == u_data.shape[0] == J_data.shape[0], "m_data, u_data and j_data must have the same number of samples"

        self.m_data = m_data
        self.u_data = u_data
        self.J_data = J_data


    def __len__(self):
        return self.m_data.shape[0]


    def __getitem__(self, idx):
        return self.m_data[idx], self.u_data[idx], self.J_data[idx]

def scipy_csr_to_torch_csr(A: csr_matrix) -> torch.Tensor:
    # SciPy's index arrays are usually int32; PyTorch CSR needs int64.
    crow = torch.from_numpy(A.indptr.astype(np.int64, copy=False))  # may copy only if type differs
    col  = torch.from_numpy(A.indices.astype(np.int64, copy=False)) # may copy only if type differs
    val  = torch.from_numpy(A.data)                                 # shares memory (no copy)
    return torch.sparse_csr_tensor(crow, col, val, size=A.shape)
def weighted_l2_error(M: torch.sparse):
    def _weighted_l2_error(u_pred, u_true):
        x = u_pred-u_true
        Mx = torch.sparse.mm(M, torch.t(x))
        return torch.mean(torch.einsum("ij,ji->i", x, Mx),axis = 0)
    return _weighted_l2_error






