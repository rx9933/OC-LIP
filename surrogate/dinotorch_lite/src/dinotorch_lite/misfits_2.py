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
#
# For additional questions contact Thomas O'Leary-Roseberry
#
# Author: Blake Christiersen

from __future__ import annotations


import hippylib as hp
import torch
from torch import nn

class WeightedQuadraticMisfit(nn.Module):
    def __init__(self, B: torch.Tensor, d: torch.Tensor, Cn: torch.Tensor):
        super().__init__()
        # Use register_buffer to properly handle device movement
        self.register_buffer('B', B)
        self.register_buffer('d', d) 
        self.register_buffer('Cn', Cn)
        
        # Precompute Cn inverse for efficiency
        if self.Cn.dim() == 0:
            self.Cn_inv = 1.0 / self.Cn
        else:
            # For vector Cn, use element-wise division
            self.Cn_inv = 1.0 / self.Cn

    def forward(self, u: torch.Tensor = None, q: torch.Tensor = None) -> torch.Tensor:
        if u is None and q is None:
            raise ValueError("Either u or q must be provided")
        
        if q is None:
            q = self.B @ u
        
        # Ensure q is on same device as d
        q = q.to(self.d.device)
        
        e = self.d - q
        return 0.5 * torch.dot(e, self.Cn_inv * e)

    @staticmethod
    def from_hippylib(misfit: hp.DiscreteStateObservation,
                  device: torch.device = None,
                  dtype: torch.dtype = torch.float32):
    
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        B = torch.as_tensor(misfit.B.array(), device=device, dtype=dtype)
        
        # SIMPLEST: Use the same method that works in your main script
        # Get the underlying numpy array directly from the vector
        obs_data = misfit.d  # This should work for hippylib vectors
        d = torch.as_tensor(obs_data, device=device, dtype=dtype)
        
        # Handle noise variance
        if isinstance(misfit.noise_variance, (float, int)):
            Cn = torch.tensor(misfit.noise_variance, device=device, dtype=dtype)
        else:
            # For matrix/vector noise, use get_local()
            try:
                noise_data = misfit.noise_variance.get_local()
                Cn = torch.as_tensor(noise_data, device=device, dtype=dtype)
            except:
                Cn = torch.tensor(1.0, device=device, dtype=dtype)  # Default
        
        return WeightedQuadraticMisfit(B, d, Cn)