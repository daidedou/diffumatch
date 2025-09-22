# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Loss functions used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import torch
from edm.torch_utils import persistence
import pdb
#----------------------------------------------------------------------------
# Loss function corresponding to the variance preserving (VP) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

@persistence.persistent_class
class VPLoss:
    def __init__(self, beta_d=19.9, beta_min=0.1, epsilon_t=1e-5):
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.epsilon_t = epsilon_t

    def noise_and_weight(self, shape, device, sds=False):
        rnd_uniform = torch.rand([shape, 1, 1, 1], device=device)
        if sds:
            rnd_uniform = 0.02 + rnd_uniform*0.96 #Between O.O2 and 0.98, see https://github.com/ashawkey/stable-dreamfusion/blob/5550b91862a3af7842bb04875b7f1211e5095a63/guidance/sd_utils.py#L180
        sigma = self.sigma(1 + rnd_uniform * (self.epsilon_t - 1))
        weight = 1 / sigma ** 2
        return sigma, weight

    def __call__(self, net, x, latents, augment_pipe=None):
        sigma, weight = self.noise_and_weight(x.shape[0], x.device)
        n = torch.randn_like(x) * sigma
        D_xn = net(x + n, sigma, latents)
        loss = weight * ((D_xn - x) ** 2)
        return loss

    def sigma(self, t):
        t = torch.as_tensor(t)
        return ((0.5 * self.beta_d * (t ** 2) + self.beta_min * t).exp() - 1).sqrt()

#----------------------------------------------------------------------------
# Loss function corresponding to the variance exploding (VE) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

@persistence.persistent_class
class VELoss:
    def __init__(self, sigma_min=0.02, sigma_max=100):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def noise_and_weight(self, shape, device, sds=False):
        rnd_uniform = torch.rand([x.shape[0], 1], device=x.device)
        sigma = self.sigma_min * ((self.sigma_max / self.sigma_min) ** rnd_uniform)
        weight = 1 / sigma ** 2
        return sigma, weight

    def __call__(self, net, x, latents, augment_pipe=None):
        sigma, weight = self.noise_and_weight(x.shape[0], x.device)
        n = torch.randn_like(x) * sigma
        D_xn = net(x + n, sigma, latents)
        loss = weight * ((D_xn - x) ** 2)
        return loss

#----------------------------------------------------------------------------
# Improved loss function proposed in the paper "Elucidating the Design Space
# of Diffusion-Based Generative Models" (EDM).

@persistence.persistent_class
class EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.sigma_min = 0.4
        self.sigma_max = 10
        self.rho=3
        
    def noise_and_weight(self, shape, device, sds=False):
        rnd_normal = torch.randn([shape, 1, 1, 1], device=device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        return sigma.float(), weight.float()

    def __call__(self, net, x, latents, augment_pipe=None):
        sigma, weight = self.noise_and_weight(x.shape[0], x.device)
        n = torch.randn_like(x) * sigma
        D_xn = net(x + n, sigma, latents)
        loss = weight * ((D_xn - x) ** 2)
        return loss

#----------------------------------------------------------------------------
