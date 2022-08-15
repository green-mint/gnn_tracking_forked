from __future__ import annotations

import torch
from torch.nn.functional import binary_cross_entropy, mse_loss, relu


class EdgeWeightLoss(torch.nn.Module):
    def forward(self, w, beta, x, y, particle_id):
        bce_loss = binary_cross_entropy(w, y, reduction="mean")
        return bce_loss


T = torch.tensor


class PotentialLoss(torch.nn.Module):
    def __init__(self, q_min=0.01, device="cpu"):
        super().__init__()
        self.q_min = q_min
        self.device = device
        #: Scale up repulsive force by this factor
        self.repulsion_scaling = 10
        self.radius_threshold = 1.0

    def condensation_loss(self, beta: T, x: T, particle_id: T) -> T:
        q = torch.arctanh(beta) ** 2 + self.q_min
        pids = torch.unique(particle_id[particle_id > 0])

        masks = particle_id[:, None] == pids[None, :]  # type: ignore

        alphas = torch.argmax(q[:, None] * masks, dim=0)
        x_alphas = x[alphas].t().to(self.device)
        q_alphas = q[alphas][None, None, :].to(self.device)

        diff = x[:, :, None] - x_alphas[None, :, :]
        norm_sq = torch.sum(diff**2, dim=1)
        va = (norm_sq * q_alphas).squeeze(dim=0)
        vr = (relu(self.radius_threshold - torch.sqrt(norm_sq)) * q_alphas).squeeze(
            dim=0
        )
        loss = q[:, None] * (masks * va + self.repulsion_scaling * (~masks) * vr)
        return torch.sum(torch.mean(loss, dim=0))

    def forward(self, w: T, beta: T, x: T, y: T, particle_id: T) -> T:
        return self.condensation_loss(beta, x, particle_id)


class BackgroundLoss(torch.nn.Module):
    def __init__(self, q_min=0.01, sb=0.1, device="cpu"):
        super().__init__()
        #: Strength of noise suppression
        self.sb = sb
        self.q_min = q_min
        self.device = device

    def background_loss(self, beta: T, particle_id: T) -> T:
        pids = torch.unique(particle_id[particle_id > 0])
        masks = particle_id[:, None] == pids[None, :]
        alphas = torch.argmax(masks * beta[:, None], dim=0)
        beta_alphas = beta[alphas]

        loss = torch.mean(1 - beta_alphas)
        noise_mask = particle_id == 0
        if noise_mask.any():
            loss += self.sb * torch.mean(beta[noise_mask])
        return loss

    def forward(self, w, beta, x, y, particle_id):
        return self.background_loss(beta, particle_id)


class ObjectLoss(torch.nn.Module):
    def __init__(self, q_min=0.01, sb=0.1, device="cpu", mode="efficiency"):
        super().__init__()
        #: Strength of noise suppression
        self.sb = sb
        self.q_min = q_min
        self.device = device
        self.mode = mode
        #: Scale up loss value by this factor
        self.scale = 100

    def MSE(self, p, t):
        return torch.sum(mse_loss(p, t, reduction="none"), dim=1)

    def object_loss(self, pred, beta, truth, particle_id):
        noise_mask = particle_id == 0
        xi = (~noise_mask) * torch.arctanh(beta) ** 2
        mse = self.MSE(pred, truth)
        if self.mode == "purity":
            return self.scale / torch.sum(xi) * torch.mean(xi * mse)
        loss = torch.tensor(0.0, dtype=torch.float).to(self.device)
        K = torch.tensor(0.0, dtype=torch.float).to(self.device)
        pids = torch.unique(particle_id[particle_id > 0])
        masks = particle_id[:, None] == pids[None, :]
        for i in range(len(pids)):
            M = masks[:, i]
            xi_p = M * pids[i]
            weight = 1.0 / (torch.sum(xi_p))
            loss += weight * torch.sum(mse * xi_p)
            K += 1.0
        return self.scale * loss / K

    def forward(self, W, beta, H, pred, Y, particle_id, track_params, reconstructable):
        mask = reconstructable > 0
        return self.object_loss(
            pred[mask], beta[mask], track_params[mask], particle_id[mask]
        )
