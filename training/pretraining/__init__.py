"""Self-supervised pretraining methods."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimCLR(nn.Module):
    def __init__(self, encoder: nn.Module, projection_dim: int = 128, temperature: float = 0.5) -> None:
        super().__init__()
        self.encoder = encoder
        self.projector = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, projection_dim),
        )
        self.temperature = temperature

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        z1 = F.normalize(self.projector(self._encode(x1)), dim=1)
        z2 = F.normalize(self.projector(self._encode(x2)), dim=1)
        return self._nt_xent_loss(z1, z2)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return features.mean(dim=[2, 3, 4]) if features.dim() == 5 else features.mean(dim=[2, 3])

    def _nt_xent_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        batch_size = z1.shape[0]
        z = torch.cat([z1, z2], dim=0)
        sim = torch.mm(z, z.t()) / self.temperature
        mask = torch.eye(2 * batch_size, device=z.device).bool()
        sim.masked_fill_(mask, float("-inf"))
        labels = torch.cat([torch.arange(batch_size, 2 * batch_size), torch.arange(batch_size)]).to(z.device)
        return F.cross_entropy(sim, labels)


class MAEPretrainer(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module, mask_ratio: float = 0.75) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.mask_ratio = mask_ratio

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, d, h, w = x.shape
        num_patches = d * h * w
        num_mask = int(num_patches * self.mask_ratio)
        noise = torch.rand(b, num_patches, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        x_flat = x.reshape(b, c, -1).transpose(1, 2)
        x_masked = x_flat.clone()
        for i in range(b):
            x_masked[i, ids_shuffle[i, :num_mask]] = 0

        encoded = self.encoder(x_masked.transpose(1, 2).reshape(b, c, d, h, w))
        decoded = self.decoder(encoded)
        return F.mse_loss(decoded, x)


class BYOLPretrainer(nn.Module):
    def __init__(self, online_encoder: nn.Module, target_encoder: nn.Module, projection_dim: int = 256) -> None:
        super().__init__()
        self.online_encoder = online_encoder
        self.target_encoder = target_encoder
        self.online_projector = nn.Sequential(nn.Linear(512, projection_dim), nn.ReLU(), nn.Linear(projection_dim, projection_dim))
        self.target_projector = nn.Sequential(nn.Linear(512, projection_dim), nn.ReLU(), nn.Linear(projection_dim, projection_dim))
        self.predictor = nn.Sequential(nn.Linear(projection_dim, projection_dim), nn.ReLU(), nn.Linear(projection_dim, projection_dim))
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        for p in self.target_projector.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update_target(self, momentum: float = 0.996) -> None:
        for online, target in zip(self.online_encoder.parameters(), self.target_encoder.parameters(), strict=True):
            target.data = momentum * target.data + (1 - momentum) * online.data

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        online_repr = self.predictor(self.online_projector(self._encode(self.online_encoder, x1)))
        with torch.no_grad():
            target_repr = self.target_projector(self._encode(self.target_encoder, x2))
        online_repr = F.normalize(online_repr, dim=1)
        target_repr = F.normalize(target_repr, dim=1)
        return 2 - 2 * (online_repr * target_repr).sum(dim=1).mean()

    def _encode(self, encoder: nn.Module, x: torch.Tensor) -> torch.Tensor:
        features = encoder(x)
        return features.mean(dim=[2, 3, 4]) if features.dim() == 5 else features.mean(dim=[2, 3])


class DINOv2Pretrainer(nn.Module):
    """DINOv2-style self-distillation pretraining."""

    def __init__(self, student: nn.Module, teacher: nn.Module, out_dim: int = 65536) -> None:
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.student_head = nn.Linear(512, out_dim)
        self.teacher_head = nn.Linear(512, out_dim)
        for p in self.teacher.parameters():
            p.requires_grad = False
        for p in self.teacher_head.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update_teacher(self, momentum: float = 0.996) -> None:
        for s, t in zip(self.student.parameters(), self.teacher.parameters(), strict=True):
            t.data = momentum * t.data + (1 - momentum) * s.data
        for s, t in zip(self.student_head.parameters(), self.teacher_head.parameters(), strict=True):
            t.data = momentum * t.data + (1 - momentum) * s.data

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        s_out = self.student_head(self._encode(self.student, x1))
        with torch.no_grad():
            t_out = self.teacher_head(self._encode(self.teacher, x2))
        s_out = F.log_softmax(s_out / 0.1, dim=1)
        t_out = F.softmax(t_out / 0.04, dim=1)
        return -(t_out * s_out).sum(dim=1).mean()

    def _encode(self, encoder: nn.Module, x: torch.Tensor) -> torch.Tensor:
        features = encoder(x)
        return features.mean(dim=[2, 3, 4]) if features.dim() == 5 else features.mean(dim=[2, 3])


PRETRAIN_REGISTRY = {
    "simclr": SimCLR,
    "mae": MAEPretrainer,
    "byol": BYOLPretrainer,
    "dinov2": DINOv2Pretrainer,
}
