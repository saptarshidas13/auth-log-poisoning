"""A small MLBAC/DLBAC-style classifier for CPU-scale reproduction (Section 4.2, C3).

The DLBAC_alpha paper uses a ResNet over an image-like encoding of user/resource
metadata. We use our own simplified architecture instead -- a feed-forward network over
hashed categorical embeddings of each metadata column -- documented here rather than
claimed as a literal reproduction. Metadata values are arbitrary integer-coded
categories (ids can run into the hundreds of thousands in the real-world datasets), so
we hash each value into a fixed-size embedding table (standard feature hashing) instead
of allocating one embedding row per raw id.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MLBACClassifier(nn.Module):
    def __init__(
        self,
        num_user_attrs: int,
        num_res_attrs: int,
        num_ops: int,
        hash_buckets: int = 4096,
        embed_dim: int = 16,
        hidden: int = 128,
    ):
        super().__init__()
        self.hash_buckets = hash_buckets
        self.embed = nn.Embedding(hash_buckets, embed_dim)
        in_dim = (num_user_attrs + num_res_attrs) * embed_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_ops),
        )

    def forward(self, user_attrs: torch.Tensor, res_attrs: torch.Tensor) -> torch.Tensor:
        """user_attrs: (B, num_user_attrs) long; res_attrs: (B, num_res_attrs) long.
        Returns logits of shape (B, num_ops)."""
        ids = torch.cat([user_attrs, res_attrs], dim=1) % self.hash_buckets
        e = self.embed(ids)
        e = e.reshape(e.size(0), -1)
        return self.net(e)
