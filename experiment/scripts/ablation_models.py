from __future__ import annotations

import torch

from common import CLIPVAD


class AdapterAblationCLIPVAD(CLIPVAD):
    """CLIPVAD variant with switchable LGT-Adapter components for eval/training ablations."""

    VALID_VARIANTS = {
        "full",
        "no_transformer",
        "no_gcn",
        "cosine_gcn_only",
        "distance_gcn_only",
        "transformer_only",
    }

    def __init__(self, *args, variant: str = "full", **kwargs):
        if variant not in self.VALID_VARIANTS:
            raise ValueError(f"Unknown adapter ablation variant: {variant}")
        self.variant = variant
        super().__init__(*args, **kwargs)

    def _position_encoded_video(self, images: torch.Tensor) -> torch.Tensor:
        images = images.to(torch.float)
        position_ids = torch.arange(self.visual_length, device=images.device)
        position_ids = position_ids.unsqueeze(0).expand(images.shape[0], -1)
        return images + self.frame_position_embeddings(position_ids)

    def _temporal_features(self, images: torch.Tensor) -> torch.Tensor:
        if self.variant == "no_transformer":
            return images
        x = images.permute(1, 0, 2)
        x, _ = self.temporal((x, None))
        return x.permute(1, 0, 2)

    def _gcn_features(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        width = int(self.visual_width / 2)
        zeros = torch.zeros(x.shape[0], x.shape[1], width, device=x.device, dtype=x.dtype)

        if self.variant in {"no_gcn", "transformer_only"}:
            return self.linear(x)

        cos_sim_adj = self.cos_sim_adj(x, lengths)
        x1_h = self.gelu(self.gc1(x, cos_sim_adj))
        x1 = self.gelu(self.gc2(x1_h, cos_sim_adj))

        dis_adj = self.disAdj(x.shape[0], x.shape[1]).to(x.device)
        x2_h = self.gelu(self.gc3(x, dis_adj))
        x2 = self.gelu(self.gc4(x2_h, dis_adj))

        if self.variant == "cosine_gcn_only":
            x = torch.cat((x1, zeros), dim=2)
        elif self.variant == "distance_gcn_only":
            x = torch.cat((zeros, x2), dim=2)
        else:
            x = torch.cat((x1, x2), dim=2)

        return self.linear(x)

    def encode_video(self, images, padding_mask, lengths):
        images = self._position_encoded_video(images)
        x = self._temporal_features(images)
        return self._gcn_features(x, lengths)
