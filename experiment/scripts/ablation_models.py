from __future__ import annotations

import torch

from common import CLIPVAD
from model import Transformer


class AdapterAblationCLIPVAD(CLIPVAD):
    """CLIPVAD variant with switchable LGT-Adapter components for eval/training ablations."""

    TABLE5_VARIANTS = {
        "baseline",
        "global_tf",
        "local_tf",
        "only_gcn",
        "local_global_tf",
        "global_tf_gcn",
        "lgt_adapter",
    }
    VARIANT_ALIASES = {
        "full": "lgt_adapter",
        "no_transformer": "only_gcn",
        "no_gcn": "local_tf",
        "transformer_only": "local_tf",
    }
    VALID_VARIANTS = TABLE5_VARIANTS | set(VARIANT_ALIASES)

    def __init__(self, *args, variant: str = "full", **kwargs):
        if variant not in self.VALID_VARIANTS:
            raise ValueError(f"Unknown adapter ablation variant: {variant}")
        self.variant = self.VARIANT_ALIASES.get(variant, variant)
        super().__init__(*args, **kwargs)
        self.global_temporal = Transformer(
            width=self.visual_width,
            layers=self.temporal.layers,
            heads=self.temporal.resblocks[0].attn.num_heads,
            attn_mask=None,
        )

    def _position_encoded_video(self, images: torch.Tensor) -> torch.Tensor:
        images = images.to(torch.float)
        position_ids = torch.arange(self.visual_length, device=images.device)
        position_ids = position_ids.unsqueeze(0).expand(images.shape[0], -1)
        return images + self.frame_position_embeddings(position_ids)

    def _transformer_features(self, images: torch.Tensor, transformer: Transformer) -> torch.Tensor:
        x = images.permute(1, 0, 2)
        x_res = x
        x, _ = transformer((x, None))
        x = x_res + x
        return x.permute(1, 0, 2)

    def _gcn_features(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        cos_sim_adj = self.cos_sim_adj(x, lengths)
        dis_adj = self.disAdj(x.shape[0], x.shape[1]).to(x.device)
        
        x1_h = self.gelu(self.gc1(x, cos_sim_adj))
        x1 = self.gelu(self.gc2(x1_h, cos_sim_adj))

        x2_h = self.gelu(self.gc3(x, dis_adj))
        x2 = self.gelu(self.gc4(x2_h, dis_adj))

        # x1 = self.gelu(self.gc1(x, cos_sim_adj))
        # x2 = self.gelu(self.gc3(x, dis_adj))

        return torch.cat((x1, x2), dim=2)

    def encode_video(self, images, padding_mask, lengths):
        images = self._position_encoded_video(images)
        if self.variant == "baseline":
            return images
        if self.variant == "global_tf":
            return self._transformer_features(images, self.global_temporal)
        if self.variant == "local_tf":
            return self._transformer_features(images, self.temporal)
        if self.variant == "only_gcn":
            return self.linear(self._gcn_features(images, lengths))
        if self.variant == "local_global_tf":
            x = self._transformer_features(images, self.temporal)
            return self._transformer_features(x, self.global_temporal)
        if self.variant == "global_tf_gcn":
            x = self._transformer_features(images, self.global_temporal)
            return self.linear(self._gcn_features(x, lengths))
        if self.variant == "lgt_adapter":
            x = self._transformer_features(images, self.temporal)
            return self.linear(self._gcn_features(x, lengths))
        raise ValueError(f"Unhandled adapter ablation variant: {self.variant}")
