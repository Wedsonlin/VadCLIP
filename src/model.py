from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from clip import clip
from utils.layers import GraphConvolution, DistanceAdj

class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor, padding_mask: torch.Tensor):
        padding_mask = padding_mask.to(dtype=bool, device=x.device) if padding_mask is not None else None
        self.attn_mask = self.attn_mask.to(device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, key_padding_mask=padding_mask, attn_mask=self.attn_mask)[0]

    def forward(self, x):
        x, padding_mask = x
        x = x + self.attention(self.ln_1(x), padding_mask)
        x = x + self.mlp(self.ln_2(x))
        return (x, padding_mask)


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class CLIPVAD(nn.Module):
    def __init__(self,
                 num_class: int,
                 embed_dim: int,
                 visual_length: int,
                 visual_width: int,
                 visual_head: int,
                 visual_layers: int,
                 attn_window: int,
                 prompt_prefix: int,
                 prompt_postfix: int,
                 device):
        """
        Initialize the CLIPVAD model.

        Args:
            num_class: Number of anomaly/action classes predicted by the model.
            embed_dim: Dimensionality of CLIP text embeddings and learned text prompts.
            visual_length: Nnumber of video segments per sample.
            visual_width: Feature dimensionality of each visual segment.
            visual_head: Number of attention heads in the temporal Transformer.
            visual_layers: Number of residual attention blocks in the temporal Transformer.
            attn_window: Temporal window size used to build the local attention mask.
            prompt_prefix: Number of learnable prompt tokens inserted before class text.
            prompt_postfix: Number of learnable prompt tokens inserted after class text.
            device: Torch device used for CLIP loading and tensor placement.
        """
        super().__init__()

        self.num_class = num_class
        self.visual_length = visual_length
        self.visual_width = visual_width
        self.embed_dim = embed_dim
        self.attn_window = attn_window
        self.prompt_prefix = prompt_prefix
        self.prompt_postfix = prompt_postfix
        self.device = device

        self.temporal = Transformer(
            width=visual_width,
            layers=visual_layers,
            heads=visual_head,
            attn_mask=self.build_attention_mask(self.attn_window)
        )

        # width = int(visual_width / 2)
        # self.gc1 = GraphConvolution(visual_width, width, residual=True)
        # self.gc2 = GraphConvolution(width, width, residual=True)
        # self.gc3 = GraphConvolution(visual_width, width, residual=True)
        # self.gc4 = GraphConvolution(width, width, residual=True)
        # self.linear = nn.Linear(visual_width, visual_width) # should be unbiased?

        width = visual_width
        self.gc1 = GraphConvolution(visual_width, width, residual=True)
        # self.gc2 = GraphConvolution(width, visual_width, residual=True)
        self.gc3 = GraphConvolution(visual_width, width, residual=True)
        # self.gc4 = GraphConvolution(width, visual_width, residual=True)
        self.linear = nn.Linear(width*2, visual_width) # should be unbiased?
        
        self.disAdj = DistanceAdj()
        self.gelu = QuickGELU()

        self.mlp1 = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(visual_width, visual_width * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(visual_width * 4, visual_width))
        ]))
        self.mlp2 = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(visual_width, visual_width * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(visual_width * 4, visual_width))
        ]))
        self.classifier = nn.Linear(visual_width, 1)

        self.clipmodel, _ = clip.load("ViT-B/16", device)
        for clip_param in self.clipmodel.parameters():
            clip_param.requires_grad = False

        self.frame_position_embeddings = nn.Embedding(visual_length, visual_width)
        self.text_prompt_embeddings = nn.Embedding(77, self.embed_dim)

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.text_prompt_embeddings.weight, std=0.01)
        nn.init.normal_(self.frame_position_embeddings.weight, std=0.01)

    def build_attention_mask(self, attn_window: int) -> torch.Tensor:
        # lazily create non-overlapping local attention mask with attention window size attn_window
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.visual_length, self.visual_length)
        mask.fill_(float('-inf'))
        for i in range(int(self.visual_length / attn_window)):
            if (i + 1) * attn_window < self.visual_length:
                mask[i * attn_window: (i + 1) * attn_window, i * attn_window: (i + 1) * attn_window] = 0
            else:
                mask[i * attn_window: self.visual_length, i * attn_window: self.visual_length] = 0

        return mask

    def cos_sim_adj(self, x: torch.Tensor, seq_len: torch.Tensor | None = None) -> torch.Tensor:
        normalized_x = F.normalize(x, p=2, dim=2)
        sim = normalized_x @ normalized_x.transpose(1, 2) # (B,T,D) @ (B,D,T) -> (B,T,T)
        sim = F.threshold(sim, 0.7, 0)

        if seq_len is None:
            return  F.softmax(sim, dim=-1)

        output = torch.zeros_like(sim)
        for i,length in enumerate(seq_len.tolist()):
            if length == 0:
                continue

            valid_sim = sim[i, :length, :length]
            output[i, :length, :length] = F.softmax(valid_sim, dim=-1)

        return output

    def encode_video(self, images, padding_mask, lengths):
        images = images.to(torch.float)

        # learnable position embeddings
        position_ids = torch.arange(self.visual_length, device=self.device) # (T,)
        position_ids = position_ids.unsqueeze(0).expand(images.shape[0], -1) # (T,) -> (1,T) -> (B,T) 
        frame_position_embeddings = self.frame_position_embeddings(position_ids) # (B,T,D)
        images = images + frame_position_embeddings

        # non-overlapping local attention transformer
        images = images.permute(1, 0, 2) # (B,T,D) -> (T,B,D)
        x, _ = self.temporal((images, None)) # why doesn't use padding_mask?
        x = x.permute(1, 0, 2) # (T,B,D) -> (B,T,D)

        # two layer GCNs with cosine similarity adjacency matrix
        cos_sim_adj = self.cos_sim_adj(x, lengths)
        x1_h = self.gelu(self.gc1(x, cos_sim_adj))
        # x1 = self.gelu(self.gc2(x1_h, cos_sim_adj))
        
        # two layer GCNs with distance adjacency matrix
        dis_adj = self.disAdj(x.shape[0], x.shape[1])
        x2_h = self.gelu(self.gc3(x, dis_adj))
        # x2 = self.gelu(self.gc4(x2_h, dis_adj))

        # x = torch.cat((x1, x2), 2)
        x = torch.cat((x1_h, x2_h), 2)
        x = self.linear(x)

        return x

    def encode_textprompt(self, text):
        """
        Encode class labels with learnable text prompts and the frozen CLIP text encoder.

        The method tokenizes each label, obtains its CLIP token embeddings, and
        inserts the label tokens into a length-77 learnable prompt sequence:
        ``[SOT] [prefix prompts] label tokens [postfix prompts] [EOT] [PAD]``.
        ``eot_indices`` records the shifted EOT positions so the modified CLIP
        text encoder can pool the final text feature from those tokens.

        Args:
            text: A list of class label strings.

        Returns:
            A tensor of encoded text features with shape ``(len(text), D)``.
        """
        label_tokens = clip.tokenize(text).to(self.device) # [SOT] tokens ... [EOT] [PAD] ... (C,77)
        label_embeddings = self.clipmodel.encode_token(label_tokens) # (C,77,D)
        pad_tokens = torch.zeros_like(label_tokens)
        prompt_embeddings = self.clipmodel.encode_token(pad_tokens) # [PAD] [PAD] ... (C,77,D)
        learnable_prompt_embeddings = self.text_prompt_embeddings(torch.arange(77).to(self.device)).unsqueeze(0).repeat([len(text), 1, 1]) # unified context (C,77,D)
        eot_indices = torch.empty(len(text), dtype=torch.long, device=self.device)
        # [SOT] {learnable prefix prompt tokens} label tokens {learnable postfix prompt tokens} [EOT] [PAD] ... (C,77)

        for i in range(len(text)):
            ind = torch.argmax(label_tokens[i], -1) # index of the EOT
            eot_index = self.prompt_prefix + ind + self.prompt_postfix
            prompt_embeddings[i, 0] = label_embeddings[i, 0] # SOT
            prompt_embeddings[i, 1: self.prompt_prefix + 1] = learnable_prompt_embeddings[i, 1: self.prompt_prefix + 1] # prefix prompt tokens
            prompt_embeddings[i, self.prompt_prefix + 1: self.prompt_prefix + ind] = label_embeddings[i, 1: ind] # label tokens
            prompt_embeddings[i, self.prompt_prefix + ind: eot_index] = learnable_prompt_embeddings[i, self.prompt_prefix + ind: eot_index] # postfix prompt tokens
            prompt_embeddings[i, eot_index] = label_embeddings[i, ind] # EOT
            eot_indices[i] = eot_index

        text_features = self.clipmodel.encode_text(prompt_embeddings, eot_indices)

        return text_features

    def forward(self, video, padding_mask, text, lengths):
        B = video.shape[0]
        video_features = self.encode_video(video, padding_mask, lengths) # (B,T,D)
        logits = self.classifier(video_features + self.mlp2(video_features)) # (B,T,1)
        anomaly_confidence = torch.sigmoid(logits)

        text_features_ori = self.encode_textprompt(text) # (C,D) text features of label classes

        text_features = text_features_ori
        visual_prompt = logits.transpose(1, 2) @ video_features # (B,1,T) @ (B,T,D) -> (B,1,D)
        visual_prompt = F.normalize(visual_prompt, p=2, dim=-1)
        visual_prompt = visual_prompt.expand(B, text_features_ori.shape[0], visual_prompt.shape[2]) # (B,C,D)
        text_features = text_features_ori.unsqueeze(0) # (C,D) -> (1,C,D)
        text_features = text_features.expand(B, -1, -1) # (B,C,D)
        text_features = text_features + visual_prompt 
        text_features = text_features + self.mlp1(text_features) # (B,C,D)
        
        video_features = F.normalize(video_features, p=2, dim=-1) # (B,T,D)
        text_features = F.normalize(text_features, p=2, dim=-1) # (B,C,D)
        alignment_map = video_features @ text_features.transpose(1, 2).type(video_features.dtype) / 0.07 # (B,T,D) @ (B,D,C) -> (B,T,C)

        return text_features_ori, anomaly_confidence, alignment_map
    