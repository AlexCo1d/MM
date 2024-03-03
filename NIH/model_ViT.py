"""
Model for NIH dataset
mainly designed for classification task, extendable

alexyhzou
"""
from functools import partial

import timm
import torch
from torch import nn
from timm.models.vision_transformer import PatchEmbed, Block


class MyViTClassifier(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, **kwargs):
        default_args = {
            'patch_size': 16,
            'embed_dim': 768,
            'depth': 12,
            'num_heads': 12,
            'mlp_ratio': 4,
            'qkv_bias': True,
            'norm_layer': partial(nn.LayerNorm, eps=1e-6)
        }
        default_args.update(kwargs)
        super(MyViTClassifier, self).__init__(**default_args)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = default_args['norm_layer']
            embed_dim = default_args['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome
