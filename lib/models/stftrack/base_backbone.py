from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import resize_pos_embed
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from lib.models.layers.patch_embed import PatchEmbed
from lib.models.stftrack.utils import combine_tokens, recover_tokens
from lib.models.stftrack.CSAM_modules import SemanticAttentionModule, PositionalAttentionModule, SliceAttentionModule, CSAM

class BaseBackbone(nn.Module):
    def __init__(self):
        super().__init__()

        self.search_shape = None
        self.template_shape = None
        self.pos_embed = None
        self.img_size = [224, 224]
        self.patch_size = 16
        self.embed_dim = 512  # change 384

        self.cat_mode = 'direct'

        # 添加残差缩放参数
        # self.res_scale = nn.Parameter(torch.tensor(0.3))
        self.cs_position = PositionalAttentionModule()
        self.cs_semantic = SemanticAttentionModule(self.embed_dim)
        self.cs_sliceatt = SliceAttentionModule(self.embed_dim)
        self.CSAM = CSAM(num_slices=self.patch_size, num_channels=self.embed_dim)
        self.pos_embed_z = None
        self.pos_embed_x = None

        self.template_segment_pos_embed = None
        self.search_segment_pos_embed = None

        self.return_inter = False
        self.return_stage = [2, 5, 8, 11]

        self.add_cls_token = False
        self.add_sep_seg = False

    def finetune_track(self, cfg, patch_start_index=1):

        search_size = to_2tuple(cfg.DATA.SEARCH.SIZE)
        template_size = to_2tuple(cfg.DATA.TEMPLATE.SIZE)
        new_patch_size = cfg.MODEL.BACKBONE.STRIDE

        self.cat_mode = cfg.MODEL.BACKBONE.CAT_MODE
        self.return_inter = cfg.MODEL.RETURN_INTER

        patch_pos_embed = self.absolute_pos_embed
        patch_pos_embed = patch_pos_embed.transpose(1, 2)
        B, E, Q = patch_pos_embed.shape
        P_H, P_W = self.img_size[0] // self.patch_size, self.img_size[1] // self.patch_size
        patch_pos_embed = patch_pos_embed.view(B, E, P_H, P_W)

        # for search region
        H, W = search_size
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        search_patch_pos_embed = nn.functional.interpolate(patch_pos_embed, size=(new_P_H, new_P_W), mode='bicubic',
                                                           align_corners=False)
        search_patch_pos_embed = search_patch_pos_embed.flatten(2).transpose(1, 2)

        # for template region
        H, W = template_size
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        template_patch_pos_embed = nn.functional.interpolate(patch_pos_embed, size=(new_P_H, new_P_W), mode='bicubic',
                                                             align_corners=False)
        template_patch_pos_embed = template_patch_pos_embed.flatten(2).transpose(1, 2)

        # self.search_shape = (new_P_H, new_P_W)
        self.template_shape = (
            template_size[0] // new_patch_size,  # 128//16=8
            template_size[1] // new_patch_size
        )
        self.search_shape = (
            search_size[0] // new_patch_size,  # 256//16=16
            search_size[1] // new_patch_size
        )
        self.pos_embed_z = nn.Parameter(template_patch_pos_embed)
        self.pos_embed_x = nn.Parameter(search_patch_pos_embed)

        if self.return_inter:
            for i_layer in self.fpn_stage:
                if i_layer != 11:
                    norm_layer = partial(nn.LayerNorm, eps=1e-6)
                    layer = norm_layer(self.embed_dim)
                    layer_name = f'norm{i_layer}'
                    self.add_module(layer_name, layer)

    def forward_features(self, z, x, mask=None):
        B = x.shape[0]

        z = self.patch_embed(z)
        x = self.patch_embed(x)

        for blk in self.blocks[:-self.num_main_blocks]:
            x = blk(x)
            z = blk(z)

        x = x[..., 0, 0, :]
        z = z[..., 0, 0, :]

        B_z, L_z, C_z = z.shape
        z_4d = z.view(B_z, self.template_shape[0], self.template_shape[1], C_z)
        z_4d = z_4d.permute(0, 3, 1, 2)  # [B, C, H, W]
        z_4d = self.CSAM(z_4d)
        z = z_4d.permute(0, 2, 3, 1).view(B_z, -1, C_z)
        B_x, L_x, C_x = x.shape
        x_4d = x.view(B_x, self.search_shape[0], self.search_shape[1], C_x)
        x_4d = x_4d.permute(0, 3, 1, 2)  # [B, C, H, W]
        x_4d = self.CSAM(x_4d)
        x = x_4d.permute(0, 2, 3, 1).view(B_x, -1, C_x)
        z += self.pos_embed_z
        x += self.pos_embed_x
        lens_z = self.pos_embed_z.shape[1]
        lens_x = self.pos_embed_x.shape[1]

        x = combine_tokens(z, x, mode=self.cat_mode)
        # x = combine_tokens(x, z,  mode=self.cat_mode)
        x = self.pos_drop(x)

        for blk in self.blocks[-self.num_main_blocks:]:
            x = blk(x)

        x = recover_tokens(x, lens_z, lens_x, mode=self.cat_mode)

        aux_dict = {"attn": None}
        x = self.norm_(x)

        return x, aux_dict

    def forward(self, z, x, **kwargs):
        """
        Joint feature extraction and relation modeling for the basic HiViT backbone.
        Args:
            z (torch.Tensor): template feature, [B, C, H_z, W_z]
            x (torch.Tensor): search region feature, [B, C, H_x, W_x]

        Returns:
            x (torch.Tensor): merged template and search region feature, [B, L_z+L_x, C]
            attn : None
        """
        x, aux_dict = self.forward_features(z, x, )

        return x, aux_dict
