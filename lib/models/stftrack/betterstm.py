import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = F.adaptive_avg_pool2d(x, 1).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class DeformableAttention(nn.Module):

    def __init__(self, embed_dim=512, num_heads=8, offset_range=0.5):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.offset_conv = nn.Conv2d(embed_dim, 2 * num_heads, kernel_size=3, padding=1)
        nn.init.constant_(self.offset_conv.weight, 0)
        nn.init.constant_(self.offset_conv.bias, 0)

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        self.offset_range = offset_range

    def forward(self, query, key, value):
        Lq, B, C = query.shape
        Lk = key.shape[0]

        query_img = query.permute(1, 2, 0).view(B, C, int(Lq ** 0.5), int(Lq ** 0.5))
        offsets = self.offset_conv(query_img)  # (B, 2*H, Hq, Wq)
        offsets = torch.tanh(offsets) * self.offset_range

        query = self.q_proj(query).view(Lq, B, self.num_heads, self.head_dim)
        key = self.k_proj(key).view(Lk, B, self.num_heads, self.head_dim)
        value = self.v_proj(value).view(Lk, B, self.num_heads, self.head_dim)

        key = self.deform_sample(key, offsets, (Lq, B))
        value = self.deform_sample(value, offsets, (Lq, B))

        attn_weights = torch.einsum('qbhc,kbhc->bhqk', query, key) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_weights, dim=-1)

        output = torch.einsum('bhqk,kbhc->qbhc', attn_weights, value)
        return output.contiguous().view(Lq, B, C)

    def deform_sample(self, x, offsets, target_shape):
        H, W = int(target_shape[0] ** 0.5), int(target_shape[0] ** 0.5)
        B, _, H_offset, W_offset = offsets.shape

        grid = self._get_grid(H, W, x.device).repeat(B, 1, 1, 1)
        grid = grid + offsets.permute(0, 2, 3, 1).view(B, H_offset, W_offset, self.num_heads, 2)
        grid = grid.view(B * self.num_heads, H, W, 2)

        x = x.permute(1, 2, 0).view(B, self.num_heads, self.head_dim, H, W)
        x = x.repeat_interleave(H * W, 0).view(B * self.num_heads * H * W, self.head_dim, H, W)
        sampled = F.grid_sample(x, grid, align_corners=False)
        return sampled.view(B, self.num_heads, H * W, self.head_dim).permute(2, 0, 1, 3)

    def _get_grid(self, H, W, device):
        y, x = torch.meshgrid(torch.linspace(-1, 1, H, device=device),
                              torch.linspace(-1, 1, W, device=device))
        return torch.stack((x, y), -1).unsqueeze(0)  # (1, H, W, 2)


class MultiScaleFeaturePyramid(nn.Module):

    def __init__(self, in_channels, scales=[0.5, 1.0, 2.0]):
        super().__init__()
        self.scales = scales
        self.convs = nn.ModuleList()

        for s in scales:
            layers = []
            if s < 1.0:
                layers.append(nn.AvgPool2d(kernel_size=int(1 / s)))
            elif s > 1.0:
                layers.append(nn.Upsample(scale_factor=s, mode='bilinear'))
            layers += [
                nn.Conv2d(in_channels, in_channels, 3, padding=1),
                nn.GroupNorm(8, in_channels),
                nn.ReLU(inplace=True)
            ]
            self.convs.append(nn.Sequential(*layers))

    def forward(self, x):
        features = []
        for conv, s in zip(self.convs, self.scales):
            if s != 1.0:
                resized = F.interpolate(x, scale_factor=s, mode='bilinear')
                features.append(conv(resized))
            else:
                features.append(conv(x))
        return features  # 返回多尺度特征列表


class MultiScaleCrossAttention(nn.Module):

    def __init__(self, embed_dim=512, num_heads=8, scales=[0.5, 1.0, 2.0]):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.enc_pyramid = MultiScaleFeaturePyramid(embed_dim, scales)
        self.dec_pyramid = MultiScaleFeaturePyramid(embed_dim, scales)

        self.attentions = nn.ModuleList([
            DeformableAttention(embed_dim, num_heads)
            for _ in scales
        ])

        self.fusion = nn.Sequential(
            nn.Conv2d(embed_dim * len(scales), embed_dim, 1),
            ChannelAttention(embed_dim),
            nn.Dropout(0.1)
        )

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, enc_feat, dec_feat):

        enc_scales = self.enc_pyramid(enc_feat)  # List[(B,C,Hs,Ws)]
        dec_scales = self.dec_pyramid(dec_feat.permute(0, 2, 1).view_as(enc_feat))

        dec_scales = [d.view(B, self.embed_dim, -1).permute(2, 0, 1)
                      for d, B in zip(dec_scales, enc_feat.size(0))]

        all_attn = []
        for enc, dec, attn in zip(enc_scales, dec_scales, self.attentions):

            B, C, H, W = enc.size()
            enc = enc.view(B, C, -1).permute(2, 0, 1)  # (Lk, B, C)

            scale_attn = attn(dec, enc, enc)  # (Lq, B, C)
            all_attn.append(scale_attn.permute(1, 2, 0).view(B, C, H, W))

        fused = self.fusion(torch.cat(all_attn, dim=1))  # (B, C, H, W)

        output = self.norm((dec_feat + fused.flatten(2).permute(0, 2, 1)))
        return output

class CrossSTMWithMSCA(nn.Module):

    def __init__(self,
                 num_classes=2,
                 hidden_dim=512,
                 num_layers=3,
                 scales=[0.5, 1.0, 2.0]):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            *[ResBlock(64) for _ in range(2)]
        )

        self.memory = nn.LSTM(64 * 56 * 56, hidden_dim, num_layers=1)

        self.decoder_layers = nn.ModuleList([
            MultiScaleCrossAttention(hidden_dim, scales=scales)
            for _ in range(num_layers)
        ])

        self.head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(hidden_dim // 2, num_classes, 1)
        )

    def forward(self, frame, prev_memory):
        enc_feat = self.encoder(frame)  # (B, 64, H//2, W//2)

        mem_input = enc_feat.flatten(1).unsqueeze(1)  # (B, 1, 64*H*W)
        _, (h_n, c_n) = self.memory(mem_input, prev_memory)
        mem_feat = h_n[-1]  # (B, C)

        dec_feat = mem_feat.unsqueeze(1)  # (B, 1, C)
        for layer in self.decoder_layers:
            dec_feat = layer(enc_feat, dec_feat)

        mask = self.head(dec_feat.permute(0, 2, 1).view_as(enc_feat))
        return F.interpolate(mask, size=frame.shape[-2:]), torch.cat([prev_memory, mem_feat], 1)


class ResBlock(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return x + self.conv(x)
