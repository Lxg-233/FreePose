
import sys
import math
import torch
import torch.nn as nn
from functools import partial
from timm.models.layers import DropPath
from einops import rearrange, repeat
import torch.nn.functional as F
from pytorch_wavelets import DWT1D, IDWT1D


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DWTProcessor(nn.Module):
    def __init__(self, wave='db1', mode='reflect', level=1):
        super().__init__()
        self.level = level
        self.dwt = DWT1D(wave=wave, mode=mode, J=1)
        self.idwt = IDWT1D(wave=wave, mode=mode)
        self.orig_lens = []  # 记录每级分解的原始长度

    def _pad_if_odd(self, x):
        orig_len = x.size(-1)
        if orig_len % 2 != 0:
            x = F.pad(x, (0, 1), mode='reflect')
            return x, orig_len
        return x, None

    def forward(self, x, inverse=False):
        if not inverse:
            self.orig_lens = []  # 清空原始长度记录
            current = x
            highs = []

            for _ in range(self.level):
                # 动态填充奇数长度
                current, orig_len = self._pad_if_odd(current)
                self.orig_lens.append(orig_len)  # 记录原始长度

                # 执行小波分解
                current, high = self.dwt(current)
                highs.append(high[0])

            return (current, highs)
        else:
            low, highs = x
            for i, h in enumerate(reversed(highs)):
                # 计算目标长度
                target_len = low.size(-1)

                # 动态调整高频长度
                if h.size(-1) != target_len:
                    h = F.interpolate(h, size=target_len, mode='nearest')

                # 执行逆变换
                low = self.idwt((low, [h]))

                # 裁剪该级分解的原始长度
                if i < len(self.orig_lens):
                    orig_len = self.orig_lens[-(i + 1)]  # 逆序获取原始长度
                    if orig_len is not None:
                        low = low[..., :orig_len]
            return low


class SmoothAttention(nn.Module):
    """´øÓÐÊ±ÓòË«ÏòÆ½»¬Ô¼ÊøµÄ×¢ÒâÁ¦»úÖÆ"""

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., window_size=5):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)


    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x).reshape(B, T, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.key(x).reshape(B, T, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.value(x).reshape(B, T, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, T, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x






class EnhancedTTEBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_hidden_dim,
                 qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.low_freq_norm = norm_layer(dim)
        # self.low_freq_attn = Attention(
        #     dim, num_heads=num_heads, qkv_bias=qkv_bias,
        #     qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # self.low_freq_attn = CrossFrequencyAttention(
        #     dim, num_heads=num_heads, qkv_bias=qkv_bias,
        #     qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.low_freq_attn=SmoothAttention(dim, num_heads=num_heads,
                                           qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.freq_filter = nn.Parameter(torch.linspace(1,0,dim//2))  #
        # 滤波
        self.freq_adapter = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # [B,C,F] -> [B,C,1]
            nn.Conv1d(dim, dim//8, 1),
            nn.ReLU(),
            nn.Conv1d(dim//8, dim, 1),  # Éú³ÉÂË²¨Æ÷²ÎÊý
            nn.Sigmoid()
        )

        # 新增门控机制
        self.high_gate = nn.Sequential(
            nn.Conv1d(dim, dim, 3, padding=1),
            nn.Sigmoid()
        )
        self.high_freq_norm = norm_layer(dim)
        # 多尺度卷积
        self.high_conv = nn.ModuleList([
            nn.Conv1d(dim, dim, k, padding=k // 2, groups=dim)
            for k in [3, 5, 7]
        ])
        self.high_fuse = nn.Linear(3 * dim, dim)

        self.wavelet = DWTProcessor(wave='db1',level=2)
        # self.wavelet = DWTProcessor(wave='sym4', level=1)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=drop)
        self.norm2 = norm_layer(dim)

    def forward(self, x):
        B, T, J, C = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B * J, T, C)

        # 小波分解
        x_dwt = x.transpose(1, 2)
        low, highs = self.wavelet(x_dwt)

        # 低频处理
        low_feat = self.low_freq_attn(
            self.low_freq_norm(low.transpose(1, 2))  # (Bn, T', C)
        ).transpose(1, 2)  # (Bn, C, T')  low=+low

        # # 高频处理    # 高频两个conv
        # high_feats = []
        # for i, h in enumerate(highs):
        #     # 多尺度卷积融合
        #     h_feats = [conv(h) for conv in self.high_conv]
        #     h_fused = self.high_fuse(torch.cat(h_feats, dim=1).transpose(1, 2)).transpose(1, 2)
        #
        #     # 层级自适应门控
        #     gate = self.high_gate(h)
        #     high_feats.append(h_fused * gate)
        #
        # # 小波重构
        # reconstructed = self.wavelet(
        #     (low_feat, high_feats),
        #     inverse=True
        # ).transpose(1, 2)


        #  高频attn+conv
        high_feats = []
        for i, h in enumerate(highs):
            if i == 0:
                h_attn = self.low_freq_attn(self.low_freq_norm(h.transpose(1, 2))).transpose(1, 2)
                high_feats.append(h_attn)
            else:
                h_feats = [conv(h) for conv in self.high_conv]
                h_fused = self.high_fuse(torch.cat(h_feats, dim=1).transpose(1, 2)).transpose(1, 2)
                gate = self.high_gate(h)
                high_feats.append(h_fused * gate)

        reconstructed = self.wavelet((low_feat, high_feats), inverse=True).transpose(1, 2)



        # 残差连接
        x = x + self.drop_path(reconstructed)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x.reshape(B, J, T, C).permute(0, 2, 1, 3)
