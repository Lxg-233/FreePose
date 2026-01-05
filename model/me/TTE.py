
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
    def __init__(self, wave='db1', mode='reflect',level=1):
        super().__init__()
        self.level = level
        self.dwt = DWT1D(wave=wave, mode=mode,J=self.level)
        self.idwt = IDWT1D(wave=wave, mode=mode)

    def _pad_if_odd(self, x):
        """¶¯Ì¬Ìî³ä²¢±ê¼ÇÔ­Ê¼³¤¶È"""
        orig_len = x.size(-1)
        if orig_len % 2 != 0:
            x = F.pad(x, (0, 1), mode='reflect')
            return x, orig_len
        return x, None

    def _store_metadata(self, tensor, orig_len):
        """×¢ÈëÔ­Ê¼³¤¶ÈÔªÊý¾Ý"""
        tensor.orig_len = torch.tensor(orig_len, dtype=torch.long)
        return tensor

    def forward(self, x, inverse=False):
        if not inverse:
            # Ç°Ïò·Ö½âÂ·¾¶
            x_padded, orig_len = self._pad_if_odd(x)
            low, high = self.dwt(x_padded)
            high = torch.stack(high, dim=2)  # [B,C,D,T//2]
            # ×¢ÈëÔªÊý¾Ý£¨ÎÞÂÛÊÇ·ñÐèÒªÌî³ä£©
            if orig_len is not None:
                low = self._store_metadata(low, orig_len)
                high = self._store_metadata(high, orig_len)
            return (low, high)
        else:
            low, high = x
            orig_len = getattr(low, 'orig_len', None) or getattr(high, 'orig_len', None)

            high_list = [high[:, :, i] for i in range(high.size(2))]

            rec = self.idwt((low, high_list))

            if orig_len is not None:
                rec = rec[..., :orig_len.item()]  # È·±£×ªÎªPython±êÁ¿
            else:
                current_len = rec.size(-1)
                if current_len % 2 == 0:
                    rec = rec[..., :-1]
            return rec


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

        self.wavelet = DWTProcessor(wave='db1',level=1)
        # self.wavelet = DWTProcessor(wave='sym4', level=1)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=drop)
        self.norm2 = norm_layer(dim)

    def forward(self, x):
        B,T,J,C=x.shape
        x=x.permute(0,2,1,3).reshape(B*J,T,C)

        x_dwt = x.transpose(1, 2)  # (Bn, C, T)
        low, high = self.wavelet(x_dwt)  # ×Ô¶¯´¦ÀíÆæÊý³¤¶È
        # 高频处理改进---多尺度卷积+门控
        high = high.mean(dim=2)  # [B C 1 T/2]---B  C T/2

        # 低频处理
        low_feat = self.low_freq_attn(
            self.low_freq_norm(low.transpose(1, 2))  # (Bn, T', C)
        ).transpose(1, 2)  # (Bn, C, T')  low=+low
       # fft
        low_fft = torch.fft.rfft(low_feat, dim=-1)
        freq_weight = self.freq_adapter(low_fft.abs())
        low_fft = low_fft * freq_weight
        low_feat = torch.fft.irfft(low_fft, n=low_feat.size(-1), dim=-1)


        # 高频处理
        high = self.high_freq_norm(high.transpose(1, 2)).transpose(1, 2)
        high_feats = [conv(high) for conv in self.high_conv]  # 之前的多尺度
        high_fused = self.high_fuse(torch.cat(high_feats, dim=1).transpose(1, 2)).transpose(1, 2)
        gate = self.high_gate(high)
        high_feat = high_fused * gate  # 自适应门控
        # high_feat=high_fused



        reconstructed = self.wavelet(
            (low_feat, high_feat.unsqueeze(2)),
            inverse=True
        ).transpose(1, 2)  # (Bn, T, C)

        x = x + self.drop_path(reconstructed)  # mlp
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        x=x.reshape(B,J,T,C).permute(0,2,1,3)
        return x