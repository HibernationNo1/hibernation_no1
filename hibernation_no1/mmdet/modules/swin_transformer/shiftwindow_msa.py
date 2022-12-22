import torch
import torch.nn as nn
import torch.nn.functional as F

from hibernation_no1.mmdet.utils import to_2tuple

from hibernation_no1.mmdet.modules.basic import DropPath
from hibernation_no1.mmdet.modules.base.module import BaseModule
from hibernation_no1.mmdet.modules.base.initialization.utils import _no_grad_trunc_normal_

class ShiftWindowMSA(BaseModule):
    """Shifted Window Multihead Self-Attention Module.

    Args:
        embed_dims (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): The height and width of the window.
        shift_size (int, optional): The shift step of each window towards
            right-bottom. If zero, act as regular window-msa. Defaults to 0.
        qkv_bias (bool, optional): If True, add a learnable bias to q, k, v.
            Default: True
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Defaults: None.
        attn_drop_rate (float, optional): Dropout ratio of attention weight.
            Defaults: 0.
        proj_drop_rate (float, optional): Dropout ratio of output.
            Defaults: 0.
        dropout_layer (dict, optional): The dropout_layer used before output.
            Defaults: dict(type='DropPath', drop_prob=0.).
        init_cfg (dict, optional): The extra config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 window_size,
                 shift_size=0,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop_rate=0,
                 proj_drop_rate=0,
                 drop_prob=0.,
                 init_cfg=None):
        super().__init__(init_cfg)
        
        self.window_size = window_size  # TODO_katib, training after fix `window_size`
        self.shift_size = shift_size
        assert 0 <= self.shift_size < self.window_size

        self.w_msa = WindowMSA(
            embed_dims=embed_dims,
            num_heads=num_heads,
            window_size=to_2tuple(window_size),
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=proj_drop_rate,
            init_cfg=None)

        self.drop = DropPath(drop_prob)
        
    def forward(self, query, hw_shape):
        B, L, C = query.shape       # B: batch_size,    C: channel
        H, W = hw_shape
        assert L == H * W, 'input feature has wrong size'
        query = query.view(B, H, W, C)      # [batch_size, H*W, C] -> [batch_size, H, W, C]
   
        # pad feature maps to multiples of window size
        pad_r = (self.window_size - W % self.window_size) % self.window_size        # 0 (pad to width-right)
        pad_b = (self.window_size - H % self.window_size) % self.window_size        # 4 (pad to height-bottom)
        query = F.pad(query, (0, 0, 0, pad_r, 0, pad_b))                            # [batch_size, 196, 336, 96]  
        # padding last dimension to (0, 0),   padding 1 dimension to (0, pad_r),     padding 2 dimension to  (0, pad_b)
        H_pad, W_pad = query.shape[1], query.shape[2]       
   
        # cyclic shift
        if self.shift_size > 0:   
            shifted_query = torch.roll(query, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            
            # calculate attention mask for SW-MSA
            img_mask = torch.zeros((1, H_pad, W_pad, 1), device=query.device)
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size,
                              -self.shift_size), slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size,
                              -self.shift_size), slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            # nW, window_size, window_size, 1
            mask_windows = self.window_partition(img_mask)
            mask_windows = mask_windows.view(
                -1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0,
                                              float(-100.0)).masked_fill(
                                                  attn_mask == 0, float(0.0))
        else:
            shifted_query = query
            attn_mask = None
            
        # [2688, window_size, window_size, 96], nW*B(number of windows*batch size)=2688,     channel= 96
        query_windows = self.window_partition(shifted_query)            
        # [nW*B, Wh*Ww, channel]
        query_windows = query_windows.view(-1, self.window_size**2, C)  
        
        # W-MSA/SW-MSA
        attn_windows = self.w_msa(query_windows, mask=attn_mask)                        # [nW*B, Wh*Ww, channel] 
        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)     # [nW*B, window_size, window_size, channel]  
  
        # B H' W' C
        shifted_x = self.window_reverse(attn_windows, H_pad, W_pad) 
        
        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(
                shifted_x,
                shifts=(self.shift_size, self.shift_size),
                dims=(1, 2))
        else:
            x = shifted_x           # [B, H, W, C]
        
        if pad_r > 0 or pad_b:
            x = x[:, :H, :W, :].contiguous()
        
        x = x.view(B, H * W, C)

        x = self.drop(x)
        return x                    # [B, H*W, C]


    def window_reverse(self, windows, H, W):
        """
        Args:
            windows: (num_windows*B, window_size, window_size, C), C: channel
            H (int): Height of image
            W (int): Width of image
        Returns:
            x: (B, H, W, C)
        """
        window_size = self.window_size
        B = int(windows.shape[0] / (H * W / window_size / window_size))     # batch size
        x = windows.view(B,                                                 # [B, H_num_windows, W_num_windows, window_size, window_size, C]
                         H // window_size, W // window_size, 
                         window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)              # [B, H, W, C]
        return x     
        
    def window_partition(self, x):
        """
        Args:
            x: (B, H, W, C)
        Returns:
            windows: (num_windows*B, window_size, window_size, C)
        """
        B, H, W, C = x.shape        # [batch_size, 196, 336, 96]
        window_size = self.window_size
        x = x.view(B, H // window_size, window_size,    # [batch_size, 28, 7, 48, 7, 96], window_size= 7    
                   W // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()          # [batch_size, 28, 48, 7, 7, 96]        
        windows = windows.view(-1, window_size, window_size, C)     # [2688, 7, 7, 96]      # number of windows * batch size = 2688
        return windows
    
    
    
    
class WindowMSA(BaseModule):
    """Window based multi-head self-attention (W-MSA) module with relative
    position bias.

    Args:
        embed_dims (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): The height and width of the window.
        qkv_bias (bool, optional):  If True, add a learnable bias to q, k, v.
            Default: True.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        attn_drop_rate (float, optional): Dropout ratio of attention weight.
            Default: 0.0
        proj_drop_rate (float, optional): Dropout ratio of output. Default: 0.
        init_cfg (dict | None, optional): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 window_size,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop_rate=0.,
                 proj_drop_rate=0.,
                 init_cfg=None):

        super().__init__()
        self.embed_dims = embed_dims
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads     
        head_embed_dims = embed_dims // num_heads
        self.scale = qk_scale or head_embed_dims**-0.5
        self.init_cfg = init_cfg

        # define a parameter table of relative position bias
        # 2*Wh-1 * 2*Ww-1, nH
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  
        
        # About 2x faster than original impl
        Wh, Ww = self.window_size
        rel_index_coords = self.double_step_seq(2 * Ww - 1, Wh, 1, Ww)
        rel_position_index = rel_index_coords + rel_index_coords.T
        rel_position_index = rel_position_index.flip(1).contiguous()
        self.register_buffer('relative_position_index', rel_position_index)

        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop_rate)

        self.softmax = nn.Softmax(dim=-1) 
        
    
    @staticmethod
    def double_step_seq(step1, len1, step2, len2):
        seq1 = torch.arange(0, step1 * len1, step1)
        seq2 = torch.arange(0, step2 * len2, step2)
        return (seq1[:, None] + seq2[None, :]).reshape(1, -1)


    def init_weights(self):
        _no_grad_trunc_normal_(self.relative_position_bias_table, 
                               mean = 0.,
                               std=0.02,
                               a = -2.,
                               b = 2.)
    
    def forward(self, x, mask=None):
        """
        Args:

            x (tensor): input features with shape of (num_windows*B, N, C)
            mask (tensor | None, Optional): mask with shape of (num_windows,
                Wh*Ww, Wh*Ww), value should be between (-inf, 0].
        """
        B, N, C = x.shape       # [nW*B, Wh*Ww, channel]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)     # [nW*B, Wh*Ww, 3, self.num_heads, channel//self.num_heads]
        qkv = qkv.permute(2, 0, 3, 1, 4)                                            # [3, nW*B, nH, W^2, C//nH]
        
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))        # [nW*B, nH, Wh*Ww, Wh*Ww]
        
      
        relative_position_bias = self.relative_position_bias_table[                    # [Wh*Ww,Wh*Ww,nH]
            self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1],
                -1)  
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # [nH, Wh*Ww, Wh*Ww]
        attn = attn + relative_position_bias.unsqueeze(0)                              # [nW*B, nH, Wh*Ww, Wh*Ww] 
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B // nW, nW, self.num_heads, N,
                             N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)       # activate func

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)     # [nW*B, Wh*Ww, channel] 
        x = self.proj(x)
        x = self.proj_drop(x)

        return x    # [nW*B, Wh*Ww, channel] 