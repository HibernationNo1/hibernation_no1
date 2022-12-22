
import torch.nn as nn
import torch.utils.checkpoint as cp

from copy import deepcopy

from hibernation_no1.mmdet.modules.base.module import BaseModule, ModuleList
from hibernation_no1.mmdet.modules.swin_transformer.ffn import FFN
from hibernation_no1.mmdet.modules.swin_transformer.shiftwindow_msa import ShiftWindowMSA

class SwinBlockSequence(BaseModule):
    """Implements one stage in Swin Transformer.

    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        depth (int): The number of blocks in this stage.
        window_size (int, optional): The local window scale. Default: 7.
        qkv_bias (bool, optional): enable bias for qkv if True. Default: True.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        drop_rate (float, optional): Dropout rate. Default: 0.
        attn_drop_rate (float, optional): Attention dropout rate. Default: 0.
        drop_path_rate (float | list[float], optional): Stochastic depth
            rate. Default: 0.
        downsample (BaseModule | None, optional): The downsample operation
            module. Default: None.
        norm_cfg (dict, optional): The config dict of normalization.
            Default: dict(type='LN').
        with_cp (bool, optional): Use checkpoint or not. Using checkpoint
            will save some memory while slowing down the training speed.
            Default: False.
        init_cfg (dict | list | None, optional): The init config.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 depth,
                 window_size=7,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 downsample=None,
                 with_cp=False,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        
        if isinstance(drop_path_rate, list):
            drop_path_rates = drop_path_rate
            assert len(drop_path_rates) == depth
        else:
            drop_path_rates = [deepcopy(drop_path_rate) for _ in range(depth)]
        
        # the variable `instance` of `SwinBlockSequence` is not return `self.blocks` 
        # if you using list() rather than ModuleList()
        self.blocks = ModuleList()  
        
        for i in range(depth):
            block = SwinBlock(
                embed_dims=embed_dims,
                num_heads=num_heads,
                feedforward_channels=feedforward_channels,
                window_size=window_size,
                shift=False if i % 2 == 0 else True,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=drop_path_rates[i],
                with_cp=with_cp,
                init_cfg=None)
            self.blocks.append(block)
            
        
        self.downsample = downsample
    
    def forward(self, x, hw_shape):
        for block in self.blocks:
            x = block(x, hw_shape)      # shape is same: [B, H*W, C]
        
        if self.downsample:   
            # x_down: [B, H/2*W/2, 2*C]     -> depends on stride, kernel size
            x_down, down_hw_shape = self.downsample(x, hw_shape)        

            return x_down, down_hw_shape, x, hw_shape
        else:
            return x, hw_shape, x, hw_shape
            
            
        

class SwinBlock(BaseModule):
    """"
    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        window_size (int, optional): The local window scale. Default: 7.
        shift (bool, optional): whether to shift window or not. Default False.
        qkv_bias (bool, optional): enable bias for qkv if True. Default: True.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        drop_rate (float, optional): Dropout rate. Default: 0.
        attn_drop_rate (float, optional): Attention dropout rate. Default: 0.
        drop_path_rate (float, optional): Stochastic depth rate. Default: 0.
        norm_cfg (dict, optional): The config dict of normalization.
            Default: dict(type='LN').
        with_cp (bool, optional): Use checkpoint or not. Using checkpoint
            will save some memory while slowing down the training speed.
            Default: False.
        init_cfg (dict | list | None, optional): The init config.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 window_size=7,
                 shift=False,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 with_cp=False,
                 init_cfg=None):

        super(SwinBlock, self).__init__()
        
        self.init_cfg = init_cfg
        self.with_cp = with_cp
        
        self.norm1 = nn.LayerNorm(embed_dims, eps = 1e-05)
        
        self.attn = ShiftWindowMSA(
            embed_dims=embed_dims,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=window_size // 2 if shift else 0,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=drop_rate,
            drop_prob = drop_path_rate,
            init_cfg=None)

        
        self.norm2 = nn.LayerNorm(embed_dims, eps = 1e-05)
        self.ffn = FFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            num_fcs=2,
            ffn_drop=drop_rate,
            drop_path_rate = drop_path_rate,  
            add_identity=True,
            init_cfg=None)
    
    def forward(self, x, hw_shape):
        
        def _inner_forward(x):           
            identity = x
            x = self.norm1(x)
            x = self.attn(x, hw_shape)          # [B, H*W, C]
                                                # B: batch size,        C: channel
            x = x + identity
            
            identity = x
            x = self.norm2(x)
            x = self.ffn(x, identity=identity)     

            return x        # [B, H*W, C]
        
        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)

        return x            # [B, H*W, C]
           