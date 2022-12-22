import torch.nn as nn

from hibernation_no1.mmdet.modules.base.module import BaseModule
from hibernation_no1.mmdet.modules.basic import Linear, DropPath

class FFN(BaseModule):
    """Implements feed-forward networks (FFNs) with identity connection.

    Args:
        embed_dims (int): The feature dimension. Same as
            `MultiheadAttention`. Defaults: 256.
        feedforward_channels (int): The hidden dimension of FFNs.
            Defaults: 1024.
        num_fcs (int, optional): The number of fully-connected layers in
            FFNs. Default: 2.
        ffn_drop (float, optional): Probability of an element to be
            zeroed in FFN. Default 0.0.
        add_identity (bool, optional): Whether to add the
            identity connection. Default: `True`.
        dropout_layer (obj:`ConfigDict`): The dropout_layer used
            when adding the shortcut.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    
    def __init__(self,
                 embed_dims=256,
                 feedforward_channels=1024,
                 num_fcs=2,
                 ffn_drop=0.,
                 drop_path_rate=0.0,
                 add_identity=True,
                 init_cfg=None):
        super().__init__(init_cfg)

        assert num_fcs >= 2, 'num_fcs should be no less ' \
            f'than 2. got {num_fcs}.'
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs
        self.activate = nn.GELU()       # or nn.ReLU(inplace=True)
        
        
        
        layers = []
        in_channels = embed_dims
        for _ in range(num_fcs - 1):
            layers.append( 
                          nn.Sequential(
                                        Linear(in_channels, feedforward_channels),
                                        self.activate,
                                        nn.Dropout(ffn_drop)
                                        )
                        )
            in_channels = feedforward_channels
        layers.append(Linear(feedforward_channels, embed_dims))
        layers.append(nn.Dropout(ffn_drop))
        self.layers = nn.Sequential(*layers)
        self.dropout_layer = DropPath(drop_path_rate)
        self.add_identity = add_identity
    
    def forward(self, x, identity=None):
        """Forward function for `FFN`.

        The function would add x to the output tensor if residue is None.
        """
        # [B, H*W, C]
        # Batch size,  Height, Width, Channel
        out = self.layers(x)     
    
        if not self.add_identity:
            return self.dropout_layer(out)
        
        if identity is None:
            identity = x
            
        out = identity + self.dropout_layer(out)
        return out