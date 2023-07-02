import torch
import torch.nn as nn

import torch.nn.functional as F
class GELU(nn.Module):
    r"""Applies the Gaussian Error Linear Units function:

    .. math::
        \text{GELU}(x) = x * \Phi(x)
    where :math:`\Phi(x)` is the Cumulative Distribution Function for
    Gaussian Distribution.

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input

    .. image:: scripts/activation_images/GELU.png

    Examples::

        >>> m = nn.GELU()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """

    def forward(self, input):
        return F.gelu(input)

ACTIVATION_LAYERS = dict(ReLU = nn.ReLU(),
                         LeakyReLU = nn.LeakyReLU(),
                         PReLU = nn.PReLU(),
                         RReLU = nn.RReLU(),
                         ReLU6 = nn.ReLU6(),
                         ELU = nn.ELU(),
                         Sigmoid = nn.Sigmoid(),
                         Tanh = nn.Tanh(),
                         GELU = nn.GELU())       

def build_activation_layer(cfg):
    if not isinstance(cfg, dict):
        raise TypeError('cfg must be a dict')
    if 'type' not in cfg:
        raise KeyError('the cfg dict must contain the key "type"')
    cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')

    if layer_type not in ACTIVATION_LAYERS.keys():
        raise KeyError(f'Unrecognized norm type {layer_type}')

    activation_layer = ACTIVATION_LAYERS.get(layer_type)
    return activation_layer