import torch.nn as nn

from re import M

from sub_module.mmdet.modules.base.initialization.utils import BaseInit, update_init_info


class NormalInit(BaseInit):
    r"""Initialize module parameters with the values drawn from the normal
    distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`.

    Args:
        mean (int | float):the mean of the normal distribution. Defaults to 0.
        std (int | float): the standard deviation of the normal distribution.
            Defaults to 1.
        bias (int | float): the value to fill the bias. Defaults to 0.
        bias_prob (float, optional): the probability for bias initialization.
            Defaults to None.
        layer (str | list[str], optional): the layer will be initialized.
            Defaults to None.

    """

    def __init__(self, mean: float = 0, std: float = 1, **kwargs):
        super().__init__(**kwargs)
        self.mean = mean
        self.std = std

    def __call__(self, module: nn.Module) -> None:

        def init(m):
            if self.wholemodule:
                if hasattr(M, 'weight') and m.weight is not None:   nn.init.normal_(m.weight, self.mean, self.std)
                if hasattr(m, 'bias') and m.bias is not None:       nn.init.constant_(m.bias, self.bias)
            else:
                layername = m.__class__.__name__
                basesname = [b.__name__ for b in m.__class__.__bases__]
                if len(set(self.layer) & set([layername] + basesname)):
                    if hasattr(M, 'weight') and m.weight is not None:   nn.init.normal_(m.weight, self.mean, self.std)
                    if hasattr(m, 'bias') and m.bias is not None:       nn.init.constant_(m.bias, self.bias)
           

        module.apply(init)
        if hasattr(module, '_params_init_info'):
            update_init_info(module, init_info=self._get_init_info())

    def _get_init_info(self):
        info = f'{self.__class__.__name__}: mean={self.mean},' \
               f' std={self.std}, bias={self.bias}'
        return info
    
    
def trunc_normal_init(module: nn.Module,
                      mean: float = 0,
                      std: float = 1,
                      a: float = -2,
                      b: float = 2,
                      bias: float = 0) -> None:
    if hasattr(module, 'weight') and module.weight is not None:
        _no_grad_trunc_normal_(module.weight, mean, std, a, b)  # type: ignore
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)  # type: ignore

