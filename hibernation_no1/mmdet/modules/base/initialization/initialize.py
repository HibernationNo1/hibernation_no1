import copy

import torch
import torch.nn as nn

from typing import List, Union, Dict

from hibernation_no1.mmdet.modules.base.initialization.normal import NormalInit
from hibernation_no1.mmdet.modules.base.initialization.xavier import XavierInit   

def initialize(module: nn.Module, init_cfg: Union[Dict, List[dict]]):
    """Initialize a module.

    Args:
        module (``torch.nn.Module``): the module will be initialized.
        init_cfg (dict | list[dict]): initialization configuration dict to
            define initializer. OpenMMLab has implemented 6 initializers
            including ``Constant``, ``Xavier``, ``Normal``, ``Uniform``,
            ``Kaiming``, and ``Pretrained``.

    Example:
            # initialization for only 1 layer
        >>> module = nn.Linear(2, 3, bias=True)
        >>> init_cfg = dict(type='Constant', layer='Linear', val =1 , bias =2)
        >>> initialize(module, init_cfg)
        
            # initialization for n(>1) layers
        >>> module = nn.Sequential(nn.Conv1d(3, 1, 3), nn.Linear(1,2))
        >>> # define key ``'layer'`` for initializing layer with different
        >>> # configuration
        >>> init_cfg = [dict(type='Constant', layer='Conv1d', val=1),
                dict(type='Constant', layer='Linear', val=2)]
        >>> initialize(module, init_cfg)

            
           # define key``'override'`` to initialize some specific part in module
        >>> class FooNet(nn.Module):
        >>>     def __init__(self):
        >>>         super().__init__()
        >>>         self.feat = nn.Conv2d(3, 16, 3)
        >>>         self.reg = nn.Conv2d(16, 10, 3)
        >>>         self.cls = nn.Conv2d(16, 5, 3)
        >>> model = FooNet()
        >>> init_cfg = dict(type='Constant', val=1, bias=2, layer='Conv2d',
        >>>                 override=dict(type='Constant', name='reg', val=3, bias=4))  
        >>> initialize(model, init_cfg)

        >>> model = ResNet(depth=50)
        >>> # Initialize weights with the pretrained model.
        >>> init_cfg = dict(type='Pretrained',
                checkpoint='torchvision://resnet50')
        >>> initialize(model, init_cfg)

        >>> # Initialize weights of a sub-module with the specific part of
        >>> # a pretrained model by using "prefix".
        >>> url = 'http://download.openmmlab.com/mmdetection/v2.0/retinanet/'\
        >>>     'retinanet_r50_fpn_1x_coco/'\
        >>>     'retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth'
        >>> init_cfg = dict(type='Pretrained',
                checkpoint=url, prefix='backbone.')
    """
    if not isinstance(init_cfg, (dict, list)):
        raise TypeError(f'init_cfg must be a dict or a list of dict, \
                but got {type(init_cfg)}')

    if isinstance(init_cfg, dict):
        init_cfg = [init_cfg]

    for cfg in init_cfg:
        cp_cfg = copy.deepcopy(cfg)
        override = cp_cfg.pop('override', None)
        
        _initialize(module, cp_cfg)
        
        if override is not None:
            cp_cfg.pop('layer', None)
            _initialize_override(module, override, cp_cfg)
        else:
            # All attributes in module have same initialization.
            pass
        
        
    
def _initialize(module: nn.Module, cfg: Dict, wholemodule: bool = False):
    # execute initialization with specific type of initialization 
    # TODO : add initialization algorithm and manege with registry
    # TODO : add `cfg.type == Pretrained`
    
    
    cp_cfg = copy.deepcopy(cfg)
    init_type = cp_cfg.pop('type')
    if init_type =='Normal':
        func = NormalInit(**cp_cfg)
    elif init_type =='Xavier':
        func = XavierInit(**cp_cfg)
    
    
    # wholemodule : using for override
    func.wholemodule = wholemodule
    func(module)
    
    

def _initialize_override(module: nn.Module, override: Union[Dict, List],
                         cfg: Dict) -> None:
    if not isinstance(override, (dict, list)):
        raise TypeError(f'override must be a dict or a list of dict, \
                but got {type(override)}')

    override = [override] if isinstance(override, dict) else override

    for override_ in override:

        cp_override = copy.deepcopy(override_)
        name = cp_override.pop('name', None)
        if name is None:
            raise ValueError('`override` must contain the key "name",'
                             f'but got {cp_override}')
        # if override only has name key, it means use args in init_cfg
        if not cp_override:
            cp_override.update(cfg)
        # if override has name key and other args except type key, it will
        # raise error
        elif 'type' not in cp_override.keys():
            raise ValueError(
                f'`override` need "type" key, but got {cp_override}')
        if hasattr(module, name):
            _initialize(getattr(module, name), cp_override, wholemodule=True)
        else:
            raise RuntimeError(f'module did not have attribute {name}, '
                               f'but init_cfg is {cp_override}.')
