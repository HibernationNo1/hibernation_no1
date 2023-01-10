import copy
import warnings
import torch
import torch.nn as nn

from logging import FileHandler
from typing import Optional, Iterable
from abc import ABCMeta
from collections import defaultdict

from docker.hibernation_no1.utils.log import get_logger
from docker.hibernation_no1.mmdet.modules.base.initialization.initialize import initialize

class BaseModule(nn.Module, metaclass=ABCMeta):
    """Base module for all modules in openmmlab.
        execute parameter initialization for each layer
        
    ``BaseModule`` is a wrapper of ``torch.nn.Module`` with additional
    functionality of parameter initialization. 
    Compared with ``torch.nn.Module``, ``BaseModule`` mainly adds three attributes.

    - ``init_cfg``: the config to control the initialization.
    - ``init_weights``: The function of parameter initialization and recording
      initialization information.
    - ``_params_init_info``: Used to track the parameter initialization
      information. This attribute only exists during executing the
      ``init_weights``.

    Args:
        init_cfg (dict, optional): Initialization config dict.
    """

    def __init__(self, init_cfg: Optional[dict] = None):
        """Initialize BaseModule, inherited from `torch.nn.Module`"""
        super().__init__()
        self._is_init = False
        self.init_cfg = copy.deepcopy(init_cfg)

    @property
    def is_init(self):
        return self._is_init

    def init_weights(self):    
        """Initialize the weights."""

        is_top_level_module = False
        # check if it is top-level module
        if not hasattr(self, '_params_init_info'):
            # dict to recored initialization information
            self._params_init_info = defaultdict(dict)
            is_top_level_module = True
            
            for name, param in self.named_parameters():
                # init_info : describes the initialization.
                self._params_init_info[param]['init_info'] = \
                                   f'The value is the same before and ' \
                                   f'after calling `init_weights` ' \
                                   f'of {self.__class__.__name__} '
                # tmp_mean_value : average of value indicating whethere parameter has been modified
                #   if `tmp_mean_value` has been modified, updata initialization information
                self._params_init_info[param]['tmp_mean_value'] = param.data.mean()
          
            # pass `params_init_info` to all submodules
            # All submodules share the same `params_init_info`,
            # so it will be updated when parameters are modified at any level of the model.
            for sub_module in self.modules():
                sub_module._params_init_info = self._params_init_info

        logger = get_logger(log_name = "initialization")

        module_name = self.__class__.__name__
        if not self._is_init:
            if self.init_cfg:   # initialize for initialization target layer 
                logger.info(f'initialize {module_name} with init_cfg {self.init_cfg}')
                
                # actually run initizalize
                initialize(self, self.init_cfg)

                if isinstance(self.init_cfg, dict):
                    # prevent the parameters of the pre-trained model from being overwritten by the `init_weights`
                    if self.init_cfg['type'] == 'Pretrained':
                        return

            for module in self.children():
                # update init infomation
                if hasattr(module, 'init_weights'):
                    module.init_weights()       # run init_weights for each module 
                    # users may overload the `init_weights`
                    assert hasattr(module,'_params_init_info'), f'Can not find `_params_init_info` in {module}'
                    
                    init_info=f'Initialized by ' \
                        f'user-defined `init_weights`' \
                        f' in {module.__class__.__name__}'
                        
                    for name, param in module.named_parameters():
                        assert param in module._params_init_info, (
                            f'Find a new :obj:`Parameter` '
                            f'named `{name}` during executing the '
                            f'`init_weights` of '
                            f'`{module.__class__.__name__}`. '
                            f'Please do not add or '
                            f'replace parameters during executing '
                            f'the `init_weights`. ')

                        # The parameter has been changed during executing the
                        # `init_weights` of module
                        mean_value = param.data.mean()
                        if module._params_init_info[param]['tmp_mean_value'] != mean_value:
                            module._params_init_info[param]['init_info'] = init_info
                            module._params_init_info[param]['tmp_mean_value'] = mean_value
            self._is_init = True
        else:
            warnings.warn(f'init_weights of {self.__class__.__name__} has '
                          f'been called more than once.')
            
        if is_top_level_module:
            self._dump_init_info()

            for sub_module in self.modules():
                del sub_module._params_init_info

    # only run in single gpu
    def _dump_init_info(self):
        """Dump the initialization information to a file named
        `initialization.log.json` in workdir.
        """

        logger = get_logger(name = "initialization")

        with_file_handler = False
        # dump the information to the logger file if there is a `FileHandler`
        for handler in logger.handlers:
            if isinstance(handler, FileHandler):
                handler.stream.write(f"{'-'*60}\n"\
                    'Name of parameter - Initialization information\n')
                for name, param in self.named_parameters():
                    handler.stream.write(
                        f'\n{name} - {param.shape}: '
                        f"\n{self._params_init_info[param]['init_info']} \n")
                handler.stream.flush()
                with_file_handler = True
        if not with_file_handler:
            for name, param in self.named_parameters():
                logger.info(f'\n{name} - {param.shape}: '
                            f"\n{self._params_init_info[param]['init_info']} \n ")        

    def __repr__(self):
        s = super().__repr__()
        if self.init_cfg:
            s += f'\ninit_cfg={self.init_cfg}'
        return s 



class ModuleList(BaseModule, nn.ModuleList):
    """ ModuleList in openmmlab.    ###
        layer를 class내에서 정의 후 list()로 감싸면 해당 class의 instance는 layer를 반환하지 않는다.
        하지만 ModuleList를 통해 layer를 감싸면 해당 class의 instance는 layer를 반환하게 되며 
        instance별로 layer list를 관리할 수 있다.
    Args:
        modules (iterable, optional): an iterable of modules to add.
        init_cfg (dict, optional): Initialization config dict.
    """

    def __init__(self,
                 modules: Optional[Iterable] = None,
                 init_cfg: Optional[dict] = None):
        BaseModule.__init__(self, init_cfg)
        nn.ModuleList.__init__(self, modules)