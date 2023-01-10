import os, os.path as osp
import re

import torch
from torch.optim import Optimizer

from typing import Optional, Union
from collections import OrderedDict



def save_checkpoint(model: torch.nn.Module,
                    filename: str,
                    optimizer: Optional[Optimizer] = None,
                    meta: Optional[dict] = None) -> None:
    """Save checkpoint to file.

    The checkpoint will have 3 fields: ``meta``, ``state_dict`` and
    ``optimizer``. By default ``meta`` will contain version and time info.

    Args:
        model (Module): Module whose params are to be saved.
        filename (str): Checkpoint filename.
        optimizer (:obj:`Optimizer`, optional): Optimizer to be saved.
        meta (dict, optional): Metadata to be saved in checkpoint.
        file_client_args (dict, optional): Arguments to instantiate a
            FileClient. See :class:`mmcv.fileio.FileClient` for details.
            Default: None.
            `New in version 1.3.16.`
    """
    if meta is None:
        meta = {}
    elif not isinstance(meta, dict):
        raise TypeError(f'meta must be a dict or None, but got {type(meta)}')
    
    model = model.module
    
    if hasattr(model, 'CLASSES') and model.CLASSES is not None:
        # save class name to the meta
        meta.update(CLASSES=model.CLASSES)
        
    # create dict that with parameters of model
    checkpoint = {
        'meta': meta,
        'state_dict': weights_to_cpu(get_state_dict(model))
    }

    # save optimizer state dict in the checkpoint
    if isinstance(optimizer, Optimizer):
        checkpoint['optimizer'] = optimizer.state_dict()
    elif isinstance(optimizer, dict):
        checkpoint['optimizer'] = {}
        for name, optim in optimizer.items():
            checkpoint['optimizer'][name] = optim.state_dict()
            
    # save model
    torch.save(checkpoint, filename)
    
 
def weights_to_cpu(state_dict: OrderedDict) -> OrderedDict:
    """Copy a model state_dict to cpu.

    Args:
        state_dict (OrderedDict): Model weights on GPU.

    Returns:
        OrderedDict: Model weights on GPU.
    """
    
    state_dict_cpu = OrderedDict()
    for key, val in state_dict.items():
        state_dict_cpu[key] = val.cpu()
        
    # Keep metadata in state_dict
    state_dict_cpu._metadata = getattr(state_dict, '_metadata', OrderedDict())
    return state_dict_cpu


def get_state_dict(module: torch.nn.Module,
                   destination: Optional[OrderedDict] = None,
                   prefix: str = '',
                   keep_vars: bool = False):
    """Returns a dictionary containing a whole state of the module.

    Both parameters and persistent buffers (e.g. running averages) are
    included. Keys are corresponding parameter and buffer names.

    This method is modified from :meth:`torch.nn.Module.state_dict` to
    recursively check parallel module in case that the model has a complicated
    structure, e.g., nn.Module(nn.Module(DDP)).

    Args:
        module (nn.Module): The module to generate state_dict.
            
        destination (OrderedDict): Returned dict for the state of the
            module.
        prefix (str): Prefix of the key.
        keep_vars (bool): Whether to keep the variable property of the
            parameters. Default: False.

    Returns:
        dict: A dictionary containing a whole state of the module.
    """
    # below is the same as torch.nn.Module.state_dict()
    if destination is None:
        destination = OrderedDict()
        destination._metadata = OrderedDict()  # type: ignore
    destination._metadata[prefix[:-1]] = local_metadata = dict(version=module._version)
    _save_to_state_dict(module, destination, prefix, keep_vars)
    
    for name, child in module._modules.items():
        if child is not None:
            get_state_dict(child, destination, prefix + name + '.', keep_vars=keep_vars)
    
    for hook in module._state_dict_hooks.values():  # None.
        hook_result = hook(module, destination, prefix, local_metadata)
        if hook_result is not None:
            destination = hook_result
    return destination  # type: ignore


def _save_to_state_dict(module: torch.nn.Module, destination: dict,
                        prefix: str, keep_vars: bool):
    """Saves module state to `destination` dictionary.

    This method is modified from :meth:`torch.nn.Module._save_to_state_dict`.

    Args:
        module (nn.Module): The module to generate state_dict.
        destination (dict): A dict where state will be stored.
        prefix (str): The prefix for parameters and buffers used in this
            module.
    """
    for name, param in module._parameters.items():
        if param is not None:
            destination[prefix + name] = param if keep_vars else param.detach()
            
    
    for name, buf in module._buffers.items():
        # remove check of _non_persistent_buffers_set to allow nn.BatchNorm2d
        if buf is not None:
            destination[prefix + name] = buf if keep_vars else buf.detach()
            
            
            
            
def load_from_http(
        filename: str,
        map_location: Optional[str] = None,
        model_dir: Optional[str] = None,
        logger = None) -> Union[dict, OrderedDict]:
    """load checkpoint through HTTP or HTTPS scheme path. In distributed
    setting, this function only download checkpoint at local rank 0.

    Args:
        filename (str): checkpoint file path with modelzoo or
            torchvision prefix
        map_location (str, optional): Same as :func:`torch.load`.
        model_dir (str, optional): directory in which to save the object,
            Default: None

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    from torch.utils.model_zoo import load_url
    checkpoint = load_url(
        filename, model_dir=model_dir, map_location=map_location)
    
    print_ = f'load checkpoint from url. path: {filename}'
    if logger is not None:
        logger.info(print_)
    else:
        print(print_)
        
    return checkpoint


def load_from_local(file_path: str, map_location='cpu', logger = None):
    filename = osp.expanduser(file_path)
    if not osp.isfile(filename):
        raise FileNotFoundError(f'{filename} can not be found.')

    checkpoint = torch.load(filename, map_location=map_location)
    print_ = f'load checkpoint from local. path: {file_path}'
    if logger is not None:
        logger.info(print_)
    else:
        print(print_)

    return checkpoint



prefixes = {'local': load_from_local,
            'http://': load_from_http,
            'https://': load_from_http }

def load_checkpoint(path: str, map_location='cpu', logger = None):
    for p, func in prefixes.items():
        if p == 'local':
            p = osp.basename(os.getcwd())
        if len(re.findall(p, path))==1:
            checkpoint = func(path, map_location= map_location, logger = logger)
  
    return checkpoint 