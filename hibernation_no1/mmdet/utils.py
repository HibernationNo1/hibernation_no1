import numpy as np
import torch
import collections.abc
import importlib
import warnings
from itertools import repeat
from getpass import getuser
from socket import gethostname


def to_2tuple(x):
    if isinstance(x, collections.abc.Iterable):
        return x
    return tuple(repeat(x, 2))




def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.

    Args:
        data (torch.Tensor | numpy.ndarray | Sequence | int | float): Data to
            be converted.
    """

    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, collections.abc.Sequence) and not isinstance(data, str):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(f'type {type(data)} cannot be converted to tensor.')
    


def load_ext(name, funcs):
    # TODO: 
    # 1. 해당 package를 pypi에 올린 후 pip install한다.
    #    필. _ext.cp38-win_amd64.pyd 또는 linux용 C소스 module을 포함하여 upload.
    # 2. 아래 "mmcv."을 f"{upload한 module_name}으로 대체" 
    ext = importlib.import_module("mmcv." + name)   
    for fun in funcs:
        assert hasattr(ext, fun), f'{fun} miss in module {name}'
    
    return ext



# TODO: using  
def auto_scale_lr(cfg, logger, num_gpus = 1):   
    """Automatically scaling LR according to GPU number and sample per GPU.

    Args:
        cfg (config): whole config.
        logger (logging.Logger): Logger.
    """
    
    # Get flag from config
    if ('auto_scale_lr' not in cfg) or \
            (not cfg.auto_scale_lr.get('enable', False)):
        logger.info('Automatic scaling of learning rate (LR)'
                    ' has been disabled.')
        return
    
    # Get base batch size from config
    base_batch_size = cfg.auto_scale_lr.get('base_batch_size', None)
    if base_batch_size is None:
        return
    
    batch_size = cfg.data.train_dataloader.samples_per_gpu
    logger.info(f'Training with {num_gpus} GPU(s). The total batch size is {batch_size}.')
    
    if batch_size != base_batch_size:
        # scale LR with
        # [linear scaling rule](https://arxiv.org/abs/1706.02677)
        scaled_lr = (batch_size / base_batch_size) * cfg.optimizer.lr
        logger.info('LR has been automatically scaled '
                    f'from {cfg.optimizer.lr} to {scaled_lr}')
        cfg.optimizer.lr = scaled_lr
    else:
        logger.info('The batch size match the '
                    f'base batch size: {base_batch_size}, '
                    f'will not scaling the LR ({cfg.optimizer.lr}).') 
        
        
def get_host_info():
    """Get hostname and username.

    Return empty string if exception raised, e.g. ``getpass.getuser()`` will
    lead to error in docker container
    """
    host = ''
    try:
        host = f'{getuser()}@{gethostname()}'
    except Exception as e:
        warnings.warn(f'Host or user not found: {str(e)}')
    finally:
        return host
    
    
def compute_sec_to_h_d(sec):
    if sec <=0: return "00:00:00"
    
    if sec < 60: return f'00:00:{f"{int(sec)}".zfill(2)}'
    
    minute = sec//60
    if minute < 60: return f"00:{f'{int(minute)}'.zfill(2)}:{f'{int(sec%60)}'.zfill(2)}"
    
    hour = minute//60
    if hour < 24: return f"{f'{int(hour)}'.zfill(2)}:{f'{int(minute%60)}'.zfill(2)}:{f'{int(sec%60)}'.zfill(2)}"
    
    day = hour//24
    return f"{day}day {f'{int(hour%24)}'.zfill(2)}:{f'{int(minute%(60))}'.zfill(2)}:{f'{int(sec%(60))}'.zfill(2)}"


