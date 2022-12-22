import numpy as np
import torch
import collections.abc
import importlib
from itertools import repeat


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