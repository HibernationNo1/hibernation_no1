import os, os.path as osp
import re

import torch

from typing import Optional, Union
from collections import OrderedDict



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