
import os, os.path as osp
import json
import numpy as np
from pathlib import Path

from .config import ConfigDict, Config


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)
        
    

def emptyfile_to_config(cfg_dict, boolean_flag_dict = None, file_path =None, use_predefined_variables=True):
    """
        copy `cfg_dict` to empty file
    Args:
        cfg_dict (dict or Config): dict of config
        boolean_flag_dict:
        file_path (str): path of tmp_file
        use_predefined_variables (bool, optional): Defaults to True.

    Returns:
        _type_: _description_
    """
    if boolean_flag_dict is not None: 
        cfg = change_to_tuple(cfg_dict, boolean_flag_dict)
    
    if file_path is None: file_path = "tmp.py"
    else:
        if osp.splitext(file_path)[-1] != "py":
            file_path = osp.splitext(file_path)[0] +".py"
    with open(file_path, 'w') as f:
        f.write('\n')       # 빈 file 생성
        
    if isinstance(file_path, Path):
            file_path = str(file_path)
    cfg_dict, _ = Config._file2dict(file_path, 
                                    json_dict = cfg_dict,
                                    use_predefined_variables= use_predefined_variables)
    return Config(cfg_dict)       
    
    
def dump_sub_key(sub_config, file_path):
    """
        dump sub dict of `Config`
        class `Config` have `dump` method, but it dose not support dump sub dict(such as `cfg.sub`) 

    Args:
        sub_config (_type_): sub dict of `Config`, `type: `ConfigDict`
        file_path (_type_): expected path of file to save as `.py`format

    Raises:
        TypeError: _description_
    """
    if not isinstance(sub_config, ConfigDict):
        raise TypeError(f"type of sub_config does not match as ConfigDict. type:{type(sub_config)}")
    
    if osp.isfile(file_path):
        raise OSError(f"path: {file_path} is exist!!")
    
    cfg_2 = Config(sub_config)
    cfg_2.dump(file_path)
    
    
def pretty_text_sub_key(sub_config):
    cfg_2 = Config(sub_config)
    return cfg_2.pretty_text
    
    
def get_tuple_key(cfg):    
    """ 
        return boolean flag equal to the dict map of input 'cfg'. 
        flag gets True or index where if type of key and type of value in list are tuple.

    Args:
        cfg (_type_): config dict

    Returns:
        _type_: config dict, all value are boolean.
    """

    if isinstance(cfg, dict) or isinstance(cfg, Config):
        tmp_dict = {}
        for key in list(cfg.keys()):
            is_tuple = get_tuple_key(cfg[key]) 
            
            if is_tuple :
                tmp_dict[key] = is_tuple
            else: continue

        
        return tmp_dict     
    elif isinstance(cfg, tuple):
        return True
    
    elif isinstance(cfg, list):
        tmp_list = []
        for i, ele in enumerate(cfg):       # list에 tuple이 포함되어 있는 경우
            is_tuple = get_tuple_key(ele)
            if isinstance(is_tuple, dict):
                tmp_list.append(is_tuple)
            elif isinstance(is_tuple, bool) and is_tuple:
                tmp_list.append(i)      
            
            
            else: continue
            
        
        if len(tmp_list) == 0: return False
        return tmp_list
    
    else: return False