import warnings
from .config import Config
from .utils import all_dict_in_list


def change_to_tuple(org_cfg, boolean_flag_dict):
    """
    org_cfg : original config
    boolean_flag_dict : key or index of list that type was tuple at original config
    """

    if isinstance(boolean_flag_dict, dict):
        if isinstance(org_cfg, Config):
            org_cfg = dict(org_cfg)
        if not (isinstance(org_cfg, dict)):
            raise TypeError(f"type of org_cfg does not match as Config or dict. type:{type(org_cfg)}")
             
        for key in list(boolean_flag_dict.keys()) :
            if key in list(org_cfg.keys()):
                org_cfg[key] = change_to_tuple(org_cfg[key], boolean_flag_dict[key])
                
    elif isinstance(boolean_flag_dict, list):
        assert isinstance(org_cfg, list)
            
        tmp_list = []
        for idx, ele in enumerate(boolean_flag_dict):
            if isinstance(ele, dict):
                if len(list(ele.keys())) == 0: tmp_list.append(org_cfg[idx])
                else: tmp_list.append(change_to_tuple(org_cfg[idx], ele))
            elif isinstance(ele, int): tmp_list.append(tuple(org_cfg[ele]))
        return tmp_list
    
    elif boolean_flag_dict :
        return tuple(org_cfg)
    
    return org_cfg


def dict2Config(cfg, key_name = "flag_list2tuple"):
    if not isinstance(cfg, dict):
        raise TypeError(f"type(cfg) must be dict, but type is '{type(cfg)}'")

    cfg_flag = cfg.pop(key_name, None)
    if cfg_flag is not None:
        cfg = change_to_tuple(cfg, cfg_flag)
        cfg = Config(cfg)
    else: raise KeyError(f"cfg must have key `{key_name}` to run `change_to_tuple`, but got None.")
    
    return cfg



def replace_config(from_cfg, to_cfg, init = False):
    """
        Search the key and value of `from_cfg`
        and copy `from_cfg.key` to `to_cfg.key` if the key value of `to_cfg` 
        exists in `from_cfg` and the value is different
    """

    def chack_cfg_type(cfg):
        if isinstance(cfg, dict):
            cfg = Config(cfg)
        elif not isinstance(cfg, Config):
            raise TypeError(f"type(cfg) must be dict or Config. but type is '{type(cfg)}'")

        return cfg
    
    from_cfg = chack_cfg_type(from_cfg)
    to_cfg = chack_cfg_type(to_cfg)

    # If have a key: 'type', need to check if the type is same.
    # If not have, continue.
    if to_cfg.get('type', 'check_type') != from_cfg.get('type', 'check_type'):    
        return None
  
    for key, item in from_cfg.items():
        # If to_cfg not have `key`: continue
        if to_cfg.get(key, None) is None:
            to_cfg.key = item
            continue

        # If value is the same: continue
        if to_cfg[key] == item:
            continue


        if isinstance(item, dict):
            result_cfg = replace_config(from_cfg = item, to_cfg = to_cfg[key])
            if result_cfg is None:
                continue
            else:
                from_cfg[key] = result_cfg
        elif isinstance(item, list) or isinstance(item, tuple):
            if all_dict_in_list:
                warnings.warn(f"Lists where all elements are dict are not replaced!! \n"
                              f"from_cfg[{key}]: {from_cfg[key]} \nto_cfg[{key}]: {to_cfg[key]}")
                continue
             
            else:   # If it is a list that does not contain a dict
              
                from_cfg[key] = to_cfg[key]

            
        else:
            from_cfg[key] = to_cfg[key]


    if init: return from_cfg
    return dict(from_cfg)
           