from .config import Config


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
    cfg_flag = cfg.pop(key_name, None)
    if cfg_flag is not None:
        cfg = change_to_tuple(cfg, cfg_flag)
        cfg = Config(cfg)
    else: raise KeyError(f"cfg must have key `{key_name}` to run `change_to_tuple`, but got None.")
    
    return cfg

