import os
# rename file name: kubernetes


def get_environ(cfg, key: str):
    """For get value from kubernetest secret

    Args:
        cfg (dict or Config): 
        key (str): key to get value
    """
    
    # If `key` is in `cfg` and its value is not 'None',
    # the corresponding value is returned.
    if cfg.get(key, None) is not None:
        return cfg.get(key, None)
    else:
        return os.environ[key]