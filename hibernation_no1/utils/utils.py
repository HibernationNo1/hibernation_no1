import os
# rename file name: kubernetes
def get_environ(cfg, key):
    if cfg.get(key, None) is not None:
        return cfg.get(key, None)
    else:
        return os.environ[key]