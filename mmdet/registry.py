import inspect


def build_from_cfg(cfg, registry: 'Registry') :
    """Build a module from config dict when it is a class configuration, or
    call a function from config dict when it is a function configuration.

    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        registry (:obj:`Registry`): The registry to search the type from.
        default_args (dict, optional): Default initialization arguments.

    Returns:
        object: The constructed object.
    """
    
    if not isinstance(cfg, dict):  raise TypeError(f'cfg must be a dict, but got {type(cfg)}')
    
    if 'type' not in cfg: raise KeyError("cfg must contain the key 'type'")
    if not isinstance(registry, Registry):
        raise TypeError('registry must be an mmcv.Registry object, '
                        f'but got {type(registry)}')
        
    args = cfg.copy()
    obj_type = args.pop('type')
    
    if not isinstance(obj_type, str):
        raise TypeError(
            f'type must be a str type, but got {type(obj_type)}')
        
    obj_cls = registry.get(obj_type)
    # print(f"obj_cls : {obj_cls}")
    if obj_cls is None:
        raise KeyError(
            f'{obj_type} is not in the {registry.name} registry')
    try:
        return obj_cls(**args)
    except Exception as e:
        # Normal TypeError does not print class name.
        
        raise type(e)(f'{obj_cls.__name__}: {e}')
    
    
class Registry:
    """A registry to map strings to classes or functions.

    Registered object could be built from registry. Meanwhile, registered
    functions could be called from registry.

    Example:
        >>> MODELS = Registry('models')
        >>> @MODELS.register_module()
        >>> class ResNet:
        >>>     pass
        >>> resnet = MODELS.build(dict(type='ResNet'))
        >>> @MODELS.register_module()
        >>> def resnet50():
        >>>     pass
        >>> resnet = MODELS.build(dict(type='resnet50'))

    Please refer to
    https://mmcv.readthedocs.io/en/latest/understand_mmcv/registry.html for
    advanced usage.

    Args:
        name (str): Registry name.
        build_func(func, optional): Build function to construct instance from
            Registry, func:`build_from_cfg` is used if neither ``parent`` or
            ``build_func`` is specified. If ``parent`` is specified and
            ``build_func`` is not given,  ``build_func`` will be inherited
            from ``parent``. Default: None.
        parent (Registry, optional): Parent registry. The class registered in
            children registry could be built from parent. Default: None.
        scope (str, optional): The scope of registry. It is the key to search
            for children registry. If not specified, scope will be the name of
            the package where class is defined, e.g. mmdet, mmcls, mmseg.
            Default: None.
    """

    def __init__(self, name):
        self._name = name
        self._module_dict = dict()
        
        self.build_func = build_from_cfg
        
            
        

    def __len__(self):
        return len(self._module_dict)

    def __contains__(self, key):
        return self.get(key) is not None

    def __repr__(self):
        format_str = self.__class__.__name__ + \
                     f'(name={self._name}, ' \
                     f'items={self._module_dict})'
        return format_str

    @property
    def name(self):
        return self._name

    @property
    def module_dict(self):
        return self._module_dict

  

    def get(self, key):
        """Get the registry record.

        Args:
            key (str): The class name in string format.

        Returns:
            class: The corresponding class.
        """
        if key in self._module_dict:
                return self._module_dict[key]
 

    def build(self, *args, **kwargs):
        return self.build_func(*args, **kwargs, registry=self)

    # Example:
    #     >>> INSTANCE = Registry('models')
    #     >>> @INSTANCE.register_module(name = "function name" or None, force=True or False)
    #     >>>  class baz:
    #     >>>  ...
    # Example:
    # >>> backbones = Registry('backbone')
    # >>> @backbones.register_module()
    # >>> class ResNet:
    # >>>     pass

    # >>> backbones = Registry('backbone')
    # >>> @backbones.register_module(name='mnet')
    # >>> class MobileNet:
    # >>>     pass

    # >>> backbones = Registry('backbone')
    # >>> class ResNet:
    # >>>     pass
    # >>> backbones.register_module(ResNet)
    def register_module(self, name=None, force=False, module=None):
        if not isinstance(force, bool):
            raise TypeError(f'force must be a boolean, but got {type(force)}')
        
        # use it as a normal method: x.register_module(module=SomeClass)
        if module is not None:
            self._register_module(module=module, module_name=name, force=force)
            return module

        # use it as a decorator: @x.register_module()
        def _register(module):
            self._register_module(module=module, module_name=name, force=force)
            return module
        
        return _register
    

    
    def _register_module(self, module, module_name=None, force=False):
        """
        Args:
            module (class): class to add to registry
            module_name (str, optional): nickname of class 
            force (bool, optional): True: override if module exist in registry.  False: raise Error

        Raises:
            TypeError: _description_
            KeyError: _description_
        """
        if not inspect.isclass(module) and not inspect.isfunction(module):
            raise TypeError('module must be a class or a function, '
                            f'but got {type(module)}')
        
        if module_name is None:
            module_name = module.__name__
        if isinstance(module_name, str):
            module_name = [module_name]
        for name in module_name:
            
            if not force and name in self._module_dict:
                raise KeyError(f'{name} is already registered '
                               f'in {self.name}')
            self._module_dict[name] = module


