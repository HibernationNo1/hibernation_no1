# Copyright (c) OpenMMLab. All rights reserved.
# modified by Hibernation_No1
from pathlib import Path
import tempfile
import platform
import os, os.path as osp
import warnings
import copy
import types
import shutil
import uuid
import sys
import ast
from importlib import import_module

from addict import Dict

if platform.system() == 'Windows':
    import regex as re  # type: ignore
else:
    import re  # type: ignore
    
BASE_KEY = '_base_'
DELETE_KEY = '_delete_'
RESERVED_KEYS = ['filename', 'text', 'pretty_text']
CONFIGDICT_NAME = 'class_config'


class ConfigDict(Dict) :
    @property
    def _class_name(self):
        return CONFIGDICT_NAME

    def __missing__(self, name):
        raise KeyError(name)

    def __getattr__(self, name):
        try:
            value = super().__getattr__(name)
        except KeyError:
            ex = AttributeError(f"'{self.__class__.__name__}' object has no "
                                f"attribute '{name}'")
        except Exception as e:
            ex = e
        else:
            return value
        raise ex
    
    
class Config:
    """A facility for config and config files.

    It supports common file formats as configs: python/json/yaml. The interface
    is the same as a dict object and also allows access config values as
    attributes.

    Example:
        >>> cfg = Config(dict(a=1, b=dict(b1=[0, 1])))
        >>> cfg.a
        1
        >>> cfg.b
        {'b1': [0, 1]}
        >>> cfg.b.b1
        [0, 1]
        >>> cfg = Config.fromfile('tests/data/config/a.py')
        >>> cfg.filename
        "/home/kchen/projects/mmcv/tests/data/config/a.py"
        >>> cfg.item4
        'test'
        >>> cfg
        "Config [path: /home/kchen/projects/mmcv/tests/data/config/a.py]: "
        "{'item1': [1, 2], 'item2': {'a': 0}, 'item3': True, 'item4': 'test'}"
    """
    
    
    def __init__(self, cfg_dict=None, cfg_text=None, filename=None):
        if cfg_dict is None:
            cfg_dict = dict()
        elif not isinstance(cfg_dict, dict):
            raise TypeError('cfg_dict must be a dict, but '
                            f'got {type(cfg_dict)}')
        for key in cfg_dict:
            if key in RESERVED_KEYS:
                raise KeyError(f'{key} is reserved for config file')

        if isinstance(filename, Path):
            filename = str(filename)

        super().__setattr__('_cfg_dict', ConfigDict(cfg_dict))
        super().__setattr__('_filename', filename)
        if cfg_text:
            text = cfg_text
        elif filename:
            with open(filename) as f:
                text = f.read()
        else:
            text = ''
            
        self._text = text
        self._filename = filename
        super().__setattr__('_text', text)
        
    @property
    def filename(self):
        return self._filename

    @property
    def text(self):
        return self._text
    
    @property
    def pretty_text(self):

        indent = 4

        def _indent(s_, num_spaces):
            s = s_.split('\n')
            if len(s) == 1:
                return s_
            first = s.pop(0)
            s = [(num_spaces * ' ') + line for line in s]
            s = '\n'.join(s)
            s = first + '\n' + s
            return s

        def _format_basic_types(k, v, use_mapping=False):
            if isinstance(v, str):
                v_str = f"'{v}'"
            else:
                v_str = str(v)

            if use_mapping:
                k_str = f"'{k}'" if isinstance(k, str) else str(k)
                attr_str = f'{k_str}: {v_str}'
            else:
                attr_str = f'{str(k)}={v_str}'
            attr_str = _indent(attr_str, indent)

            return attr_str

        def _format_list(k, v, use_mapping=False):
            # check if all items in the list are dict
            if all(isinstance(_, dict) for _ in v):
                v_str = '[\n'
                v_str += '\n'.join(
                    f'dict({_indent(_format_dict(v_), indent)}),'
                    for v_ in v).rstrip(',')
                if use_mapping:
                    k_str = f"'{k}'" if isinstance(k, str) else str(k)
                    attr_str = f'{k_str}: {v_str}'
                else:
                    attr_str = f'{str(k)}={v_str}'
                attr_str = _indent(attr_str, indent) + ']'
            else:
                attr_str = _format_basic_types(k, v, use_mapping)
            return attr_str

        def _contain_invalid_identifier(dict_str):
            contain_invalid_identifier = False
            for key_name in dict_str:
                contain_invalid_identifier |= \
                    (not str(key_name).isidentifier())
            return contain_invalid_identifier

        def _format_dict(input_dict, outest_level=False):
            r = ''
            s = []

            use_mapping = _contain_invalid_identifier(input_dict)
            if use_mapping:
                r += '{'
            for idx, (k, v) in enumerate(input_dict.items()):
                is_last = idx >= len(input_dict) - 1
                end = '' if outest_level or is_last else ','
                if isinstance(v, dict):
                    v_str = '\n' + _format_dict(v)
                    if use_mapping:
                        k_str = f"'{k}'" if isinstance(k, str) else str(k)
                        attr_str = f'{k_str}: dict({v_str}'
                    else:
                        attr_str = f'{str(k)}=dict({v_str}'
                    attr_str = _indent(attr_str, indent) + ')' + end
                elif isinstance(v, list):
                    attr_str = _format_list(k, v, use_mapping) + end
                else:
                    attr_str = _format_basic_types(k, v, use_mapping) + end

                s.append(attr_str)
            r += '\n'.join(s)
            if use_mapping:
                r += '}'
            return r

        cfg_dict = self._cfg_dict.to_dict()
        text = _format_dict(cfg_dict, outest_level=True)
        
        return text

    def __getitem__(self, name):
        return self._cfg_dict.__getitem__(name)
    
    def __getattr__(self, name):
        return getattr(self._cfg_dict, name)
        
    def __repr__(self):
        return f'Config (path: {self.filename}): {self._cfg_dict.__repr__()}'
    
    
    def __copy__(self):
        cls = self.__class__
        other = cls.__new__(cls)
        other.__dict__.update(self.__dict__)

        return other

    def __deepcopy__(self, memo):
        cls = self.__class__
        other = cls.__new__(cls)
        memo[id(self)] = other

        for key, value in self.__dict__.items():
            super(Config, other).__setattr__(key, copy.deepcopy(value, memo))

        return other
    
    @staticmethod
    def fromfile(filename, use_predefined_variables=True):
        if isinstance(filename, Path):
            filename = str(filename)
        cfg_dict, cfg_text = Config._file2dict(filename, use_predefined_variables= use_predefined_variables)

        return Config(cfg_dict, cfg_text=cfg_text, filename=filename)
        
        
    def dump(self, file=None):
        """Dumps config into a file or returns a string representation of the
        config.

        If a file argument is given, saves the config to that file using the
        format defined by the file argument extension.

        Otherwise, returns a string representing the config. The formatting of
        this returned string is defined by the extension of `self.filename`. If
        `self.filename` is not defined, returns a string representation of a
         dict (lowercased and using ' for strings).

        Examples:
            >>> cfg_dict = dict(item1=[1, 2], item2=dict(a=0),
            ...     item3=True, item4='test')
            >>> cfg = Config(cfg_dict=cfg_dict)
            >>> dump_file = "a.py"
            >>> cfg.dump(dump_file)

        Args:
            file (str, optional): Path of the output file where the config
                will be dumped. Defaults to None.
        """
        if file is None:
            if self.filename is None or self.filename.endswith('.py'):
                return self.pretty_text
            else:
                raise OSError(f'{file} is not exist path')
                
        elif file.endswith('.py'):
            with open(file, 'w', encoding='utf-8') as f:
                f.write(self.pretty_text)
        else:
            warnings.warn(f'{file} is not .py format')
            with open(file, 'w', encoding='utf-8') as f:
                f.write(self.pretty_text)
                
                
    
    @staticmethod
    def _file2dict(filename, json_dict = None, use_predefined_variables=True):
        filename = osp.abspath(osp.expanduser(filename))
        
        if not osp.isfile(filename):
            raise FileNotFoundError(f'file "{filename}" does not exist')
        
        fileExtname = osp.splitext(filename)[1]
        if fileExtname != '.py':
            raise OSError('Only py type are supported now!')
        
        with tempfile.TemporaryDirectory() as temp_config_dir:
            temp_config_file = tempfile.NamedTemporaryFile(
                dir=temp_config_dir, suffix=fileExtname)
            if platform.system() == 'Windows':
                temp_config_file.close()
            temp_config_name = osp.basename(temp_config_file.name)
            
            # Substitute predefined variables
            if use_predefined_variables:
                Config._substitute_predefined_vars(filename,temp_config_file.name)
            else:
                shutil.copyfile(filename, temp_config_file.name)
                
            # Substitute base variables from placeholders to strings
            base_var_dict = Config._pre_substitute_base_vars(temp_config_file.name, temp_config_file.name)
            
            temp_module_name = osp.splitext(temp_config_name)[0]
            sys.path.insert(0, temp_config_dir)
            Config._validate_py_syntax(filename)
            mod = import_module(temp_module_name) 
            sys.path.pop(0)
            cfg_dict = {
                name: value
                for name, value in mod.__dict__.items()
                if not name.startswith('__')
                and not isinstance(value, types.ModuleType)
                and not isinstance(value, types.FunctionType)
            }
            # delete imported module
            del sys.modules[temp_module_name]
            # close temp file
            temp_config_file.close()
            
        if json_dict is not None: 
            cfg_dict = merge_config(cfg_dict, json_dict, "init")

        cfg_text = filename + '\n'
        with open(filename, encoding='utf-8') as f:
            # Setting encoding explicitly to resolve coding issue on windows
            cfg_text += f.read()
        
        if BASE_KEY in cfg_dict:
            cfg_dir = osp.dirname(filename)
            base_filename = cfg_dict.pop(BASE_KEY)
            base_filename = base_filename if isinstance(
                base_filename, list) else [base_filename]

            cfg_dict_list = list()
            cfg_text_list = list()
            for f in base_filename:
                _cfg_dict, _cfg_text = Config._file2dict(osp.join(cfg_dir, f))
                cfg_dict_list.append(_cfg_dict)
                cfg_text_list.append(_cfg_text)

            base_cfg_dict = dict()
            for c in cfg_dict_list:
                duplicate_keys = base_cfg_dict.keys() & c.keys()
                if len(duplicate_keys) > 0:
                    raise KeyError('Duplicate key is not allowed among bases. '
                                   f'Duplicate keys: {duplicate_keys}')
                base_cfg_dict.update(c)

            # Substitute base variables from strings to their actual values
            cfg_dict = Config._substitute_base_vars(cfg_dict, base_var_dict,
                                                    base_cfg_dict)

            base_cfg_dict = Config._merge_a_into_b(cfg_dict, base_cfg_dict)
            cfg_dict = base_cfg_dict

            # merge cfg_text
            cfg_text_list.append(cfg_text)
            cfg_text = '\n'.join(cfg_text_list)

        return cfg_dict, cfg_text
    
    
    
    @staticmethod
    def _merge_a_into_b(a, b, allow_list_keys=False):
        """merge dict ``a`` into dict ``b`` (non-inplace).

        Values in ``a`` will overwrite ``b``. ``b`` is copied first to avoid
        in-place modifications.

        Args:
            a (dict): The source dict to be merged into ``b``.
            b (dict): The origin dict to be fetch keys from ``a``.
            allow_list_keys (bool): If True, int string keys (e.g. '0', '1')
              are allowed in source ``a`` and will replace the element of the
              corresponding index in b if b is a list. Default: False.

        Returns:
            dict: The modified dict of ``b`` using ``a``.

        Examples:
            # Normally merge a into b.
            >>> Config._merge_a_into_b(
            ...     dict(obj=dict(a=2)), dict(obj=dict(a=1)))
            {'obj': {'a': 2}}

            # Delete b first and merge a into b.
            >>> Config._merge_a_into_b(
            ...     dict(obj=dict(_delete_=True, a=2)), dict(obj=dict(a=1)))
            {'obj': {'a': 2}}

            # b is a list
            >>> Config._merge_a_into_b(
            ...     {'0': dict(a=2)}, [dict(a=1), dict(b=2)], True)
            [{'a': 2}, {'b': 2}]
        """
        b = b.copy()
        for k, v in a.items():
            if allow_list_keys and k.isdigit() and isinstance(b, list):
                k = int(k)
                if len(b) <= k:
                    raise KeyError(f'Index {k} exceeds the length of list {b}')
                b[k] = Config._merge_a_into_b(v, b[k], allow_list_keys)
            elif isinstance(v, dict):
                if k in b and not v.pop(DELETE_KEY, False):
                    allowed_types = (dict, list) if allow_list_keys else dict
                    if not isinstance(b[k], allowed_types):
                        raise TypeError(
                            f'{k}={v} in child config cannot inherit from '
                            f'base because {k} is a dict in the child config '
                            f'but is of type {type(b[k])} in base config. '
                            f'You may set `{DELETE_KEY}=True` to ignore the '
                            f'base config.')
                    b[k] = Config._merge_a_into_b(v, b[k], allow_list_keys)
                else:
                    b[k] = ConfigDict(v)
            else:
                b[k] = v
        return b


    
    @staticmethod
    def _validate_py_syntax(filename):
        with open(filename, encoding='utf-8') as f:
            # Setting encoding explicitly to resolve coding issue on windows
            content = f.read()
        
        try:
            ast.parse(content)
        except SyntaxError as e:
            raise SyntaxError('There are syntax errors in config '
                              f'file {filename}: {e}')
    
    
    @staticmethod
    def _substitute_base_vars(cfg, base_var_dict, base_cfg):
        """Substitute variable strings to their actual values."""
        cfg = copy.deepcopy(cfg)

        if isinstance(cfg, dict):
            for k, v in cfg.items():
                if isinstance(v, str) and v in base_var_dict:
                    new_v = base_cfg
                    for new_k in base_var_dict[v].split('.'):
                        new_v = new_v[new_k]
                    cfg[k] = new_v
                elif isinstance(v, (list, tuple, dict)):
                    cfg[k] = Config._substitute_base_vars(
                        v, base_var_dict, base_cfg)
        elif isinstance(cfg, tuple):
            cfg = tuple(
                Config._substitute_base_vars(c, base_var_dict, base_cfg)
                for c in cfg)
        elif isinstance(cfg, list):
            cfg = [
                Config._substitute_base_vars(c, base_var_dict, base_cfg)
                for c in cfg
            ]
        elif isinstance(cfg, str) and cfg in base_var_dict:
            new_v = base_cfg
            for new_k in base_var_dict[cfg].split('.'):
                new_v = new_v[new_k]
            cfg = new_v

        return cfg 
            
    
    @staticmethod
    def _pre_substitute_base_vars(filename, temp_config_name):
        """Substitute base variable placehoders to string, so that parsing
        would work."""
        with open(filename, encoding='utf-8') as f:
            # Setting encoding explicitly to resolve coding issue on windows
            config_file = f.read()
        base_var_dict = {}
        regexp = r'\{\{\s*' + BASE_KEY + r'\.([\w\.]+)\s*\}\}'
        base_vars = set(re.findall(regexp, config_file))
        for base_var in base_vars:
            randstr = f'_{base_var}_{uuid.uuid4().hex.lower()[:6]}'
            base_var_dict[randstr] = base_var
            regexp = r'\{\{\s*' + BASE_KEY + r'\.' + base_var + r'\s*\}\}'
            config_file = re.sub(regexp, f'"{randstr}"', config_file)
        with open(temp_config_name, 'w', encoding='utf-8') as tmp_config_file:
            tmp_config_file.write(config_file)
        return base_var_dict
    
    
    @staticmethod
    def _substitute_predefined_vars(filename, temp_config_name):
        file_dirname = osp.dirname(filename)
        file_basename = osp.basename(filename)
        file_basename_no_extension = osp.splitext(file_basename)[0]
        file_extname = osp.splitext(filename)[1]
        support_templates = dict(
            fileDirname=file_dirname,
            fileBasename=file_basename,
            fileBasenameNoExtension=file_basename_no_extension,
            fileExtname=file_extname)
        with open(filename, encoding='utf-8') as f:
            # Setting encoding explicitly to resolve coding issue on windows
            config_file = f.read()
        for key, value in support_templates.items():
            regexp = r'\{\{\s*' + str(key) + r'\s*\}\}'
            value = value.replace('\\', '/')
            config_file = re.sub(regexp, value, config_file)
        with open(temp_config_name, 'w', encoding='utf-8') as tmp_config_file:
            tmp_config_file.write(config_file)


def merge_config(org_cfg, from_cfg, flag = None):    
    """
        org_cfg : original config
        from_cfg : original config에 merge하고자 하는 config
    """
    if flag == "init": 
        if isinstance(from_cfg, Config): from_cfg = dict(from_cfg)   
        
        if not (isinstance(org_cfg, dict) and isinstance(from_cfg, dict)):
            raise TypeError(f" org_cfg and from_cfg type must be dict."\
                            f"\n org_cfg type: {type(org_cfg)}, from_cfg type: {type(from_cfg)}")
    
    if type(from_cfg) != type(org_cfg): raise TypeError(f" 'from_cfg' map must be same as 'org_cfg' map \nfrom_dict: {from_cfg}\norg_cfg : {org_cfg}")
        
    if isinstance(org_cfg, dict):
        org_dict_keys_list = list(org_cfg.keys())
        from_dict_keys_list = list(from_cfg.keys())
    
        for from_key in from_dict_keys_list: 
            if from_key in ["img_scale", "betas", 'out_indices', 'out_indices']:
                org_cfg[from_key] = tuple(from_cfg[from_key])       
                continue
            if from_key == "workflow":
                tmp_list = []
                for flow in from_cfg[from_key]:
                    tmp_list.append(tuple(flow))
                
                org_cfg[from_key] = tmp_list
                continue
                
            if from_key not in org_dict_keys_list:      # from_cfg의 특정 key가 org_cfg에 없는 경우
                org_cfg[from_key] = from_cfg[from_key]  # org_cfg에 from_cfg의 key:value 추가
            else:                                                                           # from_cfg의 특정 key가 org_cfg에 있는 경우
                org_cfg[from_key] = merge_config(org_cfg[from_key], from_cfg[from_key])     # 해당 key값에 merge
    else:
        org_cfg = from_cfg      # 아예 교체한다.

    return org_cfg