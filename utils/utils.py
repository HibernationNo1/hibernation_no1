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
    
    

def dict_to_pretty(a_dict: dict):
    """print key and items to pretty

    Args:
        a_dict (dict): 

    Returns:
        text (str) 
    """
    if not isinstance(a_dict, dict) : raise TypeError(f"cfg must be dict, but get {type(a_dict)}")
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

    text = _format_dict(a_dict, outest_level=True)
    
    return text



def is_list_of(seq, expected_type):
    """Check whether it is a sequence of list type.
    """
    if not isinstance(seq, list):
        return False
    
    for item in seq:
        if not isinstance(item, expected_type):
            return False

    return True


def is_tuple_of(seq, expected_type):
    """Check whether it is a sequence of tuple type.
    """
    
    if not isinstance(seq, tuple):
        return False
    
    for item in seq:
        if not isinstance(item, expected_type):
            return False

    return True