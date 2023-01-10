
from hibernation_no1.mmdet.registry import Registry, build_from_cfg

PIPELINES = Registry('pipeline')

class Compose:
    def __init__(self, transforms):
        # processing pipeline
        self.transforms = []
        for transform in transforms:
            if isinstance(transform, dict):
                transform = build_from_cfg(transform, PIPELINES)
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError('transform must be callable or a dict')


    def __call__(self, data):
        """Call function to apply transforms sequentially.

        Args:
            data (dict): A result dict contains the data to transform.

        Returns:
        dict: Transformed data.
        """
        for transform in self.transforms:
            data = transform(data)
            if data is None:
                return None
    
        return data  
    
    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            str_ = t.__repr__()
            if 'Compose(' in str_:
                str_ = str_.replace('\n', '\n    ')
            format_string += '\n'
            format_string += f'    {str_}'
        format_string += '\n)'
        return format_string
    
    