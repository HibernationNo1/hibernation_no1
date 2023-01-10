import numpy as np
import cv2
import copy

cv2_interp_codes = {
    'nearest': cv2.INTER_NEAREST,
    'bilinear': cv2.INTER_LINEAR,
    'bicubic': cv2.INTER_CUBIC,
    'area': cv2.INTER_AREA,
    'lanczos': cv2.INTER_LANCZOS4
}



def imrescale(img, scale, return_scale=False, interpolation='bilinear'):
    h, w = img.shape[:2]
    new_size, scale_factor = rescale_size((w, h), scale, return_scale=True)
    rescaled_img = imresize(img, new_size, 
                                    interpolation=interpolation)
    if return_scale:
        return rescaled_img, scale_factor
    else:
        return rescaled_img
        

def rescale_size(old_size, scale, return_scale=False):
    w, h = old_size
    if isinstance(scale, (float, int)):
        if scale <= 0:
            raise ValueError(f'Invalid scale {scale}, must be positive.')
        scale_factor = scale
    elif isinstance(scale, tuple):
        max_long_edge = max(scale)
        max_short_edge = min(scale)
        scale_factor = min(max_long_edge / max(h, w),
                        max_short_edge / min(h, w))
    else:
        raise TypeError(
            f'Scale must be a number or tuple of int, but got {type(scale)}')

    scale = (scale_factor, scale_factor)
    new_w, new_h = int(w * float(scale[0]) + 0.5), int(h * float(scale[1]) + 0.5)
        
    new_size = (new_w, new_h)

    if return_scale:
        return new_size, scale_factor
    else:
        return new_size


def imresize(img, size,
             return_scale=False,
             interpolation='bilinear',
             out=None):
    """Resize image to a given size.

    Args:
        img (ndarray): The input image.
        size (tuple[int]): Target size (w, h).
        return_scale (bool): Whether to return `w_scale` and `h_scale`.
        interpolation (str): Interpolation method, accepted values are
            "nearest", "bilinear", "bicubic", "area", "lanczos" for 'cv2'
            backend, "nearest", "bilinear" for 'pillow' backend.
        out (ndarray): The output destination.

    Returns:
        tuple | ndarray: (`resized_img`, `w_scale`, `h_scale`) or
        `resized_img`.
    """
    h, w = img.shape[:2]
    resized_img = cv2.resize(img, size, dst=out, 
                                interpolation=cv2_interp_codes[interpolation])
    
    if not return_scale:
        return resized_img
    else:
        w_scale = size[0] / w
        h_scale = size[1] / h
        return resized_img, w_scale, h_scale
    

def imflip(img, direction='horizontal'):
    """Flip an image horizontally or vertically.

    Args:
        img (ndarray): Image to be flipped.
        direction (str): The flip direction, either "horizontal" or
            "vertical" or "diagonal".

    Returns:
        ndarray: The flipped image.
    """
    assert direction in ['horizontal', 'vertical', 'diagonal']
    if direction == 'horizontal':
        return np.flip(img, axis=1)
    elif direction == 'vertical':
        return np.flip(img, axis=0)
    else:
        return np.flip(img, axis=(0, 1))
    
    

def replace_ImageToTensor(pipelines):
    """Replace the ImageToTensor transform in a data pipeline to
    DefaultFormatBundle, which is normally useful in batch inference.

    Args:
        pipelines (list[dict]): Data pipeline configs.

    Returns:
        list: The new pipeline list with all ImageToTensor replaced by
            DefaultFormatBundle.

    Examples:
        >>> pipelines = [
        ...    dict(type='LoadImageFromFile'),
        ...    dict(
        ...        type='MultiScaleFlipAug',
        ...        img_scale=(1333, 800),
        ...        flip=False,
        ...        transforms=[
        ...            dict(type='Resize', keep_ratio=True),
        ...            dict(type='RandomFlip'),
        ...            dict(type='Normalize', mean=[0, 0, 0], std=[1, 1, 1]),
        ...            dict(type='Pad', size_divisor=32),
        ...            dict(type='ImageToTensor', keys=['img']),
        ...            dict(type='Collect', keys=['img']),
        ...        ])
        ...    ]
        >>> expected_pipelines = [
        ...    dict(type='LoadImageFromFile'),
        ...    dict(
        ...        type='MultiScaleFlipAug',
        ...        img_scale=(1333, 800),
        ...        flip=False,
        ...        transforms=[
        ...            dict(type='Resize', keep_ratio=True),
        ...            dict(type='RandomFlip'),
        ...            dict(type='Normalize', mean=[0, 0, 0], std=[1, 1, 1]),
        ...            dict(type='Pad', size_divisor=32),
        ...            dict(type='DefaultFormatBundle'),
        ...            dict(type='Collect', keys=['img']),
        ...        ])
        ...    ]
        >>> assert expected_pipelines == replace_ImageToTensor(pipelines)
    """
    pipelines = copy.deepcopy(pipelines)
    for pipeline in pipelines:
        if pipeline['type'] == 'MultiScaleFlipAug':
            assert 'transforms' in pipeline
            pipeline['transforms'] = replace_ImageToTensor(pipeline['transforms'])
            
            
    return pipelines   

