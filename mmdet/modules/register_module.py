from sub_module.mmdet.registry import Registry


BACKBONES = Registry('backbone')

NECKS = Registry('neck')

RPN_HEADS = Registry('rpn_head')

ROI_HEADS = Registry('roi_head')


