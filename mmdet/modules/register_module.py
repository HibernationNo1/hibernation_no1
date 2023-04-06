from sub_module.mmdet.registry import Registry



from mmdet.models.backbones.swin import SwinTransformer as _SwinTransformer
BACKBORN = Registry('backbone')
@BACKBORN.register_module()
def SwinTransformer(**cfg):
    return _SwinTransformer(**cfg)


from mmdet.models.necks.fpn import FPN as _FPN
NECK = Registry('neck')
@NECK.register_module()
def FPN(**cfg):
    return _FPN(**cfg)

from mmdet.models.dense_heads.rpn_head import RPNHead as _RPNHead 
RPN_HEAD = Registry('rpn_head')
@RPN_HEAD.register_module()
def RPNHead(**cfg):
    return _RPNHead(**cfg)


from mmdet.models.roi_heads.standard_roi_head import StandardRoIHead as _StandardRoIHead
ROI_HEAD = Registry('roi_head')
@ROI_HEAD.register_module()
def StandardRoIHead(**cfg):
    return _StandardRoIHead(**cfg)

