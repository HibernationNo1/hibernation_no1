
from hibernation_no1.mmdet.modules.base.module import BaseModule
from hibernation_no1.mmdet.modules.swin_transformer.swin_transformer import SwinTransformer

class MaskRCNN(BaseModule):
    def __init__(self,
                 backbone,
                 neck,
                 rpn_head,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None):
        super(MaskRCNN, self).__init__(init_cfg)
        
        # TODO: maneging with registry
        self.backbone = SwinTransformer(**backbone)
        print(f"self.backbone : \n{self.backbone}")        