
from itertools import chain
from torch.nn.parallel import DataParallel
from hibernation_no1.mmdet.scatter import scatter_inputs

def build_dp(model, cfg, device='cuda', dim=0, **kwargs):
    if device == 'cuda': 
        model = model.cuda()
    
    model = MMDataParallel(model, dim=dim) 
    if kwargs.get('classes', None) is not None:
        model.CLASSES = kwargs['classes']
    model.cfg = cfg
    return model
    
    
class MMDataParallel(DataParallel):
    """The DataParallel module that supports DataContainer.

    MMDataParallel has two main differences with PyTorch DataParallel:

    - It supports a custom type :class:`DataContainer` which allows more
      flexible control of input data during both GPU and CPU inference.
    - It implement two more APIs ``train_step()`` and ``val_step()``.

    .. warning::
        MMDataParallel only supports single GPU training, if you need to
        train with multiple GPUs, please use MMDistributedDataParallel
        instead. If you have multiple GPUs and you just want to use
        MMDataParallel, you can set the environment variable
        ``CUDA_VISIBLE_DEVICES=0`` or instantiate ``MMDataParallel`` with
        ``device_ids=[0]``.

    Args:
        module (:class:`nn.Module`): Module to be encapsulated.
        device_ids (list[int]): Device IDS of modules to be scattered to.
            Defaults to None when GPU is not available.
        output_device (str | int): Device ID for output. Defaults to None.
        dim (int): Dimension used to scatter the data. Defaults to 0.
    """
    def __init__(self, model, dim=0, device_ids = [0]):
        super().__init__(model, dim=dim, device_ids = device_ids)
        self.dim = dim
        
        
    def train_step(self, *inputs):
        assert self.device_ids == [0], "this project is for only single gpu with ID == '0',\
                                        but device_ids is {self.device_ids}"
        # inputs[0]: data_batch, dict
        #    inputs[0].keys():  ['img_metas', 'img', 'gt_bboxes', 'gt_labels', 'gt_masks']
        # inputs[1]: optimizer
        for t in chain(self.module.parameters(), self.module.buffers()):
            if t.device != self.src_device_obj:
                raise RuntimeError(
                    'module must have its parameters and buffers '
                    f'on device {self.src_device_obj} (device_ids[0]) but '
                    f'found one of them on device: {t.device}')
        
       
        inputs = scatter_inputs(inputs, self.device_ids)       
        return self.module.train_step(*inputs[0])
    
    
    def forward(self, *inputs, **kwargs):
        """Override the original forward function.

        The main difference lies in the CPU inference where the data in
        :class:`DataContainers` will still be gathered.
        """
        # kwargs.keys = ['return_loss', 'rescale', 'img_metas', 'img']
       
        return super().forward(*inputs, **kwargs)