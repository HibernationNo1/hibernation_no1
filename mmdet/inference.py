import numpy as np
import torch
import itertools

from hibernation_no1.mmdet.data.transforms.utils import replace_ImageToTensor
from hibernation_no1.mmdet.data.transforms.compose import Compose
from hibernation_no1.mmdet.data.dataloader import collate
from hibernation_no1.mmdet.scatter import parallel_scatter
from hibernation_no1.mmdet.checkpoint import load_checkpoint     

def build_detector(cfg, model_path, device='cuda:0', logger = None):
    checkpoint = load_checkpoint(model_path, logger = logger)
    state_dict = checkpoint['state_dict']
    
    state_dict = checkpoint['state_dict']
    metadata = getattr(state_dict, '_metadata', dict())
    meta = checkpoint['meta']
    optimizer = checkpoint['optimizer']
    
    if meta.get("model_cfg", None) is not None:
        model_cfg = meta['model_cfg']
    elif cfg.get('model', None) is not None:
        model_cfg = cfg.model
    else: 
        raise TypeError(f"There is no config for build model.")

    # TODO : build with registry
    if model_cfg.type == 'MaskRCNN':
        model_cfg.pop("type")
        from hibernation_no1.mmdet.modules.detector.maskrcnn import MaskRCNN
        model = MaskRCNN(**model_cfg)        
        
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
        
    # load state_dict
    return load_state_dict(model, state_dict, device = device, logger = logger)
    

def load_state_dict(model: torch.nn.Module, state_dict, device = 'cuda:0',  logger = None):
    """
    Copies parameters and buffers from state_dict into module
    """
    unexpected_keys = []
    all_missing_keys = []
    err_msg = []
    metadata = getattr(state_dict, '_metadata', None)
    
    # sub function
    # assign weight to model 
    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        
        # method of nn.Module
        module._load_from_state_dict(state_dict, prefix, local_metadata, True,
                                     all_missing_keys, unexpected_keys,
                                     err_msg)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')
                

    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata  # type: ignore
        
    load(model)
    # break load->load reference cycle
    load = None  # type: ignore
    
    # ignore "num_batches_tracked" of BN layers
    missing_keys = [
        key for key in all_missing_keys if 'num_batches_tracked' not in key
    ]
    
    if unexpected_keys:
        err_msg.append('unexpected key in source '
                       f'state_dict: {", ".join(unexpected_keys)}\n')
    if missing_keys:
        err_msg.append(
            f'missing keys in source state_dict: {", ".join(missing_keys)}\n')

    if len(err_msg) > 0 :
        err_msg.insert(
            0, 'The model and loaded state dict do not match exactly\n')
        err_msg = '\n'.join(err_msg)  # type: ignore
        if logger is not None:
            logger.warning(err_msg)
        else:
            print(err_msg)
    
    model.to(device)
    model.eval()
    return model
    


def inference_detector(model, imgs_path):
    """Inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs_path (str/ndarray or list[str/ndarray] or tuple[str/ndarray]):
           Either image files or loaded images.
        val: run infernce in validation mode

    Returns:
        If imgs is a list or tuple, the same length list type results
        will be returned, otherwise return the detection results directly.
    """
    
    if isinstance(imgs_path, (list, tuple)):
        is_batch = True
    else:
        imgs_path = [imgs_path]
        is_batch = False
    
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    

    if  cfg.get("test_pipeline", None) is not None: 
        pipeline_cfg = cfg.test_pipeline
    elif cfg.get("val_infer_pipeline", None) is not None:
        pipeline_cfg = cfg.val_infer_pipeline
    else: raise ValueError("val or test config must be specific, but both got None")

    re_pipeline_cfg  = replace_ImageToTensor(pipeline_cfg)
    pipeline = Compose(re_pipeline_cfg)
    
    datas = []

    for img_path in imgs_path:
        # prepare data
        data = dict(img_info=dict(file_name=img_path), img_prefix=None)
        # build the data pipeline
        data = pipeline(data)
        datas.append(data)
    
    # just get the actual data from DataContainer
    # len(data): batch_szie
    data = collate(datas, samples_per_gpu=len(datas))
    
    
    data['img_metas'] = [img_metas.data[0] for img_metas in data['img_metas']]
    data['img'] = [img.data[0] for img in data['img']]
    
    assert next(model.parameters()).is_cuda, f"modules must be is_cuda, but is not"
    # scatter to specified GPU
    
    # data.keys(): ['img_metas', 'img'],       len(data['key']): 1
    # len(data['key'][0]): batch_size
    data = parallel_scatter(data, [device])[0]

    # forward the model
    with torch.no_grad():
        results = model(return_loss=False, rescale=True, **data)        # call model.forward
    if not is_batch:
        return results[0]
    else:
        return results

 
def parse_inferece_result(result):
    if isinstance(result, tuple):
        bbox_result, segm_result = result
        if isinstance(segm_result, tuple):
            segm_result = segm_result[0]  # ms rcnn
    else:
        bbox_result, segm_result = result, None

    
    # bboxes.shape: (num of instance, 5)    5: [x_min, y_min, x_max, y_max, score]
    bboxes = np.vstack(bbox_result)
    
  
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    # labels.shape[0]: num of instance
    labels = np.concatenate(labels)

    # draw segmentation masks
    segms = None
    if segm_result is not None and len(labels) > 0:  # non empty
        # len(segms): num of instance
        segms = list(itertools.chain(*segm_result))

        # segms.shape: (num of instance , height, widrh)
        if isinstance(segms[0], torch.Tensor):
            segms = torch.stack(segms, dim=0).detach().cpu().numpy()
        else:
            segms = np.stack(segms, axis=0)         
        
    return bboxes, labels, segms
