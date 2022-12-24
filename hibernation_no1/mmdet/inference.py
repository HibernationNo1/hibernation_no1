import numpy as np
import torch
import itertools

from hibernation_no1.mmdet.data.transforms.utils import replace_ImageToTensor
from hibernation_no1.mmdet.data.transforms.compose import Compose
from hibernation_no1.mmdet.data.dataloader import collate
from hibernation_no1.mmdet.scatter import parallel_scatter


def inference_detector(model, imgs_path, batch_size):
    """Inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str/ndarray or list[str/ndarray] or tuple[str/ndarray]):
           Either image files or loaded images.

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
    else: raise ValueError("val or test config must be specific, but both got None")

    re_pipeline_cfg  = replace_ImageToTensor(pipeline_cfg)
    pipeline = Compose(re_pipeline_cfg)
    
    datas = []
    for img_path in imgs_path:
        # prepare data
        data = dict(img_info=dict(filename=img_path), img_prefix=None)
        
        # build the data pipeline
        data = pipeline(data)
        datas.append(data)
    
    # just get the actual data from DataContainer
    # len(data): batch_szie
    data = collate(datas, samples_per_gpu=batch_size)
    
    
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
