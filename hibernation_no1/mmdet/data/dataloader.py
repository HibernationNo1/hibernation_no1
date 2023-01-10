from functools import partial
import numpy as np
import random
from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from docker.hibernation_no1.mmdet.data.datacontainer import DataContainer
from docker.hibernation_no1.mmdet.data.sampler import GroupSampler


def worker_init_fn(worker_id, num_workers, seed):
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    worker_seed = num_workers + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    
    
def _build_dataloader(dataset,
                      batch_size,
                      num_workers,
                      seed,
                      shuffle = True):
    if dataset is None: return None
    sampler = GroupSampler(dataset, batch_size) if shuffle else None
    batch_sampler = None
    
    init_fn = partial(worker_init_fn, num_workers=num_workers,seed=seed) if seed is not None else None
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        batch_sampler=batch_sampler,
        collate_fn=partial(collate, samples_per_gpu=batch_size),    
        pin_memory=False,
        worker_init_fn=init_fn) 
    return data_loader



def build_dataloader(num_workers, seed,
                     train_dataset=None, train_batch_size=None,
                     val_dataset=None, val_batch_size=None,
                     shuffle = True):
    train_dataloader = _build_dataloader(train_dataset, train_batch_size, num_workers, seed, shuffle = shuffle)
    val_dataloader = _build_dataloader(val_dataset, val_batch_size, num_workers, seed, shuffle = shuffle)
    
    if train_dataloader is not None and val_dataloader is None: return train_dataloader, None        # only train dataset
    elif val_dataloader is not None and train_dataloader is None: return None, val_dataloader          # only val dataset

    return train_dataloader, val_dataloader



def collate(batch, samples_per_gpu=1):
    """Puts each data field into a tensor/DataContainer with outer dimension batch size.

    batch : len(batch) = batch size
            batch[0] = dict, dict_keys(['img_metas', 'img', 'gt_bboxes', 'gt_labels', 'gt_masks'])
    samples_per_gpu : batch size
    
    Extend default_collate to add support for
    :type:`~mmcv.parallel.DataContainer`. There are 3 cases.

    # 1. cpu_only = True,                   // key: 'gt_masks', 'img_metas'
    # 2. cpu_only = False, stack = True,    // key: 'img', 'gt_semantic_seg'
    # 3. cpu_only = False, stack = False,   // key: 'proposals', 'gt_bboxes', 'gt_bboxes_ignore', 'gt_labels'
    """
    
    if not isinstance(batch, Sequence):
        raise TypeError(f'{batch.dtype} is not supported.')
    if isinstance(batch[0], DataContainer):
        stacked = []        
        if batch[0].cpu_only:  
            # stack by batch_size
            for i in range(0, len(batch), samples_per_gpu):     # samples_per_gpu == batch size
                stacked.append([sample.data for sample in batch[i:i + samples_per_gpu]])
            return DataContainer(stacked, batch[0].stack, batch[0].padding_value, 
                                 cpu_only=True)
        elif batch[0].stack:        # cpu_only = False, stack = True
            # tensor used training
            # >>> expected: 
            #   len(stacked) == 1
            #   stacked[n].shape = [batch size, chennel, height, width]
            for i in range(0, len(batch), samples_per_gpu):
                assert isinstance(batch[i].data, torch.Tensor)

                if batch[i].pad_dims is not None:
                    ndim = batch[i].dim()               # dimension of image
                    assert ndim > batch[i].pad_dims     # pad_dims = 2 (w, h)
                    max_shape = [0 for _ in range(batch[i].pad_dims)]
                  
                    for dim in range(1, batch[i].pad_dims + 1):
                        max_shape[dim - 1] = batch[i].size(-dim)        # max_shape = [w, h]  pedded width, height
                  
                    for sample in batch[i:i + samples_per_gpu]:
                        for dim in range(0, ndim - batch[i].pad_dims):
                            assert batch[i].size(dim) == sample.size(dim)
                        for dim in range(1, batch[i].pad_dims + 1):
                            max_shape[dim - 1] = max(max_shape[dim - 1],
                                                     sample.size(-dim))
                            
                    padded_samples = []                 # expected: len(padded_samples[n]): 3 = chennel
                                                        #   size of each chennel = [height, width] 
                    for sample in batch[i:i + samples_per_gpu]:
                        pad = [0 for _ in range(batch[i].pad_dims * 2)]
                        for dim in range(1, batch[i].pad_dims + 1):
                            pad[2 * dim - 1] = max_shape[dim - 1] - sample.size(-dim)
                        padded_samples.append(F.pad(sample.data, pad, value=sample.padding_value))

                    stacked.append(default_collate(padded_samples))
                elif batch[i].pad_dims is None:
                    stacked.append(
                        default_collate([
                            sample.data
                            for sample in batch[i:i + samples_per_gpu]
                        ]))
                else:
                    raise ValueError(
                        'pad_dims should be either None or integers (1-3)')

        else:       # cpu_only = False, stack = False
            # tensor used for training
            # >> expected: 
            #   len(stacked) == 1
            #   len(stacked[n]) == batchsize
        #   stacked[n].shape = [m, 4],    m: num of instance
            for i in range(0, len(batch), samples_per_gpu):
                stacked.append([sample.data for sample in batch[i:i + samples_per_gpu]])
        return DataContainer(stacked, batch[0].stack, batch[0].padding_value)

        
    elif isinstance(batch[0], Sequence):
        transposed = zip(*batch)
        return [collate(samples, samples_per_gpu) for samples in transposed]
    elif isinstance(batch[0], Mapping):
        return {key: collate([d[key] for d in batch], samples_per_gpu)
                             for key in batch[0] }
    else:
        return default_collate(batch)