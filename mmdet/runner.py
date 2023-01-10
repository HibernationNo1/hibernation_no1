import os, os.path as osp
import copy
import time
import logging
import numpy as np
import torch
from torch.optim import Optimizer


from hibernation_no1.mmdet.utils import get_host_info, compute_sec_to_h_d
from hibernation_no1.mmdet.eval import Evaluate

from hibernation_no1.mmdet.registry import build_from_cfg
from hibernation_no1.mmdet.hooks.hook import Hook, HOOK
from hibernation_no1.mmdet.checkpoint import save_checkpoint as sc_save_checkpoint 

priority_dict = {'HIGHEST' : 0,
                 'VERY_HIGH' : 10,
                 'HIGH' : 30,
                 'ABOVE_NORMAL' : 40,
                 'NORMAL' : 50,
                 'BELOW_NORMAL' : 60,
                 'LOW' : 70,
                 'VERY_LOW' : 90,
                 'LOWEST' : 100}


def build_runner(cfg: dict):
    runner_cfg = copy.deepcopy(cfg)
    runner = Runner(**runner_cfg)
 
    return runner


        
class Runner:
    """The base class of Runner, a training helper for PyTorch.
    Args:
        model (:obj:`torch.nn.Module`): The model to be run.
        batch_processor (callable): A callable method that process a data
            batch. The interface of this method should be
            `batch_processor(model, data, train_mode) -> dict`
        optimizer (dict or :obj:`torch.optim.Optimizer`): It can be either an
            optimizer (in most cases) or a dict of optimizers (in models that
            requires more than one optimizer, e.g., GAN).
        work_dir (str, optional): The working directory to save checkpoints
            and logs. Defaults to None.
        logger (:obj:`logging.Logger`): Logger used during training.
             Defaults to None. (The default value is just for backward
             compatibility)
        meta (dict | None): A dict records some import information such as
            environment info and seed, which will be logged in logger hook.
            Defaults to None.
        max_epochs (int, optional): Total training epochs.
        max_iters (int, optional): Total training iterations.
    """
    def __init__(self,
                 model,
                 iterd_per_epochs,
                 max_iters=None,
                 max_epochs=None,
                 optimizer=None,
                 work_dir=None,
                 logger=None,
                 meta=None,
                 **kwargs):
       
        
        assert hasattr(model, 'train_step')
        
        
        # check the type of `optimizer`
        if isinstance(optimizer, dict):
            for name, optim in optimizer.items():
                if not isinstance(optim, Optimizer):
                    raise TypeError(
                        f'optimizer must be a dict of torch.optim.Optimizers, '
                        f'but optimizer["{name}"] is a {type(optim)}')
        elif not isinstance(optimizer, Optimizer) and optimizer is not None:
            raise TypeError(
                f'optimizer must be a torch.optim.Optimizer object '
                f'or dict or None, but got {type(optimizer)}')
            
        # check the type of `logger`
        if not isinstance(logger, logging.Logger):
            raise TypeError(f'logger must be a logging.Logger object, '
                            f'but got {type(logger)}')

        # check the type of `meta`
        if meta is not None and not isinstance(meta, dict):
            raise TypeError(
                f'meta must be a dict or None, but got {type(meta)}')
        
        self.batch_size = kwargs.get('batch_size', None)
         
        self.model = model
    
        self.optimizer = optimizer
        self.logger = logger
        self.meta = meta
        self.work_dir = work_dir
        if work_dir is None: raise TypeError(f"work_dir must be specific, but work_dir is 'None'") 
        if not os.path.isdir(work_dir): os.makedirs(work_dir, exist_ok=True)
        
        # get model name from the model class
        if hasattr(self.model, 'module'):
            self._model_name = self.model.module.__class__.__name__
        else:
            self._model_name = self.model.__class__.__name__
        
        self._rank, self._world_size = 0, 1
        self.timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        self.val_result = []    # will be appended when run Validation_Hook
        self._hooks = []    
        self._epoch = 1         # current epoch during training
        self._iter = 1          # current iter during training
        self._inner_iter = 0    # current iter during epoch
        
        self._iterd_per_epochs = iterd_per_epochs
        if max_epochs is not None and max_iters is not None:
            raise ValueError(
                'Only one of `max_epochs` or `max_iters` can be set.')
        self._max_epochs = max_epochs 
        self._max_iters = max_iters
        
        if self._max_epochs is not None:     
            self._by_epoch = True
            # expected total iter according to the number of epochs set by the user
            self._max_iters = self._max_epochs * self._iterd_per_epochs            
        else: 
            self._by_epoch = False
            self._max_epochs = self._max_iters // self._iterd_per_epochs
            raise ValueError(f"training in iteration units is not yet implemented.")
  
        
        self.log_buffer = LogBuffer()
        
        
        
        
    def run(self, 
            train_dataloader, 
            **kwargs):
        """Start running.

        Args:
            data_loaders (list[:obj:`DataLoader`]): Dataloaders for training
                and validation.
            workflow (list[tuple]): A list of (phase, epochs) to specify the
                running order and epochs. E.g, [('train', 2), ('val', 1)] means
                running 2 epochs for training and 1 epoch for validation,
                iteratively.
        """
        if self._by_epoch: self.logger.info(f'Training in epoch units.   max epochs: {self._max_epochs}')
        else: self.logger.info(f'Training in iteration units.   max iters: {self._max_iters}')       
        self.logger.info(f'Start running, host: {get_host_info()}, work_dir: {self.work_dir}')
        self.logger.info(f'Hooks will be executed in the following order:\n{self.get_hook_info()}')
        
        
        self.call_hook('before_run')
        self.start_time = time.time()
        if self._by_epoch:        
            while self._epoch < self._max_epochs + 1:        # Training in epochs unit
                self.train(train_dataloader, **kwargs)
        else:   
            while self._iter < self._max_iters + 1:
                # self.train(train_dataloader, **kwargs)
                
                pass
        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_run')
        
        
    def run_iter(self, data_batch):

        # MMDataParallel.train_step
        # outputs: 
        # loss:total loss, log_vars: log_vars, num_samples: batch_size
        outputs = self.model.train_step(data_batch, self.optimizer)
      
            
        if not isinstance(outputs, dict):
            raise TypeError('"batch_processor()" or "model.train_step()"'
                            'and "model.val_step()" must return a dict')        # delete val_step
        if 'log_vars' in outputs:
            self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])
        
        self.outputs = outputs
            
        
                
    def train(self, train_dataloader, **kwargs):
        self.model.train()
        self.train_dataloader = train_dataloader
        self.call_hook('before_train_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        
        for i, data_batch in enumerate(self.train_dataloader):
            # data_batch: data of passed by pipelines in dataset and collate train_dataloader
            # data_batch.keys() = ['img_metas', 'img', 'gt_bboxes', 'gt_labels', 'gt_masks']
            self.data_batch = data_batch        
            self._inner_iter = i+1
            self.call_hook('before_train_iter')
            # self.outputs: 
            # loss:total loss, log_vars: log_vars, num_samples: batch_size
            self.run_iter(data_batch)
            self.call_hook('after_train_iter')

            del self.data_batch
            self._iter += 1
        self.call_hook('after_train_epoch')
        self._epoch += 1
            
    
    def call_hook(self, fn_name):
        """Call all hooks.

        Args:
            fn_name (str): The function name in each hook to be called
                "before_run"
                "before_train_epoch"
                "before_train_iter"
                "after_train_iter"
                "after_train_epoch"
                "before_val_epoch"
                "before_val_iter"
                "after_val_iter"
                "after_run"
                
        """
        for hook in self._hooks:
            getattr(hook, fn_name)(self)   
       
    def register_hook(self, hook, priority='NORMAL'):
        """Register a hook into the hook list.

        The hook will be inserted into a priority queue, with the specified
        priority (See :class:`Priority` for details of priorities).
        For hooks with the same priority, they will be triggered in the same
        order as they are registered.

        Args:
            hook (:obj:`Hook`): The hook to be registered.
            priority (int or str or :obj:`Priority`): Hook priority.
                Lower value means higher priority.
        """
        assert isinstance(hook, Hook)
        if hasattr(hook, 'priority'):
            raise ValueError('"priority" is a reserved attribute for hooks')

        # priority setting
        for key in list(priority_dict.keys()):
            if priority == key : 
                priority = priority_dict[key]
                priority_dict[f'{priority}'] = key
                hook.priority = priority
                break
   
        # insert the hook to a sorted list
        inserted = False
        for i in range(len(self._hooks) - 1, -1, -1):
            if priority >= self._hooks[i].priority:
                self._hooks.insert(i + 1, hook)
                inserted = True
                break
        if not inserted:
            self._hooks.insert(0, hook)
         
    
    def register_training_hooks(self, hook_cfg_list):
        for hook_cfg in hook_cfg_list:            
            if hook_cfg.get("priority", None) is None: priority = "VERY_LOW"
            else: priority = hook_cfg.pop("priority")
            
            hook = build_from_cfg(hook_cfg, HOOK)
            self.register_hook(hook, priority=priority)
  
        
      
    def get_hook_info(self):
        # Get hooks info in each stage
        stage_hook_map = {stage: [] for stage in Hook.stages}
        for hook in self.hooks:
            try:
                priority = priority_dict[f'{hook.priority}']
            except ValueError:
                priority = hook.priority
            classname = hook.__class__.__name__
            hook_info = f'({priority:<12}) {classname:<35}'
            for trigger_stage in hook.get_triggered_stages():
                stage_hook_map[trigger_stage].append(hook_info)

 
        stage_hook_infos = []
        for stage in Hook.stages:
            hook_infos = stage_hook_map[stage]
            if len(hook_infos) > 0:
                info = f'{stage}:\n'
                info += '\n'.join(hook_infos)
                info += '\n -------------------- '
                stage_hook_infos.append(info)
        return '\n'.join(stage_hook_infos)  
    
    
       
    def current_lr(self):
        """Get current learning rates.

        Returns:
            list[float] | dict[str, list[float]]: Current learning rates of all
            param groups. If the runner has a dict of optimizers, this method
            will return a dict.
        """
        if isinstance(self.optimizer, torch.optim.Optimizer):           # 
            lr = [group['lr'] for group in self.optimizer.param_groups]
        elif isinstance(self.optimizer, dict):
            lr = dict()
            for name, optim in self.optimizer.items():
                lr[name] = [group['lr'] for group in optim.param_groups]
        else:
            raise RuntimeError(
                'lr is not applicable because optimizer does not exist.')
      
        return lr

    def current_momentum(self):
        """Get current momentums.

        Returns:
            list[float] | dict[str, list[float]]: Current momentums of all
            param groups. If the runner has a dict of optimizers, this method
            will return a dict.
        """

        def _get_momentum(optimizer):
            momentums = []
            for group in optimizer.param_groups:
                if 'momentum' in group.keys():
                    momentums.append(group['momentum'])
                elif 'betas' in group.keys():
                    momentums.append(group['betas'][0])
                else:
                    momentums.append(0)
            return momentums

        if self.optimizer is None:
            raise RuntimeError(
                'momentum is not applicable because optimizer does not exist.')
        elif isinstance(self.optimizer, torch.optim.Optimizer):
            momentums = _get_momentum(self.optimizer)
        elif isinstance(self.optimizer, dict):
            momentums = dict()
            for name, optim in self.optimizer.items():
                momentums[name] = _get_momentum(optim)
        return momentums
    
    
    def save_checkpoint(self,
                        out_dir,
                        filename_tmpl='epoch_{}.pth',
                        save_optimizer=True,
                        meta=None,
                        model_cfg =None):
        """Save the checkpoint.

        Args:
            out_dir (str): The directory that checkpoints are saved.
            filename_tmpl (str, optional): The checkpoint filename template,
                which contains a placeholder for the epoch number.
                Defaults to 'epoch_{}.pth'.
            save_optimizer (bool, optional): Whether to save the optimizer to
                the checkpoint. Defaults to True.
            meta (dict, optional): The meta information to be saved in the
                checkpoint. Defaults to None.
        """
        if meta is None:
            meta = {}
        elif not isinstance(meta, dict):
            raise TypeError(
                f'meta should be a dict or None, but got {type(meta)}')
        
        
        if self.meta is not None:
            runner_meta = self.meta.copy()
            runner_meta.pop("config")
            meta.update(runner_meta)

        if model_cfg is not None:
            meta.update(model_cfg = model_cfg)
            
        meta.update(epoch=self._epoch, 
                    iter=self._iter)
        
   
        filename = filename_tmpl.format(self._epoch)
        filepath = osp.join(out_dir, filename)
        optimizer = self.optimizer if save_optimizer else None
        
        sc_save_checkpoint(self.model, filepath, optimizer=optimizer, meta=meta)
    
    
    def get(self, att_name: str):
        try:
            return getattr(self, att_name)
        except:
            return None
            # raise AttributeError(f"{self.__class__.__name__} object has no attribute {att_name}")
            
                  
    @property
    def model_name(self):
        """str: Name of the model, usually the module class name."""
        return self._model_name

    @property
    def rank(self):
        """int: Rank of current process. (distributed training)"""
        return self._rank

    @property
    def world_size(self):
        """int: Number of processes participating in the job.
        (distributed training)"""
        return self._world_size

    @property
    def hooks(self):
        """list[:obj:`Hook`]: A list of registered hooks."""
        return self._hooks

    @property
    def epoch(self):
        """int: Current epoch."""
        return self._epoch

    @property
    def iter(self):
        """int: Current iteration."""
        return self._iter

    @property
    def inner_iter(self):
        """int: Iteration in an epoch."""
        return self._inner_iter

    @property
    def max_epochs(self):
        """int: Maximum training epochs."""
        return self._max_epochs

    @property
    def max_iters(self):
        """int: Maximum training iterations."""
        return self._max_iters                 



class LogBuffer:

    def __init__(self):
        self.val_history = dict()
        self.n_history = dict()
        self.output = dict()
        self.log_output = dict()
        self.ready = False

    def clear(self) -> None:
        self.val_history.clear()
        self.n_history.clear()
        self.clear_output()

    def clear_output(self) -> None:
        self.output.clear()
        self.ready = False

    def update(self, vars: dict, count: int = 1) -> None:
        assert isinstance(vars, dict)
        for key, var in vars.items():
            if key not in self.val_history:
                self.val_history[key] = []
                self.n_history[key] = []
            self.val_history[key].append(var)
            self.n_history[key].append(count)

    def average(self, n: int = 0) -> None:
        """Average latest n values or all values."""
        assert n >= 0
        for key in self.val_history:
            values = np.array(self.val_history[key][-n:])
            nums = np.array(self.n_history[key][-n:])
            avg = np.sum(values * nums) / np.sum(nums)
            self.output[key] = avg
        self.ready = True
        
    
    def log(self, n):
        for key in self.val_history:
            values = np.array(self.val_history[key][-n:])
            nums = np.array(self.n_history[key][-n:])
            avg = np.sum(values * nums) / np.sum(nums)
            self.log_output[key] = avg
            
            
    def get_last(self):
        output = dict()
        for key in self.val_history:
            output[key] = self.val_history[key][-1]
        return output
    
    def clear_log(self):
        self.log_output.clear()
        
        