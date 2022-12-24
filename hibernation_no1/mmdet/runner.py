
import copy
import time
import torch


from hibernation_no1.mmdet.utils import get_host_info, compute_sec_to_h_d
from hibernation_no1.mmdet.modules.base.runner import BaseRunner
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
    runner = EpochBasedRunner(**runner_cfg)
 
    return runner


        
class EpochBasedRunner(BaseRunner):
    def run(self, train_dataloader, val_dataloader, flow,
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
        self.val_cfg = kwargs.get('val_cfg', None)
        self.mask_to_polygon = kwargs.get('mask_to_polygon', None)
        
        mode, iter = flow
        if not isinstance(mode, str): 
            raise TypeError(f'mode in workflow must be a str, but got {type(mode)}') 
        if not isinstance(iter, int) : 
            raise TypeError(f'epoch in workflow must be a int, but got {type(iter)}') 
                        
        work_dir = self.work_dir
        self.logger.info('Start running, host: %s, work_dir: %s',
                         get_host_info(), work_dir)
        self.logger.info('Hooks will be executed in the following order:\n%s',
                         self.get_hook_info())
        if self._max_epochs is not None:        
            self.logger.info(f'mode: {mode}, max: {self._max_epochs} epochs')
            
            # expected total ite according to the number of epochs set by the user
            self._max_iters = self._max_epochs * len(train_dataloader)            
        else: raise ValueError(f"epoch must be specified in cfg.workflow, but got None.")   # TODO: Training in epochs unit

        self.iterd_per_epochs = len(train_dataloader)
        work_dir = self.work_dir
        
        if not hasattr(self, mode):
            raise ValueError(f'runner has no method named "{mode}" to run an epoch')
        
        self.call_hook('before_run')
        self.start_time = time.time()
        if self._max_epochs is not None:        
            while self.epoch < self._max_epochs:        # Training in epochs unit
                # epoch_runner = getattr(self, mode)      # call method (train, val, eval)
                epoch_runner = self.train
                for _ in range(self._max_epochs):
                    epoch_runner(train_dataloader, val_dataloader, **kwargs)
        else:   # TODO: Training in epochs unit
            
            while self.iter < self._max_iters:
                pass
                
            pass
        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_run')
        
        
    def run_iter(self, data_batch, train_mode):
        if train_mode:
            # MMDataParallel.train_step
            # outputs: 
            # loss:total loss, log_vars: log_vars, num_samples: batch_size
            outputs = self.model.train_step(data_batch, self.optimizer)
        else:   # TODO
            outputs = self.model.val_step(data_batch, self.optimizer)
            
        if not isinstance(outputs, dict):
            raise TypeError('"batch_processor()" or "model.train_step()"'
                            'and "model.val_step()" must return a dict')
        if 'log_vars' in outputs:
            self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])
        self.outputs = outputs
            
        
                
    def train(self, train_dataloader, val_dataloader, **kwargs):
        self.model.train()
        self.mode = 'train'
        self.train_dataloader = train_dataloader
        self.call_hook('before_train_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        
        for i, data_batch in enumerate(self.train_dataloader):
            # data_batch: data of passed by pipelines in dataset and collate train_dataloader
            # data_batch.keys() = ['img_metas', 'img', 'gt_bboxes', 'gt_labels', 'gt_masks']
            self.data_batch = data_batch        
            self._inner_iter = i
            self.call_hook('before_train_iter')
            # self.outputs: 
            # loss:total loss, log_vars: log_vars, num_samples: batch_size
            self.run_iter(data_batch, train_mode=True)
            self.call_hook('after_train_iter')
            
            if self.mode=="val" and self.val_cfg is not None:
                self.val(val_dataloader, **kwargs)
                
            del self.data_batch
            self._iter += 1
        self.call_hook('after_train_epoch')
        self._epoch += 1
        
        
    def val(self, val_dataloader, **kwargs):     
        self.model.eval()
        eval_cfg = dict(model= self.model.eval(), 
                        cfg= self.val_cfg,
                        dataloader= val_dataloader,
                        mask_to_polygon= self.mask_to_polygon)
        eval = Evaluate(**eval_cfg)   
        mAP = eval.compute_mAP()
        datatime = compute_sec_to_h_d(time.time() - self.start_time)
        log_str = f"epoch: [{self.epoch}|{self.max_epochs}],    iter: [{self._inner_iter+1}|{self.iterd_per_epochs}]    "
        log_str +=f"mAP={mAP}       datatime={datatime}\n"
        
        katib_logger = kwargs.get("katib_logger", None)
        if katib_logger is not None:
            katib_logger.logger.info(log_str)
        else:
            print(log_str)
        
        self.mode = "train"
        self.model.train()
        
    
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
        
    
    def register_training_hooks(self,
                                hook_cfg_list,
                                ev_iter):       # len(train_dataloader)

        for hook_cfg in hook_cfg_list:            
            if hook_cfg.get("priority", None) is None: priority = "VERY_LOW"
            else: priority = hook_cfg.priority
            
            if hook_cfg.type == 'LoggerHook': hook_cfg.ev_iter = ev_iter
                
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
        if isinstance(self.optimizer, torch.optim.Optimizer):
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
            meta.update(self.meta)

        if model_cfg is not None:
            meta.update(model_cfg = model_cfg)
            
        meta.update(epoch=self.epoch + 1, 
                    iter=self.iter)
        
   
        filename = filename_tmpl.format(self.epoch + 1)
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
