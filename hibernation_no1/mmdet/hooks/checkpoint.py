
import os, os.path as osp

from docker.hibernation_no1.mmdet.hooks.hook import Hook, HOOK

@HOOK.register_module()
class CheckpointHook(Hook):
    """Save checkpoints periodically.

    Args:
        interval (int): The saving period. If ``by_epoch=True``, interval
            indicates epochs, otherwise it indicates iterations.
            Default: -1, which means "never".
        by_epoch (bool): Saving checkpoints by epoch or by iteration.
            Default: True.
        save_optimizer (bool): Whether to save optimizer state_dict in the
            checkpoint. It is usually used for resuming experiments.
            Default: True.
        out_dir (str, optional): The root directory to save checkpoints. If not
            specified, ``runner.work_dir`` will be used by default. If
            specified, the ``out_dir`` will be the concatenation of ``out_dir``
            and the last level directory of ``runner.work_dir``.
            `Changed in version 1.3.16.`
        max_keep_ckpts (int, optional): The maximum checkpoints to keep.
            In some cases we want only the latest few checkpoints and would
            like to delete old ones to save the disk space.
            Default: -1, which means unlimited.
        save_last (bool, optional): Whether to force the last checkpoint to be
            saved regardless of interval. Default: True.
        sync_buffer (bool, optional): Whether to synchronize buffers in
            different gpus. Default: False.
        file_client_args (dict, optional): Arguments to instantiate a
            FileClient. See :class:`mmcv.fileio.FileClient` for details.
            Default: None.
            `New in version 1.3.16.`

    .. warning::
        Before v1.3.16, the ``out_dir`` argument indicates the path where the
        checkpoint is stored. However, since v1.3.16, ``out_dir`` indicates the
        root directory and the final path to save checkpoint is the
        concatenation of ``out_dir`` and the last level directory of
        ``runner.work_dir``. Suppose the value of ``out_dir`` is "/path/of/A"
        and the value of ``runner.work_dir`` is "/path/of/B", then the final
        path will be "/path/of/A/B".
    """

    def __init__(self,
                 interval=['epoch', 1],           # Target unit of epoch(or iter) performing CheckpointHook
                 by_epoch=True,         # True : run CheckpointHook unit by epoch , False : unit by iter
                 save_optimizer=True,
                 out_dir=None,
                 max_keep_ckpts=-1,     # number of model which save by .pth format  
                                        # why?: when save model whevery epoch(or iter), the number increases.
                                        # so, delete number from 'last epoch' to 'max_keep_ckpts'
                 save_last=True,
                 filename_tmpl = None,
                 **kwargs):              # ['model_cfg', 'meta']
        self.unit, self.interval = interval[0], interval[1]
        self.by_epoch = by_epoch    
        self.save_optimizer = save_optimizer
        self.out_dir = out_dir      # declaration required to replace `runner.work_dir`
        self.max_keep_ckpts = max_keep_ckpts
        self.save_last = save_last

        self.filename_tmpl = filename_tmpl
        self.set_filename_tmpl()
        self.model_cfg = kwargs.get("model_cfg", None)
        self.meta = kwargs.get("meta", None)
       
                    
                
        
    
    def set_filename_tmpl(self):
        filename = self.filename_tmpl
        if filename is None: 
            if self.by_epoch: self.filename_tmpl = 'epoch_{}.pth'
            else: self.filename_tmpl = 'iter_{}.pth'
        else:
            assert len(filename.split('.')) < 3, "wrong model name. \
                                                        \ncheck : cfg.checkpoint_config.filename_tmpl \
                                                        \ncurrent: {cfg.checkpoint_config.filename_tmpl}"
            if len(filename.split('.')) == 1: 
                filename += "_{}.pth"
            elif len(filename.split('.')) == 2:                    
                if filename.split('.')[-1] != "pth": 
                    filename = filename.split('.')[0] + ".pth"
        
        self.filename_tmpl = filename
            
        
    def before_run(self, runner):   
        if not self.out_dir:
            self.out_dir = runner.work_dir
        
        runner.logger.info(f'Checkpoints will be saved to {self.out_dir} ')
        
        # self.args['create_symlink'] = True
        
    def after_train_epoch(self, runner):
        # run whenever epoch is a multiple of self.interval(default: 1)
        if self.unit == "epoch" and self.every_n_epochs(runner, self.interval) or \
            self.save_last and self.is_last_epoch(runner):
            runner.logger.info(f'Saving checkpoint at {runner.epoch} epochs')
            self.save_checkpoint_hook(runner)
        elif self.unit == "iter" and self.every_n_inner_iters(runner, self.interval):
            # TODO: check 
            runner.logger.info(f'Saving checkpoint at {runner.iter} iters')
            self.save_checkpoint_hook(runner)
        
        if self.save_last and self.is_last_epoch(runner):
            self.save_checkpoint_hook(runner)
       
  
            
    def save_checkpoint_hook(self, runner):
        """Save the current checkpoint and delete unwanted checkpoint."""
        # save meta, parameters of model, optimazers 
        checkpoint_cfg = dict(out_dir = self.out_dir,
                              filename_tmpl = self.filename_tmpl,
                              save_optimizer = self.save_optimizer,
                              model_cfg = self.model_cfg,
                              meta = self.meta) 
        runner.save_checkpoint(**checkpoint_cfg)
        
        if runner.meta is not None:
            if self.unit == "iter":
                cur_ckpt_filename = self.filename_tmpl.format(runner.iter)
            else: 
                cur_ckpt_filename = self.filename_tmpl.format(runner.epoch)
            
            runner.meta.setdefault('hook_msgs', dict())
            runner.meta['hook_msgs']['last_ckpt'] = osp.join(self.out_dir, cur_ckpt_filename)

       
        # remove other checkpoints      # do not 
        if self.max_keep_ckpts > 0:
            if self.by_epoch:
                current_ckpt = runner.epoch
            else:
                current_ckpt = runner.iter
      
            
            redundant_ckpts = range(
                current_ckpt - self.max_keep_ckpts * self.interval, 0,
                -self.interval)
            
            for _step in redundant_ckpts: 
                ckpt_path = osp.join(self.out_dir, self.filename_tmpl.format(_step))
                if osp.isfile(ckpt_path):
                    os.remove(ckpt_path)
                else:
                    break
    
    
    def after_train_iter(self, runner):
        if self.by_epoch:
            return

        # save checkpoint for following cases:
        # 1. every ``self.interval`` iterations
        # 2. reach the last iteration of training
        if self.every_n_iters(
                runner, self.interval) or (self.save_last
                                           and self.is_last_iter(runner)):
            runner.logger.info(f'Saving checkpoint at {runner.iter} iterations')
        
            self.save_checkpoint_hook(runner)