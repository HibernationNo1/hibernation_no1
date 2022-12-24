
import os, os.path as osp

from hibernation_no1.mmdet.hooks.hook import Hook, HOOK

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
                 interval=-1,           # Target unit of epoch(or iter) performing CheckpointHook
                 by_epoch=True,         # True : run CheckpointHook unit by epoch , False : unit by iter
                 save_optimizer=True,
                 out_dir=None,
                 max_keep_ckpts=-1,     # number of model which save by .pth format  
                                        # why?: when save model whevery epoch(or iter), the number increases.
                                        # so, delete number from 'last epoch' to 'max_keep_ckpts'
                 save_last=True,
                 file_client_args=None,
                 **kwargs):             # ['filename_tmpl', 'meta']
        self.interval = interval
        self.by_epoch = by_epoch    
        self.save_optimizer = save_optimizer
        self.out_dir = out_dir      # declaration required to replace `runner.work_dir`
        self.max_keep_ckpts = max_keep_ckpts
        self.save_last = save_last
        self.args = kwargs
        self.file_client_args = file_client_args
        
        
        
    def before_run(self, runner):   
        if not self.out_dir:
            self.out_dir = runner.work_dir
        
        runner.logger.info(f'Checkpoints will be saved to {self.out_dir} ')
        
        # self.args['create_symlink'] = True
        
    def after_train_epoch(self, runner):
        if not self.by_epoch:
            return

       
        # run whenever epoch is a multiple of self.interval(default: 1)
        if self.every_n_epochs(runner, self.interval) \
                           or (self.save_last and self.is_last_epoch(runner)):
                               
            runner.logger.info(f'Saving checkpoint at {runner.epoch + 1} epochs')
            self._save_checkpoint(runner)
            
            
    def _save_checkpoint(self, runner):
        """Save the current checkpoint and delete unwanted checkpoint."""
        # save meta, parameters of model, optimazers 
        runner.save_checkpoint(self.out_dir, save_optimizer=self.save_optimizer, **self.args)
        
        if runner.meta is not None:
            if self.by_epoch:
                cur_ckpt_filename = self.args.get('filename_tmpl', 'epoch_{}.pth').format(runner.epoch + 1)
            else:
                cur_ckpt_filename = self.args.get(
                    'filename_tmpl', 'iter_{}.pth').format(runner.iter + 1)

            runner.meta.setdefault('hook_msgs', dict())
            runner.meta['hook_msgs']['last_ckpt'] = osp.join(self.out_dir, cur_ckpt_filename)

       
        # remove other checkpoints      # do not 
        if self.max_keep_ckpts > 0:
            if self.by_epoch:
                name = 'epoch_{}.pth'
                current_ckpt = runner.epoch + 1
            else:
                name = 'iter_{}.pth'
                current_ckpt = runner.iter + 1
            filename_tmpl = self.args.get('filename_tmpl', name)
            
            redundant_ckpts = range(
                current_ckpt - self.max_keep_ckpts * self.interval, 0,
                -self.interval)
            
            for _step in redundant_ckpts: 
                ckpt_path = osp.join(self.out_dir, filename_tmpl.format(_step))
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
            runner.logger.info(f'Saving checkpoint at {runner.iter + 1} iterations')
        
            self._save_checkpoint(runner)