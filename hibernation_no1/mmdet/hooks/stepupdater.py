from typing import List, Optional, Union
 
from hibernation_no1.mmdet.hooks.hook import Hook, HOOK

@HOOK.register_module()
class StepLrUpdaterHook(Hook):
    
    def __init__(self,
                 step: Union[int, List[int]],
                 gamma: float = 0.1,            # constance used to decay learning rate
                 min_lr: Optional[float] = None,
                 warmup: Optional[str] = None,
                 warmup_iters: int = 0,
                 warmup_ratio: float = 0.1,
                 warmup_by_epoch: bool = False  # TODO : apply warmup
                 ) -> None:
        if isinstance(step, list):
            for s in step:
                assert isinstance(s, int)
            assert all([s > 0 for s in step])
        elif isinstance(step, int):
            assert step > 0
        else:
            raise TypeError('"step" must be a list or integer')
        self.step = step
        self.gamma = gamma      
        self.min_lr = min_lr
        
        # validate the "warmup" argument
        if warmup is not None:
            if warmup not in ['constant', 'linear', 'exp']:
                raise ValueError(
                    f'"{warmup}" is not a supported type for warming up, valid'
                    ' types are "constant", "linear" and "exp"')
            assert warmup_iters > 0, \
                '"warmup_iters" must be a positive integer'
            assert 0 < warmup_ratio <= 1.0, \
                '"warmup_ratio" must be in range (0,1]'
        
        
        self.warmup = warmup
        self.warmup_iters: Optional[int] = warmup_iters
        self.warmup_ratio = warmup_ratio
        self.warmup_by_epoch = warmup_by_epoch
   
        if self.warmup_by_epoch:
            self.warmup_epochs: Optional[int] = self.warmup_iters
            self.warmup_iters = None
        else:
            self.warmup_epochs = None
            
        self.base_lr: Union[list, dict] = []  # initial lr for all param groups
        self.regular_lr: list = []  # expected lr if no warming up is performed
        

    def get_lr(self, runner , base_lr: float):
        progress = runner.epoch
        
        # calculate exponential term
        if isinstance(self.step, int):
            exp = progress // self.step
        else:
            exp = len(self.step)
            for i, s in enumerate(self.step):
                if progress < s:
                    exp = i
                    break

        lr = base_lr * (self.gamma**exp)
        if self.min_lr is not None:
            # clip to a minimum value
            lr = max(lr, self.min_lr)
        return lr
    
    
    def get_regular_lr(self, runner):
        return [self.get_lr(runner, _base_lr) for _base_lr in self.base_lr]
        
        
    def _set_lr(self, runner, lr_groups):
        if isinstance(runner.optimizer, dict):
            for k, optim in runner.optimizer.items():
                for param_group, lr in zip(optim.param_groups, lr_groups[k]):
                    param_group['lr'] = lr
        else:
            for param_group, lr in zip(runner.optimizer.param_groups,
                                    lr_groups):
                param_group['lr'] = lr
    
    

    def before_run(self, runner):
        # NOTE: when resuming from a checkpoint, if 'initial_lr' is not saved,
        # it will be set according to the optimizer params

        # learning rate to apply to each group
        for group in runner.optimizer.param_groups:
            group.setdefault('initial_lr', group['lr'])
        self.base_lr = [group['initial_lr'] for group in runner.optimizer.param_groups]
        
    
    def before_train_epoch(self, runner):
        if self.warmup_iters is None:
            epoch_len = len(runner.train_dataloader)  # type: ignore
            self.warmup_iters = self.warmup_epochs * epoch_len  # type: ignore

        self.regular_lr = self.get_regular_lr(runner)   # fixed(or not) learning rate
        self._set_lr(runner, self.regular_lr)           # apply learning rete to each parameter group
    
    
    def before_train_iter(self, runner):
        cur_iter = runner.iter

        if self.warmup is None or cur_iter > self.warmup_iters:
            return
        elif cur_iter == self.warmup_iters:
            self._set_lr(runner, self.regular_lr)
        else:
            warmup_lr = self.get_warmup_lr(cur_iter)
            self._set_lr(runner, warmup_lr)
    
    
    def get_warmup_lr(self, cur_iters: int):
        # apply specific value computed by warmup type to learning rate
        def _get_warmup_lr(cur_iters, regular_lr):
            if self.warmup == 'constant':
                warmup_lr = [_lr * self.warmup_ratio for _lr in regular_lr]
            elif self.warmup == 'linear':
                k = (1 - cur_iters / self.warmup_iters) * (1 - self.warmup_ratio)
                warmup_lr = [_lr * (1 - k) for _lr in regular_lr]
            elif self.warmup == 'exp':
                k = self.warmup_ratio**(1 - cur_iters / self.warmup_iters)
                warmup_lr = [_lr * k for _lr in regular_lr]
            return warmup_lr

        if isinstance(self.regular_lr, dict):
            lr_groups = {}
            for key, regular_lr in self.regular_lr.items():
                lr_groups[key] = _get_warmup_lr(cur_iters, regular_lr)
            return lr_groups
        else:
            return _get_warmup_lr(cur_iters, self.regular_lr)