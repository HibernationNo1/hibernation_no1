import time

from hibernation_no1.mmdet.hooks.hook import Hook, HOOK

@HOOK.register_module()
class Validation_Hook(Hook):
    # TODO:
    def __init__(self,
                 show_eta_iter):
        self.iter_count = 1
        self.sum_time_iter = 0
        self.show_eta_iter = show_eta_iter
        
    
    def before_train_iter(self, runner):          
        self.i_t = time.time()      # iter start time 
        
        
    def after_train_iter(self, runner) -> None:   
        self.sum_time_iter +=round(time.time() - self.i_t, 2)
        self.iter_count+=1
        
        if self.every_n_inner_iters(runner, self.val_iter):   # training unit: epoch, iter +1
            runner.mode = 'val'     # change runner mode to val for run validation 
        if self.every_n_inner_iters(runner, self.show_eta_iter):  
            remain_time = self.compute_remain_time(self.sum_time_iter/self.iter_count)['remain_time']     
            
            # estimated time of arrival
            print(f"eta: [{remain_time}]\
                    epoch: [{runner.epoch+1}/{self.max_epochs}]\
                    iter: [{self.get_iter(runner, inner_iter=True)}/{self.ev_iter}]")


@HOOK.register_module()
class Check_Hook(Hook):
    def __init__(self, 
                 show_eta_iter, 
                 val_iter, 
                 max_epochs,
                 ev_iter,        # iters_per_epochs          
                 by_epoch: bool = True):  
        self.iter_count = 1
        self.sum_time_iter = 0
        self.val_iter = val_iter
        self.max_epochs = max_epochs 
        self.ev_iter = ev_iter
        self.by_epoch = by_epoch
        self.show_eta_iter = show_eta_iter
 
            
    
    def before_val_epoch(self, runner):
        """Check whether the dataset in val epoch is compatible with head.

        Args:
            runner (obj:`EpochBasedRunner`): Epoch based Runner.
        """
        self._check_head(runner)
        
    
    def before_train_epoch(self, runner):
        """Check whether the training dataset is compatible with head.

        Args:
            runner (obj:`EpochBasedRunner`): Epoch based Runner.
        """
        self._check_head(runner)
        
        
    def _check_head(self, runner):
        """Check whether the `num_classes` in head matches the length of
        `CLASSES` in `dataset`.

        Args:
            runner (obj:`EpochBasedRunner`): Epoch based Runner.
        """
      
        model = runner.model
        dataset = runner.train_dataloader.dataset
        
        if dataset.CLASSES is None:
            runner.logger.warning(
                f'Please set `CLASSES` '
                f'in the {dataset.__class__.__name__} and'
                f'check if it is consistent with the `num_classes` '
                f'of head')
        
        else:
            assert type(dataset.CLASSES) is not str, \
                (f'`CLASSES` in {dataset.__class__.__name__}'
                 f'should be a tuple of str.'
                 f'Add comma if number of classes is 1 as '
                 f'CLASSES = ({dataset.CLASSES},)')
                
            for name, module in model.named_modules():
                # Check something important at each head before run train. 
                # exam)
                    # if hasattr(module, 'num_classes') and not isinstance(module, RPNHead):
                    #     assert module.num_classes == len(dataset.CLASSES), \
                    #         (f'The `num_classes` ({module.num_classes}) in '
                    #          f'{module.__class__.__name__} of '
                    #          f'{model.__class__.__name__} does not matches '
                    #          f'the length of `CLASSES` '
                    #          f'{len(dataset.CLASSES)}) in '
                    #          f'{dataset.__class__.__name__}')
                pass