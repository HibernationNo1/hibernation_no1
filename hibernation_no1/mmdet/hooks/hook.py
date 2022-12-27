from hibernation_no1.mmdet.registry import Registry

HOOK = Registry('hook')

def is_method_overridden(method, base_class, derived_class):
    """Check if a method of base class is overridden in derived class.

    Args:
        method (str): the method name to check.
        base_class (type): the class of the base class.
        derived_class (type | Any): the class or instance of the derived class.
    """
    assert isinstance(base_class, type), \
        "base_class doesn't accept instance, Please pass class instead."

    if not isinstance(derived_class, type):
        derived_class = derived_class.__class__

    base_method = getattr(base_class, method)
    derived_method = getattr(derived_class, method)
    return derived_method != base_method



class Hook:
    stages = ('before_run', 'before_train_epoch', 'before_train_iter',
              'after_train_iter', 'after_train_epoch', 'before_val_epoch',
              'before_val_iter', 'after_val_iter', 'after_val_epoch',
              'after_run')        

    def before_run(self, runner):
        pass

    def after_run(self, runner):
        pass

    def before_epoch(self, runner):
        pass

    def after_epoch(self, runner):
        pass

    def before_iter(self, runner):
        pass

    def after_iter(self, runner):
        pass

    def before_train_epoch(self, runner):
        self.before_epoch(runner)

    def before_val_epoch(self, runner):
        self.before_epoch(runner)

    def after_train_epoch(self, runner):
        self.after_epoch(runner)

    def after_val_epoch(self, runner):
        self.after_epoch(runner)

    def before_train_iter(self, runner):
        self.before_iter(runner)

    def before_val_iter(self, runner):
        self.before_iter(runner)

    def after_train_iter(self, runner):
        self.after_iter(runner)

    def after_val_iter(self, runner):
        self.after_iter(runner)

    def every_n_epochs(self, runner, n):
        return (runner.epoch) % n == 0 if n > 0 else False

    def every_n_inner_iters(self, runner, n):
        return (runner.inner_iter) % n == 0 if n > 0 else False

    def every_n_iters(self, runner, n):
        return (runner.iter) % n == 0 if n > 0 else False

    def end_of_epoch(self, runner):
        return runner.inner_iter == len(runner.train_dataloader)

    def is_last_epoch(self, runner):
        return runner.epoch == runner._max_epochs

    def is_last_iter(self, runner):
        return runner.iter == runner._max_iters

    def get_triggered_stages(self):
        """
            choosing hook to activate at a specific stage 
            stage : before_run              
                    before_train_epoch      
                    before_train_iter       
                    after_train_iter        
                    after_train_epoch        
                    before_val_epoch        
                    before_val_iter          
                    after_val_iter
                    after_val_epoch
                    after_run               
        """
        trigger_stages = set()
        for stage in Hook.stages:
            if is_method_overridden(stage, Hook, self):
                trigger_stages.add(stage)


        
        # some methods will be triggered in multi stages
        # use this dict to map method to stages.
        method_stages_map = {
            'before_epoch': ['before_train_epoch', 'before_val_epoch'],
            'after_epoch': ['after_train_epoch', 'after_val_epoch'],
            'before_iter': ['before_train_iter', 'before_val_iter'],
            'after_iter': ['after_train_iter', 'after_val_iter'],
        }
        for method, map_stages in method_stages_map.items():
            if is_method_overridden(method, Hook, self):
                trigger_stages.update(map_stages)

        
        return [stage for stage in Hook.stages if stage in trigger_stages]
    
    
    def compute_remain_time(self, taken_time, max_iters):
        remain_iter = max_iters - self.iter_count
        return self.compute_sec_to_h_d(taken_time * remain_iter)           
        
    
    def get_epoch(self, runner):
        if runner.mode == 'train':
            epoch = runner.epoch
        elif runner.mode == 'val':
            # normal val mode
            # runner.epoch += 1 has been done before val workflow
            epoch = runner.epoch
        else:
            raise ValueError(f"runner mode should be 'train' or 'val', "
                             f'but got {runner.mode}')
        return epoch
    
    
    def compute_sec_to_h_d(self, sec):
        if sec <=0: return "00:00:00"
        
        if sec < 60: return f'00:00:{f"{int(sec)}".zfill(2)}'
        
        minute = sec//60
        if minute < 60: return f"00:{f'{int(minute)}'.zfill(2)}:{f'{int(sec%60)}'.zfill(2)}"
        
        hour = minute//60
        if hour < 24: return f"{f'{int(hour)}'.zfill(2)}:{f'{int(minute%60)}'.zfill(2)}:{f'{int(sec%60)}'.zfill(2)}"
        
        day = hour//24
        return f"{day}day {f'{int(hour%24)}'.zfill(2)}:{f'{int(minute%(60))}'.zfill(2)}:{f'{int(sec%(60))}'.zfill(2)}"