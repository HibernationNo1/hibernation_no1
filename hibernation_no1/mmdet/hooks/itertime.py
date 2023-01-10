import time

from hibernation_no1.mmdet.hooks.hook import Hook, HOOK

@HOOK.register_module()
class IterTimerHook(Hook):
    def __init__(self, 
                 show_eta_iter):
        self.sum_time_iter = 0
        self.iter_count = 0
        self.show_eta_iter = show_eta_iter
                
        
    def before_epoch(self, runner):
        self.t = time.time()

    def before_iter(self, runner):
        runner.log_buffer.update({'data_time': time.time() - self.t})

    def after_iter(self, runner):
        self.iter_count +=1
        
        taken_time = time.time() - self.t
        runner.log_buffer.update({'time': taken_time})
        
        self.sum_time_iter +=round(taken_time, 2)
        
        if self.every_n_inner_iters(runner, self.show_eta_iter):
            remain_time = self.compute_remain_time(self.sum_time_iter/self.iter_count, runner._max_iters)
            
            # estimated time of arrival
            print(f"eta: [{remain_time}]\
                    epoch: [{runner.epoch}/{runner._max_epochs}]\
                    iter: [{runner._inner_iter}/{runner._iterd_per_epochs}]")
                
        self.t = time.time()

