import time

from hibernation_no1.mmdet.hooks.hook import Hook, HOOK

@HOOK.register_module()
class IterTimerHook(Hook):
    def __init__(config):
        pass
    
    def before_epoch(self, runner):
        self.t = time.time()

    def before_iter(self, runner):
        runner.log_buffer.update({'data_time': time.time() - self.t})

    def after_iter(self, runner):
        runner.log_buffer.update({'time': time.time() - self.t})
        self.t = time.time()