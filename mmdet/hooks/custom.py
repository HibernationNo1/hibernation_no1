import time
from torch.utils.data import DataLoader

from hibernation_no1.mmdet.hooks.hook import Hook, HOOK
from hibernation_no1.mmdet.eval import Evaluate
from torch.utils.tensorboard import SummaryWriter

@HOOK.register_module()
class Validation_Hook(Hook):
    def __init__(self,
                 val_dataloader: DataLoader,
                 val_cfg,
                 logger, 
                 interval = ['iter', 50]
                ):
        self.iter_count = 1
        self.unit, self.val_timing = interval[0], interval[1]
        self.val_dataloader = val_dataloader
        self.val_cfg = val_cfg
        self.logger = logger
   
        
    def after_train_iter(self, runner) -> None:           
        if self.unit == 'iter' and\
            self.every_n_inner_iters(runner, self.val_timing):
            
            result = self.validation(runner)
            runner.val_result.append(result)
        
    
    def after_train_epoch(self, runner) -> None: 
        if self.unit == 'epoch' and\
            self.every_n_epochs(runner, self.val_timing):
            
            result = self.validation(runner)
            runner.val_result.append(result)
                
        
    def validation(self, runner):
        model = runner.model
        model.eval()
        
        eval_cfg = dict(model= runner.model, 
                        cfg= self.val_cfg,
                        dataloader= self.val_dataloader)
        eval = Evaluate(**eval_cfg)   
        mAP = eval.compute_mAP()
        
        model.train()
        log_dict_loss = dict(**runner.log_buffer.get_last())        
        if log_dict_loss.get("data_time", None) is not None: del log_dict_loss['data_time']
        if log_dict_loss.get("time", None) is not None: del log_dict_loss['time']

        result = dict(epoch = runner.epoch, 
                      inner_iter = runner.inner_iter, 
                      mAP = mAP,
                      **log_dict_loss)
    
        log_str = ""
        for key, item in result.items():
            if key == "epoch":
                log_str +=f"EPOCH [{item}]"
                continue
            elif key == "inner_iter":
                log_str +=f"[{item}/{runner._iterd_per_epochs}]     "
                continue
            
            if type(item) == float:
                item = round(item, 4)
            log_str +=f"{key}: {item},     "
            
            
        datatime = compute_sec_to_h_d(time.time() - runner.start_time)
        log_str+=f"datatime: {datatime}"
        
        if self.logger is not None:
            self.logger.info(log_str)
        else: print(log_str)      # for Katib
        
        return result


@HOOK.register_module()
class TensorBoard_Hook(Hook):
    def __init__(self,
                 pvc_dir = None,
                 out_dir = None,
                 interval = ['iter', 50]):
        self.unit, self.timing = interval[0], interval[1]
        self.writer_pvc = SummaryWriter(log_dir = pvc_dir)
        self.writer_result_dir = SummaryWriter(log_dir = out_dir)
    
    
    def after_train_iter(self, runner) -> None: 
        if self.unit == 'iter' and\
            self.every_n_inner_iters(runner, self.timing):  
            self.write_to_board(runner)
    
                
    def after_train_epoch(self, runner) -> None: 
        if self.unit == 'epoch' and\
            self.every_n_epochs(runner, self.timing):
            self.write_to_board(runner)
              
        
    def write_to_board(self, runner):
        log_dict = runner.log_buffer.get_last()
        if log_dict.get("data_time", None) is not None: del log_dict['data_time']
        if log_dict.get("time", None) is not None: del log_dict['time']
        
        cur_lr = runner.current_lr()[0]
        
        log_dict['else_lr'] = cur_lr
        
        if len(runner.val_result)>0:
            log_dict['acc_mAP'] = runner.val_result[-1]['mAP']
            
        
        for key, item in log_dict.items():
            category = key.split("_")[0]
            name = key.split("_")[-1]
            
            if category == "loss":
                if key == "loss": 
                    name = 'total_loss'
                self.writer_pvc.add_scalar(f"Loss/{name}", item, runner._iter)
                self.writer_result_dir.add_scalar(f"Loss/{name}", item, runner._iter)
                
            elif category == "acc":
                self.writer_pvc.add_scalar(f"Acc/{name}", item, runner._iter)
                self.writer_result_dir.add_scalar(f"Acc/{name}", item, runner._iter)
                
            else:
                self.writer_pvc.add_scalar(f"else/{name}", item, runner._iter)
                self.writer_result_dir.add_scalar(f"else/{name}", item, runner._iter)
                

    def after_run(self, runner):
        self.writer_pvc.close()
        self.writer_result_dir.close()


@HOOK.register_module()
class Check_Hook(Hook):
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
            
            
def compute_sec_to_h_d(sec):
    if sec <=0: return "00:00:00"
    
    if sec < 60: return f'00:00:{f"{int(sec)}".zfill(2)}'
    
    minute = sec//60
    if minute < 60: return f"00:{f'{int(minute)}'.zfill(2)}:{f'{int(sec%60)}'.zfill(2)}"
    
    hour = minute//60
    if hour < 24: return f"{f'{int(hour)}'.zfill(2)}:{f'{int(minute%60)}'.zfill(2)}:{f'{int(sec%60)}'.zfill(2)}"
    
    day = hour//24
    return f"{day}day {f'{int(hour%24)}'.zfill(2)}:{f'{int(minute%(60))}'.zfill(2)}:{f'{int(sec%(60))}'.zfill(2)}"