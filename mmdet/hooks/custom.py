import os, os.path as osp
import time
import logging
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from sub_module.mmdet.hooks.hook import Hook, HOOK
from sub_module.mmdet.eval import Evaluate

@HOOK.register_module()
class Validation_Hook(Hook):
    def __init__(self,
                 val_dataloader: DataLoader,
                 result_dir = None,
                 logger = None,
                 run_infer = False,
                 interval = ['iter', 50],
                 val_cfg = None,
                 **kwargs
                ):
        self.iter_count = 1
        self.result_dir = result_dir
        self.run_infer = run_infer
        self.unit, self.val_timing = interval[0], interval[1]
        self.val_dataloader = val_dataloader
        self.val_cfg = val_cfg
        self.run_val = self.val_cfg['run']
        self.kwargs = kwargs
        
        log_file = osp.join(os.getcwd(), "test.log")

        self.logger = logger
      
        if self.kwargs.get('best_model', None) is not None:
            self.get_best_model = True
            self.save_best_model(init = True)
        else: self.get_best_model = False
  

    def every_n_inner_iters(self):
        return (self.iter_count) % self.val_timing == 0 if self.val_timing > 0 else False
    
    def after_train_iter(self, runner) -> None:     
        if self.run_val:

            self.iter_count +=1
            if self.unit == 'iter' and\
                self.every_n_inner_iters():
                
                result = self.validation(runner)
                if result is None: 
                    print(f"Memory usage is over. Validation is canceled")
                runner.val_result.append(result)           
    
    def after_train_epoch(self, runner) -> None: 
        if self.run_val: 
            if self.unit == 'epoch' and\
                self.every_n_epochs(runner, self.val_timing):
                
                result = self.validation(runner)
                if result is None: 
                    print(f"Memory usage is over. Validation is canceled")
                runner.val_result.append(result)
    
    def save_best_model(self, result: dict = None, runner = None, init = False):
        model_cfg = self.kwargs.get('model_cfg', None)
        if model_cfg is None: 
            raise KeyError(f"Must need `model_cfg` to save model, but got None.")
        
        if init:
            best_model_cfg = self.kwargs.get('best_model')
            self.keyword = best_model_cfg.key
            self.model_path = osp.join(best_model_cfg.get('dir', 'best'),
                                       best_model_cfg.get('name', "model.pth"))
            self.logfile_name = best_model_cfg.get("log", f"{time.strftime('%Y_%m_%d_%H_%M_%S')}.log")
       
            self.best_value = -1
            return None
        
        if result is None or runner is None: return None
        
        # Check if the directory path to save exists.
        
        
        if not hasattr(self, "best_model_dir"):
            if hasattr(runner, "dir_to_save"):
                output_path = runner.dir_to_save
            else:
                output_path = self.result_dir
            
            if output_path is None: return None
            self.best_model_dir = osp.join(osp.dirname(output_path), osp.dirname(self.model_path))
            os.makedirs(self.best_model_dir, exist_ok = True)
            
            self.best_model_logger = logging.getLogger("best_model_save_logger")
            self.best_model_logger.setLevel(logging.INFO)
            file_handler = logging.FileHandler(osp.join(self.best_model_dir, self.logfile_name), mode="a")
            file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
            self.best_model_logger.addHandler(file_handler)
            
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
            self.best_model_logger.addHandler(stream_handler)

        
    
        value_keyword = result[f'{self.keyword}']
        
        if self.best_value >= value_keyword: return None
        self.best_value = value_keyword
        
        self.best_model_logger.info(f"  Save best model to {self.best_model_dir}/{osp.basename(self.model_path)}")
        self.best_model_logger.info(f"  Epoch [{result['epoch']}]{result['inner_iter']}")
        self.best_model_logger.info(f"  Keyword: {self.keyword}, Value: {value_keyword}\n")
       
        save_cfg = dict(out_dir = self.best_model_dir,
                        filename_tmpl = osp.basename(self.model_path),
                        save_optimizer = True,
                        model_cfg = model_cfg,
                        val_mode = True)
        runner.save_checkpoint(**save_cfg)
        
        
    def validation(self, runner):
        model = runner.model
        val_dataloader = self.val_dataloader
        
        # Check if the directory path to save exists.
        if hasattr(runner, "dir_to_save"):
            output_path = runner.dir_to_save
        else:
            output_path = self.result_dir
        eval_cfg = dict(model= model, 
                        cfg= self.val_cfg,
                        dataloader= val_dataloader,
                        get_memory_info = self.get_memory_info,
                        output_path = output_path)  
        
        eval_ = Evaluate(**eval_cfg) 
        summary = eval_.get_mAP()  
        if summary is None: return None

        if self.run_infer:
            correct_inference_rate = eval_.run_inference()
            if correct_inference_rate is not None:
                summary['correct_inference_rate'] = correct_inference_rate
        
        model.train()
        log_dict_loss = dict(**runner.log_buffer.get_last())        
        if log_dict_loss.get("data_time", None) is not None: del log_dict_loss['data_time']
        if log_dict_loss.get("time", None) is not None: del log_dict_loss['time']

        result = dict(epoch = runner.epoch, 
                      inner_iter = f"[{runner.inner_iter}/{runner._iterd_per_epochs}]",
                      mAP = summary['normal']['mAP'],
                      dv_mAP = summary['dv']['mAP'],
                      **log_dict_loss)
        
        if self.get_best_model:
            self.save_best_model(result = result, runner = runner)
            
            
        log_str = ""
        for key, item in result.items():
            if key == "epoch":
                log_str +=f"EPOCH [{item}]"
                continue
            elif key == "inner_iter":
                log_str +=f"{item}     "
                log_str +=f"\n>>   "
                continue                
            if type(item) == float:
                item = round(item, 4)
            log_str +=f"{key}={item} ,     "
            if key == "dv_mAP":
                log_str +=f"\n>>   "
                    
        log_str +=f"\n>>   "
        datatime = self.compute_sec_to_h_d(time.time() - runner.start_time)
        log_str+=f"datatime: {datatime}"
        log_str +=f"\n"
        
        if self.logger is not None:
            self.logger.info(log_str)
        else: print(log_str)   

        return result


@HOOK.register_module()
class TensorBoard_Hook(Hook):
    def __init__(self,
                 pvc_dir = None,
                 out_dir = None,
                 interval = ['iter', 50]):
        self.unit, self.timing = interval[0], interval[1]
        self.pvc_dir = pvc_dir
        self.writer_result_dir = SummaryWriter(log_dir = out_dir)    

    def after_train_iter(self, runner) -> None: 
        if self.unit == 'iter' and\
            self.every_n_inner_iters(runner, self.timing):  
            self.write_to_board(runner)
    
                
    def after_train_epoch(self, runner) -> None: 
        if self.unit == 'epoch' and\
            self.every_n_epochs(runner, self.timing):
            self.write_to_board(runner)
              
        
    def writer_meta(self, board_path, value, runner):
        if runner.in_pipeline:      # save pvc path only run by kubeflow pipeline
            self.writer_pvc.add_scalar(board_path, value, runner._iter)
        self.writer_result_dir.add_scalar(board_path, value, runner._iter)
            
            
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
                self.writer_meta(f"Loss/{name}", item, runner)
                    
            elif category == "acc":
                self.writer_meta(f"Acc/{name}", item, runner)
                
            else:
                self.writer_meta(f"else/{name}", item, runner)
                
                
        memory = self.get_memory_info(runner)
        
        
        for key_i, item_i in memory.items():
            for key_j, item_j in item_i.items():
                if key_j == "percent": 
                    item_j = float(item_j.split("%")[0])
                elif key_j in ["max_allocated_tensor", "leakage", "free", "total"]: continue
                
 
                self.writer_meta(f"memory/{key_i}_{key_j}", item_j, runner)
             
    
    def before_run(self, runner):
        if runner.in_pipeline: 
            if not osp.isdir(self.pvc_dir):
                os.makedirs(self.pvc_dir, exist_ok=True)
            self.writer_pvc = SummaryWriter(log_dir = self.pvc_dir)
                
    def after_run(self, runner):
        if runner.in_pipeline: 
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
        
    def after_train_iter(self, runner) -> None: 
        self.check_memory_leakage(runner)
        self.check_memory_allocated(runner)
  
        
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
            
    
    def check_memory_leakage(self, runner):
        memory = self.get_memory_info(runner)
        GPU_total = memory['GPU']['total']
        GPU_used = memory['GPU']['used']
        GPU_allocated_tensor = memory['GPU']['allocated_tensor']
        GPU_leakage = memory['GPU']['leakage']
        
  
                
    
    def check_memory_allocated(self, runner):
        memory = self.get_memory_info(runner)
        RAM_memory_usage = float(memory['RAM']['percent'].split("%")[0])
        GPU_memory_usage = float(memory['GPU']['percent'].split("%")[0])
        
        if RAM_memory_usage > 90:   
            raise OSError(f"Memory resource usage: {RAM_memory_usage}% , process terminate!!")
                    
       