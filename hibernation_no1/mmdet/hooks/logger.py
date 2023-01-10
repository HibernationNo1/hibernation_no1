import torch
import os, os.path as osp
import json
import time
from collections import OrderedDict

from docker.hibernation_no1.utils.utils import is_list_of, is_tuple_of, dict_to_pretty
from docker.hibernation_no1.mmdet.hooks.hook import Hook, HOOK
from typing import Optional, Dict

@HOOK.register_module()
class LoggerHook(Hook):
    """Logger hook in text.

    In this logger hook, the information will be printed on terminal and
    saved in json file.

    Args:
        by_epoch (bool, optional): Whether EpochBasedRunner is used.
            Default: True.
        interval (int, optional): Logging interval (every k iterations).
            Default: 10.
        ignore_last (bool, optional): Ignore the log of last iterations in each
            epoch if less than :attr:`interval`. Default: True.
        reset_flag (bool, optional): Whether to clear the output buffer after
            logging. Default: False.
        interval_exp_name (int, optional): Logging interval for experiment
            name. This feature is to help users conveniently get the experiment
            information from screen or log file. Default: 1000.
        out_dir (str, optional): Logs are saved in ``runner.work_dir`` default.
            If ``out_dir`` is specified, logs will be copied to a new directory
            which is the concatenation of ``out_dir`` and the last level
            directory of ``runner.work_dir``. Default: None.
            `New in version 1.3.16.`
        out_suffix (str or tuple[str], optional): Those filenames ending with
            ``out_suffix`` will be copied to ``out_dir``.
            Default: ('.log.json', '.log', '.py').
            `New in version 1.3.16.`
        keep_local (bool, optional): Whether to keep local log when
            :attr:`out_dir` is specified. If False, the local log will be
            removed. Default: True.
            `New in version 1.3.16.`
    """

    def __init__(self,
                 interval: int = 10,
                 ignore_last: bool = True,
                 reset_flag: bool = False,
                 interval_exp_name: int = 1000,
                 out_dir: Optional[str] = None,
                 out_suffix: str = '.log',
                 log_file_name: str = 'RUN_log',
                 keep_local: bool = True):
        self.iter_count = 1     # for compute remain time at self.compute_remain_time
        self.interval = interval
        self.ignore_last = ignore_last
        self.reset_flag = reset_flag
        self.time_sec_tot = 0
        self.interval_exp_name = interval_exp_name
        
        self.out_dir = out_dir  
        
        if not (out_dir is None or isinstance(out_dir, str)
                or is_tuple_of(out_dir, str)):
            raise TypeError('out_dir should be  "None" or string or tuple of '
                            'string, but got {out_dir}')
        self.out_suffix = out_suffix
        self.log_file_name = log_file_name + out_suffix

        self.keep_local = keep_local
    
    
    def _round_float(self, items):
        if isinstance(items, list):
            return [self._round_float(item) for item in items]
        elif isinstance(items, float):
            return round(items, 5)
        else:
            return items
    
    
    def _get_max_memory(self, runner) :
        """
            Size of tensor allocated to GPU (unit: MB)
        """
        device = getattr(runner.model, 'output_device', None)
        mem = torch.cuda.max_memory_allocated(device=device)
        mem_mb = torch.tensor([int(mem) // (1024 * 1024)],
                              dtype=torch.int,
                              device=device)
        return mem_mb.item()
    
    
    
    def _log_info(self, log_dict: Dict, runner) -> None:
        # print exp name for users to distinguish experiments
        # at every ``interval_exp_name`` iterations and the end of each epoch
        if runner.meta is not None and 'exp_name' in runner.meta:
            if (self.every_n_iters(runner, self.interval_exp_name)) or (
                    self.end_of_epoch(runner)):
                exp_info = f'Exp name: {runner.meta["exp_name"]}'
                runner.logger.info(exp_info)

        
        if isinstance(log_dict['lr'], dict):
            lr_str = []
            for k, val in log_dict['lr'].items():
                lr_str.append(f'lr_{k}: {val:.3e}')
            lr_str = ' '.join(lr_str)  # type: ignore
        else:
            lr_str = f'lr: {log_dict["lr"]:.3e}'  # type: ignore

        # by epoch: Epoch [4][100/1000]
        # by iter:  Iter [100/100000]
        
        log_str = f'Epoch [{log_dict["epoch"]}]' \
                    f'[{log_dict["iter"]}/{len(runner.train_dataloader)}]\t'
        log_str += f'{lr_str}, '

        import datetime
        if 'time' in log_dict.keys():
            self.time_sec_tot += (log_dict['time'] * self.interval)
            time_sec_avg = self.time_sec_tot / (
                runner.iter - self.start_iter)
            eta_sec = time_sec_avg * (runner.max_iters - runner.iter - 1)
            eta_str = str(datetime.timedelta(seconds=int(eta_sec)))     
            log_str += f'eta: {eta_str}, '
            log_str += f'time: {log_dict["time"]:.3f}, ' \
                        f'data_time: {log_dict["data_time"]:.3f}, '
            # statistic memory
            if torch.cuda.is_available():
                log_str += f'memory: {log_dict["memory"]}, '
        

        
        
        log_items = []
        for name, val in log_dict.items():
            # TODO: resolve this hack
            # these items have been in log_str
            if name in [
                    'mode', 'Epoch', 'iter', 'lr', 'time', 'data_time',
                    'memory', 'epoch'
            ]:
                continue
            if isinstance(val, float):
                val = f'{val:.4f}'
            log_items.append(f'{name}: {val}')
        
        log_str += ', '.join(log_items)
        
        
        runner.logger.info(log_str)

        
        
    def write_log(self, status, log_):    
        # dump log in .log format

        if self.out_suffix=='.json' :       # can write max 21889 lines
            if not isinstance(log_, dict): raise TypeError(f"if out_suffix is '.json', input log type must be dict! ")
            if not osp.isfile(self.log_file_path):
                text_log = dict()
                
            else:
                with open(self.log_file_path, "r", encoding='utf-8') as f:
                    text_log = json.load(f)
        
            if status == 'before_run':
                text_log[status] = log_
            elif status == 'before_epoch': 
                dict_map = dict(before_epoch= None,
                                EPOCH = list(),
                                after_epoch = None)  
                if text_log.get('RUN', None) is None:
                    text_log['RUN'] = [dict_map] 
                else: 
                    text_log['RUN'].append(dict_map)
                text_log['RUN'][-1]['before_epoch'] = log_
            elif status =="after_iter":
                text_log['RUN'][-1]['EPOCH'].append(log_)
            elif status == 'after_epoch':
                text_log['RUN'][-1]['after_epoch'] = log_
            elif status == 'after_run':
                text_log[status] = log_
            
            json.dump(text_log, open(self.log_file_path, "w"), indent = 4)
                
        elif self.out_suffix=='.log':
            if not isinstance(log_, list): raise TypeError(f"if out_suffix is '.log', input log type must be list! ")
            if not is_list_of(log_, dict): raise TypeError(f"Element of list must be dict, but was not.")
            
            text_log = '\n'
            if status in ['before_run', 'after_run']:
                num_bar = 60
            elif status in ['before_epoch', 'after_epoch']:
                num_bar = 45
            elif status in ['before_iter', 'after_iter']:
                num_bar = 30
            
            for i in range(num_bar):
                if i == num_bar//2:
                    text_log += f'< {status} >'
                text_log += '-'
            
            for log_dict in log_:
                text_log = text_log + "\n" + dict_to_pretty(log_dict) + "\n"
                
            if osp.isfile(self.log_file_path):
                with open(self.log_file_path, 'a+') as f:
                    f.write(f"{text_log}")
            else:
                with open(self.log_file_path, 'w') as f:        
                    f.write(f"{text_log}")
    
        
                
    def before_run(self, runner):
        for hook in runner.hooks[::-1]:
            if isinstance(hook, LoggerHook):
                hook.reset_flag = True
                break
            
        if self.out_dir is not None:
            runner.logger.info(
                f'Text logs will be saved to {self.out_dir} after the training process.')
            self.log_file_path = osp.join(self.out_dir, self.log_file_name)
        else:
            self.log_file_path = osp.join(runner.work_dir, self.log_file_name)
              
        self.start_iter = runner.iter

        log_dict = dict(bach_size = runner.get('batch_size') ,
                        max_iteration = runner.get('_max_iters'),
                        max_epoche = runner.get('_max_epochs'))

        if self.out_suffix=='.log':
            log_dict = [log_dict]
        self.write_log('before_run', log_dict)  
        
        self.r_t = time.time()  
            
        
        
    def before_train_epoch(self, runner):
        log_dict = dict(epoch = f'[{runner.epoch}/{runner._max_epochs}]',
                        iterd_per_epochs = runner._iterd_per_epochs)
        
        if self.out_suffix=='.log':
            log_dict = [log_dict]
            
        self.write_log('before_epoch', log_dict)
        runner.log_buffer.clear()  # clear logs of last epoch
        self.e_t = time.time()      # epoch start time 
    
                
        
    def after_train_iter(self, runner) -> None:    
        self.iter_count +=1
        if self.every_n_inner_iters(runner, self.interval):   # training unit: epoch, iter +1
            runner.log_buffer.average(self.interval)   
        # elif not self.by_epoch and self.every_n_iters(runner, self.interval):   # training unit: iter, iter +1
        #     runner.log_buffer.average(self.interval)
        elif self.end_of_epoch(runner) and not self.ignore_last:    # at last epoch
            # not precise but more stable
            runner.log_buffer.average(self.interval)

        
        if runner.log_buffer.ready:
            self.log(runner)
            
            if self.reset_flag:
                runner.log_buffer.clear_output()
        
        
        current_iter = runner._inner_iter  
        log_dict = dict(epoch=f'[{runner.epoch}/{runner._max_epochs}]',
                        iter=f'[{current_iter}/{runner._iterd_per_epochs}]')
        
        log_dict_loss = dict(**runner.log_buffer.get_last())        
        if log_dict_loss.get("data_time", None) is not None: del log_dict_loss['data_time']
        if log_dict_loss.get("time", None) is not None: del log_dict_loss['time']

        for key, item in log_dict_loss.items():
            log_dict[key] = item

        if self.out_suffix=='.log':
            log_dict = [log_dict]

        self.write_log("after_iter", log_dict)
        
        runner.log_buffer.log(self.interval)
        runner.log_buffer.clear_log()
        
     


    def after_train_epoch(self, runner) -> None:
        if runner.log_buffer.ready:
            self.log(runner)
        if self.reset_flag:
            runner.log_buffer.clear_output()
        
        taken_time_epoch = time.time() - self.e_t
        log_dict_meta = dict(
            taken_time_epoch = self.compute_sec_to_h_d(taken_time_epoch),
            remain_time = self.compute_remain_time(taken_time_epoch/runner._iterd_per_epochs, runner._max_iters))
        
        if self.out_suffix=='.log':
            log_dict_meta = [log_dict_meta]
        self.write_log("after_epoch", log_dict_meta)

   
    def after_run(self, runner):
        log_dict = dict(taken_time_training = self.compute_sec_to_h_d(time.time() - self.r_t))

        log_dict_loss = dict(**runner.log_buffer.get_last()) 
        if log_dict_loss.get("data_time", None) is not None: del log_dict_loss['data_time']
        if log_dict_loss.get("time", None) is not None: del log_dict_loss['time']

        for key, item in log_dict_loss.items():
            log_dict[key] = item
            
        if self.out_suffix=='.log':
            log_dict = [log_dict]
 
        
        self.write_log("after_run", log_dict)
        runner.log_buffer.clear_log()
         
    
    def log(self, runner) :
        if 'eval_iter_num' in runner.log_buffer.output:
            # this doesn't modify runner.iter and is regardless of by_epoch
            cur_iter = runner.log_buffer.output.pop('eval_iter_num')
        else:
            cur_iter = runner._inner_iter

        log_dict = OrderedDict(epoch=runner.epoch, 
                               iter=cur_iter)
        
        # assign learning rate
        cur_lr = runner.current_lr()
        if isinstance(cur_lr, list):
            log_dict['lr'] = cur_lr[0]
        else:
            assert isinstance(cur_lr, dict)
            log_dict['lr'] = {}
            for k, lr_ in cur_lr.items():
                assert isinstance(lr_, list)
                log_dict['lr'].update({k: lr_[0]})
                
        if 'time' in runner.log_buffer.output:
            # statistic memory
            if torch.cuda.is_available():
                log_dict['memory'] = self._get_max_memory(runner)
        
        log_dict = dict(log_dict, **runner.log_buffer.output) 
        self._log_info(log_dict, runner)
        
        