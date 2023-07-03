from tqdm import tqdm

class AverageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, inp):
        val = inp[0]
        n = inp[1]
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    def feedback(self):
        return self.avg
    
class PrintMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.value = 0
    def update(self, inp):
        self.value = inp
    def feedback(self):
        return self.value
    
class store_info(object):
    def __init__(self,metrics):
        self.current_epoch = 1
        self.metrics = metrics           
        self.keys = list(metrics.keys())
        self.all_info = {'Metrics':self.keys}
        
        self.info_function = {}
        for key in metrics.keys():
            self.info_function.update({'Train'+key:metrics[key]()})
            self.info_function.update({'Val'+key:metrics[key]()})            
            
        self._init_new_epoch(self.current_epoch)
    def _init_new_epoch(self,epoch_no):
        train_info = {}
        val_info = {}
        for key in self.metrics.keys():
            train_info.update({'Epoch_'+key:0})
            val_info.update({'Epoch_'+key:0})
        
        self.all_info.update({'Epoch_'+str(epoch_no):{'Train':train_info,'Val':val_info}})
        self.current_epoch = epoch_no
        self.reset_info()
    def reset_info(self):
        for key in self.info_function.keys():
            self.info_function[key].reset()        
    def update_train_info(self,info_dict):
        for key in info_dict.keys():
            self.info_function['Train'+key].update(info_dict[key])           
            self.all_info['Epoch_'+str(self.current_epoch)]['Train']['Epoch_'+key] = self.info_function['Train'+key].feedback()
    def update_val_info(self,info_dict):
        for key in info_dict.keys():        
            self.info_function['Val'+key].update(info_dict[key])           
            self.all_info['Epoch_'+str(self.current_epoch)]['Val']['Epoch_'+key] = self.info_function['Val'+key].feedback()
    def request_current_epoch_metric_info(self):
        info = {}
        for key in self.keys:
              info.update({'Train'+key:self.all_info['Epoch_'+str(self.current_epoch)]['Train']['Epoch_'+key]})
              info.update({'Val'+key:self.all_info['Epoch_'+str(self.current_epoch)]['Val']['Epoch_'+key]})
        return info
    def request_current_epoch_train_metric_info(self):
        info = {}
        for key in self.keys:
              info.update({'Train'+key:self.all_info['Epoch_'+str(self.current_epoch)]['Train']['Epoch_'+key]})
        return info
    def request_current_epoch_val_metric_info(self):
        info = {}
        for key in self.keys:
              info.update({'Val'+key:self.all_info['Epoch_'+str(self.current_epoch)]['Val']['Epoch_'+key]})
        return info
    def request_allinfo(self):
        return self.all_info
    def load_info(self,all_info):
        self.all_info = all_info
    def get_info(self,epoch_no,metric,mode='Val'):
        if mode not in ['Val','Train']:
            raise print('Mode should be either Val or Train!')
        return self.all_info['Epoch_'+str(epoch_no)][mode]['Epoch_'+metric]

class TQDM(object):
    def __init__(self):
        self.progbar_train = None
        self.progbar_val = None
        super(TQDM, self).__init__()
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        # make sure the dbconnection gets closed
        if self.progbar_train is not None:
            self.progbar_train.close()
            
        if self.progbar_val is not None:
            self.progbar_val.close()
    
    def on_train_begin(self, logs):
        self.train_logs = logs
    def on_val_begin(self, logs):
        self.val_logs = logs
    
    def on_epoch_train_begin(self, fold, epoch):
        try:
            self.progbar_train = tqdm(total=self.train_logs['num_batches'],
                                unit=' batches')
            self.progbar_train.set_description('(Train) Fold %i Epoch %i/%i' % 
                            (fold, epoch, self.train_logs['num_epoch']))
        except:
            pass
    
    def on_epoch_train_end(self, logs=None):
        log_data = {key: '%.04f' % logs[key] for key in logs.keys()}
        self.progbar_train.set_postfix(log_data)
        self.progbar_train.update()
        self.progbar_train.close()
        print('')
        
    def on_epoch_val_begin(self, fold, epoch):
        try:            
            self.progbar_val = tqdm(total=self.val_logs['num_batches'],
                                unit=' batches')
            self.progbar_val.set_description('(Valid) Fold %i Epoch %i/%i' % 
                            (fold, epoch, self.val_logs['num_epoch']))                                    
        except:
            pass
    def on_epoch_val_end(self, logs=None):
        log_data = {key: '%.04f' % logs[key] for key in logs.keys()}        
        self.progbar_val.set_postfix(log_data)
        self.progbar_val.update()
        self.progbar_val.close()
        print('')
        
    def on_train_batch_begin(self):
        self.progbar_train.update(1)
        
    def on_train_batch_end(self, logs=None):
        log_data = {key: '%.04f' % logs[key] for key in logs.keys()}
        self.progbar_train.set_postfix(log_data)
        
    def on_val_batch_begin(self):
        self.progbar_val.update(1)
        
    def on_val_batch_end(self, logs=None):
        log_data = {key: '%.04f' % logs[key] for key in logs.keys()}
        self.progbar_val.set_postfix(log_data)

