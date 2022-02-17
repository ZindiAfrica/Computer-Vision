import os
import numba
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from optimizer import *
from trainer_callbacks import *
from utils import *

#%% #################################### Model Trainer Class #################################### 
class ModelTrainer():
    def __init__(self, 
                 model=None, 
                 Loaders=[None,[]], 
                 metrics=None, 
                 fold=None, 
                 lr=None, 
                 epochsTorun=None,
                 checkpoint_saving_path=None,
                 resume_train_from_checkpoint=False, 
                 resume_checkpoint_path=None,
                 test_run_for_error=False,
                 batch_size=None,
                 do_grad_accum=False,
                 grad_accum_steps=4,
                 use_fp16=True,
                 problem_name=None
                 ):     
        super(ModelTrainer, self).__init__()
                    
        self.problem_name = problem_name
        self.model = model.cuda()
        self.trainLoader = Loaders[0]
        self.valLoader = Loaders[1]        
        self.info_bbx = store_info(metrics)
        self.fold = fold
                
        if self.fold != None:
            self.checkpoint_saving_path = checkpoint_saving_path + '/fold' + str(self.fold) + '/'
        else:
            self.checkpoint_saving_path = checkpoint_saving_path + '/'
            self.fold = 0       
        os.makedirs(self.checkpoint_saving_path,exist_ok=True)
        
        self.lr = lr
        self.epochsTorun = epochsTorun
        self.init_epoch = -1        
        self.test_run_for_error = test_run_for_error
        self.current_checkpoint_save_count = 1
        self.resume_checkpoint_path = resume_checkpoint_path

        self.best_loss = 9999
        self.best_f1_score = -9999
        self.best_rmse = 9999
        self.batch_size = batch_size          
        
        
        self.optimizer = Over9000(params=self.model.parameters(),lr=self.lr)  
        self.scheduler = ReduceLROnPlateau(self.optimizer, factor=0.5, mode='min', patience=5, verbose=True)
        
        self.do_grad_accum = do_grad_accum
        self.grad_accum_steps = grad_accum_steps
        
        self.trainer_settings_dict = { 
                                    'do_grad_accum': self.do_grad_accum,
                                    'grad_accum_steps':self.grad_accum_steps,
                                    'epochsTorun':self.epochsTorun,
                                    'lr':self.lr,
                                    'batch_size':batch_size,
                                    }
        
        self.use_fp16 = use_fp16
        self.scheduler_flag = 9999
               
        self.criterion = RMSELoss().cuda()
        self.criterion_2 = nn.CrossEntropyLoss().cuda()
        
        self.scaler = torch.cuda.amp.GradScaler()

        if resume_train_from_checkpoint:
            if os.path.isfile(resume_checkpoint_path):
                print("=> Loading checkpoint from '{}'".format(resume_checkpoint_path))
                checkpoint_dict = torch.load(resume_checkpoint_path)                
                self.model.load_state_dict(checkpoint_dict['Model_state_dict'])
                self.scheduler.load_state_dict(checkpoint_dict['Scheduler_state_dict'])
                self.optimizer.load_state_dict(checkpoint_dict['Optimizer_state_dict'])        
                
                self.best_loss = checkpoint_dict['Best_val_loss']
                self.best_f1_score = checkpoint_dict['Best_val_f1_score']
                
                self.info_bbx.all_info = checkpoint_dict['All_info']
                self.init_epoch = checkpoint_dict['Epoch']
                            
                print('Best Val loss is {}'.format(self.best_loss))
                print('Best Val f1_score is {}'.format(self.best_f1_score))
                
                print('Current val loss is {}'.format(checkpoint_dict['Current_val_Loss']))
                print('Current val f1 score is {}'.format(checkpoint_dict['Current_val_f1_score']))
                
                self.scheduler_flag = checkpoint_dict['Scheduler_flag']
                
                del checkpoint_dict
                torch.cuda.empty_cache()
            else:
                print("=> No checkpoint found at '{}' !".format(resume_checkpoint_path))                                                
#%% train part starts here
    def fit(self):                
        with TQDM() as pbar:           
            pbar.on_train_begin({'num_batches':len(self.trainLoader),'num_epoch':self.epochsTorun})
            pbar.on_val_begin({'num_batches':len(self.valLoader),'num_epoch':self.epochsTorun})            
           
            self.train_metric_meter = Metric_Meter() 
            self.val_metric_meter = Metric_Meter()
            
            for epoch in range(self.epochsTorun):
                current_epoch_no = epoch+1                                           
                if current_epoch_no <= self.init_epoch:
                    continue 
                
                pbar.on_epoch_train_begin(self.fold,current_epoch_no)                
                self.info_bbx._init_new_epoch(current_epoch_no)                                
                                
                self.model.train()
                torch.set_grad_enabled(True)
                #self.optimizer.zero_grad()
                
                self.train_metric_meter.reset()
                self.val_metric_meter.reset()                               
                for itera_no, data in enumerate(self.trainLoader):
                    pbar.on_train_batch_begin()                                    
                    self.optimizer.zero_grad()

                    images, targets = data
                    images = images.cuda() 
                    targets = targets.cuda()                                                         
                    
                    with torch.cuda.amp.autocast():
                        out = self.model(images)                                                            
                        batch_loss = self.criterion(out['LOGITS'], targets[:,None]) + self.criterion_2(out['LOGITS_2'], targets.long())
                                                                            
                    self.scaler.scale(batch_loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    
                    self.train_metric_meter.update(out['LOGITS'].clone(), targets, 'single')                    
                    self.info_bbx.update_train_info({'Loss':[(batch_loss.detach().item()),images.shape[0]]})
                                                                            
                    pbar.on_train_batch_end(logs=self.info_bbx.request_current_epoch_train_metric_info())                                     
                    torch.cuda.empty_cache()
                                       
                    if self.test_run_for_error:                                     
                        if itera_no==5:
                            break                        
#%% validation part starts here 
                f1_score, rmse = self.train_metric_meter.feedback()                                                          
                self.info_bbx.update_train_info({'f1_score': f1_score, 'rmse': rmse})           
                pbar.on_epoch_train_end(self.info_bbx.request_current_epoch_train_metric_info())                
                
                pbar.on_epoch_val_begin(self.fold,current_epoch_no)                
                self.model.eval()
                torch.set_grad_enabled(False)                                
                with torch.no_grad():
                    for itera_no, data in enumerate(self.valLoader):
                        pbar.on_val_batch_begin()
                        
                        images, targets = data
                        images = images.cuda() 
                        targets = targets.cuda()                                                        
                        
                        with torch.cuda.amp.autocast():
                            out = self.model(images) 
                            batch_loss = self.criterion(out['LOGITS'], targets[:,None]) + self.criterion_2(out['LOGITS_2'], targets.long())
                                                                                                                       
                        self.val_metric_meter.update(out['LOGITS'].clone(), targets, 'single')
                        self.info_bbx.update_val_info({'Loss':[(batch_loss.detach().item()),images.shape[0]]})
                                                                                          
                        pbar.on_val_batch_end(logs=self.info_bbx.request_current_epoch_val_metric_info())
                        torch.cuda.empty_cache()
                        if self.test_run_for_error:
                            if itera_no==5:
                                break
                    f1_score, rmse = self.val_metric_meter.feedback()
                    self.info_bbx.update_val_info({'f1_score': f1_score, 'rmse': rmse})                                             
                    pbar.on_epoch_val_end(self.info_bbx.request_current_epoch_val_metric_info())
#%% Update best parameters
                if self.best_loss > self.info_bbx.get_info(current_epoch_no,'Loss','Val'):
                    print( ' Val Loss is improved from {:.4f} to {:.4f}! '.format(self.best_loss,self.info_bbx.get_info(current_epoch_no,'Loss','Val')) )
                    self.best_loss = self.info_bbx.get_info(current_epoch_no,'Loss','Val')
                    is_best_loss = True
                else:
                    print( ' Val Loss is not improved from {:.4f}! '.format(self.best_loss))
                    is_best_loss = False
                    
                if self.best_f1_score < self.info_bbx.get_info(current_epoch_no,'f1_score','Val'):
                    print( ' Val f1 score is improved from {:.4f} to {:.4f}! '.format(self.best_f1_score,self.info_bbx.get_info(current_epoch_no,'f1_score','Val')) )
                    self.best_f1_score = self.info_bbx.get_info(current_epoch_no,'f1_score','Val')
                    is_best_f1_score = True
                else:
                    print( ' Val f1 score is not improved from {:.4f}! '.format(self.best_f1_score))
                    is_best_f1_score = False                    
#%%Learning Rate Schedulers                
                if is_best_loss or is_best_f1_score:
                    self.scheduler_flag = self.scheduler_flag - 1 
                    self.scheduler.step(self.scheduler_flag)
                else:
                    self.scheduler.step(self.scheduler_flag+1)                    
#%%checkpoint dict creation                                    
                checkpoint_dict = {
                    'Epoch': current_epoch_no,
                    'Model_state_dict': self.model.state_dict(),
                    'Current_val_Loss': self.info_bbx.get_info(current_epoch_no,'Loss','Val'),
                    'Current_train_Loss': self.info_bbx.get_info(current_epoch_no,'Loss','Train'),
                    'Current_val_f1_score':self.info_bbx.get_info(current_epoch_no,'f1_score','Val'),
                    'Current_train_f1_score':self.info_bbx.get_info(current_epoch_no,'f1_score','Train'),
                    'Current_val_rmse':self.info_bbx.get_info(current_epoch_no,'rmse','Val'),
                    'Current_train_rmse':self.info_bbx.get_info(current_epoch_no,'rmse','Train'),
                    'Best_val_loss' : self.best_loss,
                    'Best_val_f1_score': self.best_f1_score,
                    'Best_val_rmse': self.best_rmse,
                    }
#%%checkpoint dict saving                          
                if is_best_f1_score:                                    
                    torch.save(checkpoint_dict, self.checkpoint_saving_path+'checkpoint_best_f1_score_fold{}.pth'.format(self.fold))
              
                del checkpoint_dict
                torch.cuda.empty_cache()