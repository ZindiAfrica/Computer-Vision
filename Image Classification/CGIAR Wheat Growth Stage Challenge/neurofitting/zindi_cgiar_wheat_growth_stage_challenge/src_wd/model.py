import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from torch.nn.parameter import Parameter

class SE_BLOCK(nn.Module):
    def __init__(self, input_dim=1280):        
        super(SE_BLOCK,self).__init__()
        self.layer = nn.Sequential(
                      nn.Linear(input_dim, 512),
                      nn.ReLU(),
                      nn.Linear(512, input_dim),
                      nn.Sigmoid(),
                      )                                                                            
    def forward(self, x):
        out = F.avg_pool2d(x,(x.size(-2), x.size(-1))).view(x.size(-4), -1)
        out = self.layer(out)[:,:,None,None]
        out = x*out
        return out
    
def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM,self).__init__()
        self.p = Parameter(torch.ones(1)*p)
        self.eps = eps
    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)       
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'

class CNN_Model(nn.Module):
    def __init__(self, model_name='efficientnet-b2', global_dim=1408):        
        super(CNN_Model,self).__init__()            
        self.cnn_backbone = EfficientNet.from_pretrained(model_name) 
        self.se_block = SE_BLOCK(global_dim)
        self.avg_pool = GeM()
        self.head = nn.Sequential(
                            nn.Dropout(p=0.25),
                            nn.Linear(global_dim, 1)
            )
        self.head_2 = nn.Sequential(
                            nn.Dropout(p=0.25),
                            nn.Linear(global_dim, 7)
            )                                                         
    def forward(self, x):
        global_feat = self.cnn_backbone.extract_features(x) 
        global_feat = self.se_block(global_feat)
        global_feat = self.avg_pool(global_feat)                          
        global_feat = global_feat.view(global_feat.size(0), -1) 
        out = self.head(global_feat)   
        out_2 = self.head_2(global_feat)
        return {
            'LOGITS':out,
            'LOGITS_2':out_2,
            }