import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from torch.nn.parameter import Parameter
    
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
        self.avg_pool = GeM()
        self.head = nn.Sequential(
                            nn.Dropout(p=0.05),
                            nn.BatchNorm1d(global_dim),
                            nn.Linear(global_dim, 768),
                            nn.Dropout(p=0.05),
                            nn.BatchNorm1d(768),
                            nn.Linear(768, 512),
                            nn.Dropout(p=0.05),
                            nn.BatchNorm1d(512),
                            nn.Linear(512, 256),
                            nn.Dropout(p=0.05),
                            nn.BatchNorm1d(256),
                            nn.Linear(256, 1),
            )                                                        
    def forward(self, x):
        global_feat = self.cnn_backbone.extract_features(x) 
        global_feat = self.avg_pool(global_feat)                          
        global_feat = global_feat.view(global_feat.size(0), -1)         
        out = self.head(global_feat)   
        return {
            'LOGITS':out,
            }