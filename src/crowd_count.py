import torch.nn as nn
import network
import models

class CrowdCounter(nn.Module):
    def __init__(self, modelname=''):
        super(CrowdCounter, self).__init__()
        global selfmodelname
        if modelname != '':
             selfmodelname= modelname
        assert selfmodelname != ''
        if selfmodelname == 'mcnn':
            self.DME = models.MCNN()
        elif selfmodelname == 'MCNNandMBv1':
            self.DME = models.MCNNandMBv1()
        elif selfmodelname == 'MCNNandMBv1-1':
            self.DME = models.MCNNandMBv1_1()
        elif selfmodelname == 'MCNNandMBv1-2':
            self.DME = models.MCNNandMBv1_2
        self.loss_fn = nn.MSELoss()
        
    @property
    def loss(self):
        return self.loss_mse
    
    def forward(self,  im_data, gt_data=None):        
        im_data = network.np_to_variable(im_data, is_cuda=True, is_training=self.training)                
        density_map = self.DME(im_data)
        
        if self.training:                        
            gt_data = network.np_to_variable(gt_data, is_cuda=True, is_training=self.training)            
            self.loss_mse = self.build_loss(density_map, gt_data)
            
        return density_map
    
    def build_loss(self, density_map, gt_data):
        loss = self.loss_fn(density_map, gt_data)
        return loss
