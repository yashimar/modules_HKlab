import os
import random
import math
import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import special
import plotly
import plotly.graph_objs as go

import torch 
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import grad

from torchvision import transforms
from torchsummary import summary
from visdom import Visdom



import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = torch.device('cuda') 
print('device : ', device)
print('num gpu : ', torch.cuda.device_count())


def step(source):
    return (source>0)*1.


class BasicRegressor(nn.Module):
    
    def __init__(self):
        super(BasicRegressor, self).__init__()
        self.layer1 = nn.Linear(1, 200)
        self.layer2 = nn.Linear(200, 200)
        self.last = nn.Linear(200, 1)
        self.dropout = nn.Dropout(p=0.01)
        
    def forward(self, x):
        out = self.layer1(x)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.layer2(out)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.last(out)
        
        return out
    

class BasicRegressor2(nn.Module):
    
    def __init__(self):
        super(BasicRegressor2, self).__init__()
        self.layer1 = nn.Linear(1, 2048)
        self.layer2 = nn.Linear(2048, 2048)
        self.layer3 = nn.Linear(2048, 2048)
        self.layer4 = nn.Linear(2048, 2048)
        self.last = nn.Linear(2048, 1)
        self.dropout = nn.Dropout(p=0.01)
        
    def forward(self, x):
        out = self.layer1(x)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.layer2(out)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.layer3(out)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.layer4(out)
        out = F.relu(out)
        out = self.last(out)
        
        return out
    

def train(model, dataloader, num_epochs, criterion=nn.MSELoss(), lr=2e-3, schedule_param=None, save_log_path=None, save_model_path=None):   
    if (save_log_path!=None and os.path.exists(f'{save_log_path}')): os.remove(f'{save_log_path}')  
    
    model = nn.DataParallel(model)
    model.to(device)
    model.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if schedule_param!=None: scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **schedule_param)
    criterion = criterion
    
    for epoch in range(num_epochs):      
        t_epoch_start = time.time()
        epoch_loss = 0.0
        
        for data in dataloader:       
            batch_size = len(data[0])
            x = data[0].to(device)
            y = data[1].to(device)
            out = model(x)
            
            loss = criterion(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        t_epoch_finish = time.time()
        if save_log_path!=None:
            with open(f'{save_log_path}', 'a') as f:
                print('-------------', file=f)
                print('Epoch {}/{}'.format(epoch+1, num_epochs),file=f)
                print('Loss:{:.4f} '.format(epoch_loss/batch_size), file=f)
                print('timer:  {:.4f} sec.'.format(t_epoch_finish - t_epoch_start), file=f)
        else:
            print('-------------')
            print('Epoch {}/{}'.format(epoch+1, num_epochs))
            print('Loss:{:.4f} '.format(epoch_loss/batch_size))
            print('timer:  {:.4f} sec.'.format(t_epoch_finish - t_epoch_start))
        
        if schedule_param!=None: scheduler.step(epoch_loss/batch_size)
        if save_model_path!=None: torch.save(model.state_dict(), save_model_path)
        
    model.eval()        
    return model


def NNgrad(model, x):
    weights = []
    biases = []
    length = len(model.state_dict().keys())
    keys = iter(model.state_dict().keys())
    for i in range(length):
        key = next(keys)
        if "weight" in key:
            weights.append(model.state_dict()[key])
        elif "bias" in key:
            biases.append(model.state_dict()[key])
        else:
            raise Exception("Unseened Prameter was given. Weight or Bias are desirable.")
    if (len(weights)!=len(biases)): raise Exception(f"len(weights) is {len(weights)}, but len(biases) is {len(biases)}.")
    
    h = [x] + [0]*len(weights)
    grad_layer = [0]*len(weights)
    for i in range(len(weights)):        
        if i==len(weights)-1:
            h[i+1] = h[i]@weights[i].T + biases[i]
            grad_layer[i] = weights[i].T
        else:
            h[i+1] = F.relu(h[i]@weights[i].T + biases[i])
            grad_layer[i] = weights[i].T@torch.diag_embed(step(h[i]@weights[i].T+biases[i]))    
            

    for i, grad in enumerate(grad_layer):
        dydx = dydx@grad if i!=0 else grad
    dydx = dydx.view(-1,weights[0].size(1))
        
    return dydx


def plot3D(model=None, true_function=None, length=50, start=-1, end=1, dot_size=3):
    plotly.offline.init_notebook_mode()
    waith = end - start
    val_x = torch.tensor([[i, j] for i in np.arange(start,end,waith/length) for j in np.arange(start,end,waith/length)])
    val_x1, val_x2 = val_x[:,0], val_x[:,1]
    
    trace_data = []
    if model!=None: 
        pred_y = model(val_x).detach().cpu().view(-1)
        pred = go.Scatter3d(
            x=val_x1,  
            y=val_x2, 
            z=pred_y,  
            mode='markers',
            marker={
                'size': dot_size,
                'opacity': 0.8,
            }, 
            name='pred',
        )
        trace_data.append(pred)
    
    if true_function!=None:
        true_y = true_function(val_x1, val_x2)
        true = go.Scatter3d(
            x=val_x1,  
            y=val_x2, 
            z=true_y,  
            mode='markers',
            marker={
                'size': dot_size,
                'opacity': 0.8,
            },  
            name='true',
        )
        trace_data.append(true)
        
    layout = go.Layout(
        margin={'l': 0, 'r': 0, 'b': 0, 't': 0},
        scene={
            'xaxis': {'title': 'x1'},
            'yaxis': {'title': 'x2'},
            'zaxis': {'title': 'y'}
        }
    )
    plot_figure = go.Figure(data=trace_data, layout=layout)

    plotly.offline.iplot(plot_figure)