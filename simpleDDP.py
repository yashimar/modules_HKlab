import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import pandas as pd

import torch
from torch.autograd import grad



def node(tensor):
    return tensor.clone().detach().requires_grad_(True)

def init_state(N, init_x):
    one = torch.ones(1)*1.
    zero = torch.zeros(1)

    nx = [node(one*init_x) for i in range(N+1)]
    ex = [node(zero) for i in range(N)]
    er = [node(zero) for i in range(N+1)]

    fx = [node(zero) for i in range(N)]
    fa = [node(zero) for i in range(N)]
    rx = [node(zero) for i in range(N+1)]
    sigma = [node(zero) for i in range(N+1)]
    ja = [node(zero) for i in range(N)]
    
    return nx, ex, er, fx, fa, rx, sigma, ja

def record(nx, ex, na, er, fx, fa, rx, sigma, ja, item_only=True):    
    df = pd.DataFrame({"nx": nx,"ex":ex+[[""]],"na":na+[[""]],"er":er,"fx":fx+[[""]],"fa":fa+[[""]],
                                     "rx":rx,"sigma":sigma,"ja":ja+[[""]]})
    if item_only: df=df.applymap(lambda x: x[0].item() if x[0]!="" else np.nan)
        
    return df

def optimize(na, ja, eta, clip_a_value, mode_ascent):
    next_na = na + eta*ja if mode_ascent else na - eta*ja
    if clip_a_value!=None: next_na = torch.clamp(next_na, -clip_a_value, clip_a_value)
        
    return next_na

def forward(f, r, nx, ex, na, fx, fa, er, rx):
    N = len(na)
    for i in range(N):
        ex[i] = f(nx[i], na[i])
        nx[i+1] = node(ex[i])
        fx[i] = grad(ex[i], nx[i])[0]
        fa[i] = grad(ex[i], na[i])[0]
        er[i+1] = r(nx[i+1])
        rx[i+1] = grad(er[i+1], nx[i+1])[0]
        
def backward(na, fx, fa, rx, sigma, ja, eta, clip_a_value, mode_ascent):
    N = len(fx)
    for i in range(N-1,-1,-1):
        sigma[i] = fx[i+1]*sigma[i+1]+rx[i+1] if i!=N-1 else rx[i+1]
        ja[i] = fa[i]*sigma[i]
        na[i] = optimize(na[i], ja[i], eta, clip_a_value, mode_ascent)
        
def solve(f, r, n_roop, time_length, eta=0.01, init_x=30., init_a_range=2.0, clip_a_value=None, mode_ascent=True, item_only=True, stack_log=False):
    N = time_length
    na = [node(torch.empty(1).uniform_(-init_a_range, init_a_range)) for i in range(N)]
    logs = []
    for roop in range(n_roop):
        if stack_log: logs.append(record(nx, ex, na, er, fx, fa, rx, sigma, ja, item_only=item_only))
        nx, ex, er, fx, fa, rx, sigma, ja = init_state(N, init_x)
        forward(f, r, nx, ex, na, fx, fa, er, rx)
        backward(na, fx, fa, rx, sigma, ja, eta, clip_a_value, mode_ascent)
     
    logs.append(record(nx, ex, na, er, fx, fa, rx, sigma, ja, item_only=item_only))
    return logs