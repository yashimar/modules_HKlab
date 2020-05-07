import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact
from matplotlib.widgets import Slider
import matplotlib.animation as animation
import math
import time


def fk(L, th):
    l1, l2 = L
    th1, th2 = th
    x1 = l1 * math.cos(th1)
    y1 = l1 * math.sin(th1)
    x2 = x1 + l2 * math.cos(th1 + th2)
    y2 = y1 + l2 * math.sin(th1 + th2)

    return np.array([[0, 0], [x1, y1], [x2, y2]])

L = [1.0, 1.0]
def update(i, x1, x2, trajects, isgrid, size=2.1):
    if i!=0:
        plt.cla()
    plt.xlim([-size, size])
    plt.ylim([-size, size])
    if isgrid: plt.grid()
      
    th1 = x1[i]
    th2 = x2[i]
    p = fk(L, [th1,th2])
    trajects.append(p[-1])
    
    graph, = plt.plot(p.T[0], p.T[1])
    graph.set_data(p.T[0], p.T[1])
    graph.set_linestyle('-')
    graph.set_linewidth(5)
    graph.set_marker('o')
    graph.set_markerfacecolor('g')
    graph.set_markeredgecolor('g')
    graph.set_markersize(15)  
    
def draw(commands, save_path, length=20, size=2.1, return_traj=False, isgrid=True):
    t1 = time.time() 
    fig = plt.figure(figsize=(5,5))
    
    trajects = []
    n_step = len(commands)//2
    x1_traj = np.zeros(1)
    x2_traj = np.zeros(1)
    for i in range(n_step):
        x1_traj = np.hstack( (x1_traj, np.linspace(x1_traj[-1], x1_traj[-1]+commands[2*i], length)) )
        x2_traj = np.hstack( (x2_traj, np.linspace(x2_traj[-1], x2_traj[-1]+commands[2*i+1], length)) )

    ani = animation.FuncAnimation(
        fig, update, fargs = (x1_traj,x2_traj,trajects,isgrid,size,), 
        interval = 50, frames = len(x1_traj))
    ani.save(save_path, writer = 'imagemagick')
    t2 = time.time() 
    elapsed_time = t2-t1
    print(f"drawing time: {elapsed_time:.3f}s")
    
    if return_traj: return trajects