import os
import sys
sys.path.append("..")

def check_dir(fn):
    if not os.path.exists(fn):
        os.makedirs(fn)

def get_learning_rate(optimizer):
    lr=[]
    for param_group in optimizer.param_groups:
        lr +=[ param_group['lr'] ]
    assert(len(lr)==1)

    return lr[0]