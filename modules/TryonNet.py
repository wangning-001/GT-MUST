import functools
import torch.nn as nn
from modules import networks


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def define_ILM():
    netILM = networks.STN()
    return netILM

def define_MWM():
    netMWM = networks.Inv()
    return netMWM

def define_GTM(input_nc):
    netGTM = networks.Tryon(input_nc, output_nc=3)
    return netGTM
