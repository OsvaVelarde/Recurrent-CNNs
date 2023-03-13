from torch import sigmoid
from torch import tensor
import torch
import torch.nn as nn
from torchvision.models.resnet import conv3x3

class op_rgcell(nn.Module):
    def __init__(self,in_channels, out_channels, name, gamma_init):
        super(op_rgcell, self).__init__()
        self.operator = conv3x3(in_channels, out_channels) if name == 'conv' else nn.Linear(in_channels, out_channels,bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.name = name
        self.bn.weight = nn.Parameter(self.bn.weight * gamma_init)
        nn.init.zeros_(self.operator.weight)

    def forward(self, X):
        out = self.operator(X) if self.name =='conv' else self.operator(X.permute(0,2,3,1)).permute(0,3,1,2)
        out = self.bn(out)
        return out

# ------------------------------------------------------------------

class time_decay_cell(nn.Module):
    def __init__(self, inp_channels, mm_channels, st_channels):
        super(time_decay_cell, self).__init__()
        self.temporal_parameter = nn.Parameter(tensor([0.0]))

    def forward(self,input_dyn,internal_state,internal_memory):
        new_state  = self.temporal_parameter * internal_state + input_dyn
        new_memory = None
        return new_state, new_memory

# ------------------------------------------------------------------

class recipgated_cell(nn.Module):
    def __init__(self, inp_channels, mm_channels, st_channels):
        super(recipgated_cell, self).__init__()

        assert inp_channels == st_channels, "Num of Channels of Input should be equal to Num of Channels of Internal State"

        name = 'conv'
        self.op_inp_mm = op_rgcell(inp_channels,mm_channels,name,gamma_init=1.0) #gamma =0.0

        self.op_tau_mm = op_rgcell(mm_channels,mm_channels,name,gamma_init=1.0) #gamma =0.1
        self.op_gat_mm = op_rgcell(st_channels,mm_channels,name,gamma_init=1.0) #gamma =0.1
        self.op_tau_st = op_rgcell(st_channels,st_channels,name,gamma_init=1.0) #gamma =0.1
        self.op_gat_st = op_rgcell(mm_channels,st_channels,name,gamma_init=1.0) #gamma =0.1

        self.activation = nn.ELU()

        #N, C, H, W = 20, 5, 10, 10
        #self.layernorm = nn.LayerNorm([C, H, W])
        #layer normalization using 1,0

    def forward(self,input_dyn,internal_state,internal_memory):
        # *Recibir el num de canales (mismo para memory/state)
        # *fb_input es la integracion de todos los feedbacks.

        # * integracion fb_input y ff_input!!! 
        #     aplicar conv2d to fb_input (parecido a case 2 de mis opciones)
        #     sumar ff_input


        tau_mm = sigmoid(self.op_tau_mm(internal_memory))
        gat_mm = sigmoid(self.op_gat_mm(internal_state ))
        tau_st = sigmoid(self.op_tau_st(internal_state ))
        gat_st = sigmoid(self.op_gat_st(internal_memory))

        new_memory = (1 - tau_mm) * internal_memory + (1 - gat_mm) * self.op_inp_mm(input_dyn)
        new_state  = (1 - tau_st) * internal_state  + (1 - gat_st) * input_dyn

        #new_memory = self.layernorm(new_memory)         
        new_memory = self.activation(new_memory)
        new_state  = self.activation(new_state)
        
        return new_state, new_memory

# ------------------------------------------------------------------