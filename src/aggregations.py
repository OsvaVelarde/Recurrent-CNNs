'''
Title: Definition of information integration mechanism.
Author: Osvaldo M Velarde
Project: Feedback connections in visual system

-------------------------------------------------------------
Description:
I) We implement a function RSZ to change the number of channels of an input.
RSZ can be:
Ia)rsz_ch_linear: a linear function on the channel axis
Ib)rsz_ch_conv: a 3x3 convolution that preserves spatial dimensions and modifies the number of channels.

II) We define three ways of integrating the information:
IIa) aggregate_f1: Apply an RSZ function so that all inputs have the same number of channels.
Then, the result is the sum or the product of all the inputs.
IIb) aggregate_f2: All inputs coming from feedback connections are concatenated. RSZ is applied to the resulting tensor. Finally, the result is
the sum or product of the previous tensor and the feedforward input.
IIc) aggregate_f3: All inputs are concatenated and the RSZ function is applied.
'''

# ===========================================================

import torch.nn as nn
from torchvision.models.resnet import conv3x3
import torch

# ===========================================================
# ============ Resize channels of tensor ====================

class rsz_ch_linear(nn.Module):
    def __init__(self,in_channels, out_channels):
        super(rsz_ch_linear, self).__init__()
        self.operator = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, X):
        return self.operator(X.permute(0,2,3,1)).permute(0,3,1,2)

class rsz_ch_conv3x3(nn.Module):
    def __init__(self,in_channels, out_channels):
        super(rsz_ch_conv3x3, self).__init__()
        self.operator = conv3x3(in_channels, out_channels)

    def forward(self, X):
        return self.operator(X)

# ---------------------------------------------------------

OPTS_RZS = {'linear':rsz_ch_linear, 'conv':rsz_ch_conv3x3}

# ===========================================================

class aggreg_f1(nn.Module):
    def __init__(self,nums_ch_fb, num_ch_ff, rzs = 'linear', agg = 'sum'):
        super(aggreg_f1, self).__init__()
        self.operators = nn.ModuleList([OPTS_RZS[rzs](in_channels, num_ch_ff) for in_channels in nums_ch_fb])
        self.function = torch.sum if agg == 'sum' else torch.prod

    def forward(self, fbs, ff):
    	out = torch.stack([mm(fbs[ii]) for ii, mm in enumerate(self.operators)]+ [ff] ,dim=-1)
    	return self.function(out,dim=-1)

# ---------------------------------------------------------
class aggreg_f2(nn.Module):
    def __init__(self,nums_ch_fb, num_ch_ff, rzs = 'linear', agg = 'sum'):
        super(aggreg_f2, self).__init__()       
        self.operator = OPTS_RZS[rzs](sum(nums_ch_fb), num_ch_ff) if len(nums_ch_fb)>0 else None
        self.function = torch.sum if agg == 'sum' else torch.prod

    def forward(self, fbs, ff):
        yy = [self.operator(torch.cat(fbs,dim=1))] if len(fbs)>0 else []
        out = torch.stack(yy + [ff], dim=-1)
        return self.function(out, dim=-1)

# ---------------------------------------------------------
class aggreg_f3(nn.Module):
    def __init__(self,nums_ch_fb, num_ch_ff, rzs = 'linear', agg = 'sum'):
        super(aggreg_f3, self).__init__()
        self.operator = OPTS_RZS[rzs](sum(nums_ch_fb) + num_ch_ff, num_ch_ff)
        # if sum(nums_ch_fb) == 0: select identity
        self.function = torch.sum if agg == 'sum' else torch.prod

    def forward(self, fbs, ff):
    	return self.operator(torch.cat(fbs + [ff],dim=1))

# ---------------------------------------------------------

OPTS_AGGRS = {'I':aggreg_f1, 'II':aggreg_f2, 'III':aggreg_f3}

# ===========================================================
