'''
Title: Calculation and update of the gradient (RBP - BP)
Author: Osvaldo M Velarde
Project: Feedback connections in visual system

-------------------------------------------------------------
References:

'''

# ==============================================================

import torch
import rbp

# ==============================================================

def training_rbp(module,loss,state_last,state_2nd_last, new_state_last):
    tuple_predict = [(nn, pp) for nn, pp in module.named_parameters() if 'convnet' not in nn and pp.requires_grad==True]
    tuple_convnet = [(nn, pp) for nn, pp in module.named_parameters() if 'convnet' in nn and pp.requires_grad==True]

    name_predict, params_predict = zip(*tuple_predict)
    name_convnet, params_convnet = zip(*tuple_convnet)

    if module.contractor:
    	loss_p = loss + rbp.CLPenalty(state_last,state_2nd_last)
    else:
    	loss_p = loss

    grad_predict = torch.autograd.grad(loss_p, params_predict,retain_graph=True)
    grad_state_last = torch.autograd.grad(loss_p, new_state_last, retain_graph=True,allow_unused=True)
    grad_convnet = rbp.RBP(params_convnet, state_last, state_2nd_last, grad_state_last, module.truncate_iter, module.typeRBP)

    grad = {}

    for nn, gg in zip(name_predict, grad_predict):
        grad[nn] = gg

    for nn, gg in zip(name_convnet, grad_convnet):
        grad[nn] = gg

    return grad

# =============================================================================

def training_bp(module,loss,state_last,state_2nd_last, new_state_last):
    tuple_params = [(nn, pp) for nn, pp in module.named_parameters() if pp.requires_grad==True]
    name_params, params = zip(*tuple_params)
    grad_values = torch.autograd.grad(loss, params, retain_graph=True,allow_unused=True)            
    grad = {}

    for nn, gg in zip(name_params, grad_values):
        grad[nn] = gg

    return grad

# =============================================================================

def update_grad_rbp(model,loss,grad):
    for nn, pp in model.named_parameters():
        if pp.requires_grad==True:
            pp.grad = grad[nn[7:]]

def update_grad_bp(model,loss,grad):
    loss.backward()