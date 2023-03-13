'''
Referece: https://arxiv.org/pdf/1803.06396.pdf
Reviving and Improving Recurrent Back-Propagation

'''

from torch.autograd import grad
from torch import rand_like, ones_like
import torch
# ------------------------------------------------------------------------
def RBP(params_dynamic,state_last,state_2nd_last,grad_state_last,
        truncate_iter=50,method='Neumann_RBP'):

  """ Two variants of Recurrent Back Propagation:
    1, RBP
    2, Neumann-RBP
    Context:
      Dynamic system: 
      Considere N - estados ocultos independientes.

      h_i,t = F_i( h_i,(t-1) , w_i ) 
      Loss = G (h_1,T, h_2,T,..., h_N,T) 

      Last (convergent at time step T) state: h(T)

    Args:
      params_dynamic:         list, parameters                                [w_1, w_,2, ..., w_N]
      state_last:             list, last state,                               [h_1,T h_2,T ... h_N,T]
      state_2nd_last:         list, 2nd last state,                           [h_1,T-1 h_2,T-1 ... h_N,T-1]
      grad_state_last:        list, gradient of loss w.r.t. last state,       [dl/dh_1(T) dl/dh_2(T) ... dl/dh_N(T)]
      truncate_iter:          int, truncation iteration
      method:                 str, specification of rbp method
    
    Returns:
      grad: tuple, gradient of loss w.r.t. parameters                         [dl/dw_1, dl/dw_2, ..., dl/dw_N] 

    N.B.:
      1, 2nd last state h(T-1) must be detached from the computation graph
  """
  LIST_OPTIONS_RBP = {'n': nRBP, 'o': tRBP, 'f':tRBP_feedbacks}
  assert method in LIST_OPTIONS_RBP, "Nonsupported RBP method {}".format(rbp_method)
  method_rbp = LIST_OPTIONS_RBP.get(method)
  z_star = method_rbp(state_last, state_2nd_last, grad_state_last, truncate_iter)
  return grad(state_last,params_dynamic,grad_outputs=z_star, retain_graph=True,allow_unused=True)


# ------------------------------------------------------------------------

def nRBP(state_last, state_2nd_last, grad_state_last, truncate_iter):
  neumann_g = None
  neumann_v = None
  neumann_g_prev = grad_state_last
  neumann_v_prev = grad_state_last
  aux = [None for pp in state_last]

  for ii in range(truncate_iter):

    for j in range(len(state_2nd_last)):
      aux[j] = grad(state_last, state_2nd_last[j], grad_outputs=neumann_v_prev, retain_graph=True, allow_unused=True)[0]

    neumann_v = aux
    neumann_g = [x + y for x, y in zip(neumann_g_prev, neumann_v)]
    neumann_v_prev = neumann_v
    neumann_g_prev = neumann_g

  z_star = neumann_g
  return z_star

# ------------------------------------------------------------------------

def tRBP(state_last, state_2nd_last, grad_state_last, truncate_iter):
  z_T = [rand_like(pp) for pp in state_last] # Random initialization for each dimension "i"

  for ii in range(truncate_iter):
    z_T = grad(state_last, state_2nd_last, grad_outputs=z_T, retain_graph=True, allow_unused=True)
    z_T = list(z_T)

    for j in range(len(z_T)):
      z_T[j] = z_T[j] + grad_state_last[j] if grad_state_last[j] is not None else z_T[j]

    # #z_T = [x + y for x, y in zip(z_T, grad_state_last)]

    #print('z_star,ii:',ii,[torch.max(xx) for xx in z_T])

  z_star = z_T

  return z_star

# ------------------------------------------------------------------------

# ------------------------------------------------------------------------


# CHECK HE FUNCTIONS
# def tRBP_coupled_states(state_last, state_2nd_last, grad_state_last, truncate_iter):
#   '''
# Observacion:
# * Los estados h_{l} pueden tener distintas dimensiones.

#     Args:
#       state_last:             list, last state,                               [h_1,T h_2,T ... h_N,T]
#       state_2nd_last:         list, 2nd last state,                           [h_1,T-1 h_2,T-1 ... h_N,T-1]
#       grad_state_last:        list, gradient of loss w.r.t. last state,       [dl/dh_1(T) dl/dh_2(T) ... dl/dh_N(T)]
#       truncate_iter:          int, truncation iteration
# REVISAR. CREO QUE HACE LO MISMO QUE tRBP_independent states
#   '''

#   z_T = [rand_like(pp) for pp in state_last] # Random initialization for each dimension "i"
#   aux = [None for pp in state_last]

#   prueba = grad(state_last, state_2nd_last, grad_outputs=z_T, retain_graph=True, allow_unused=True)

#   for ii in range(truncate_iter):
#     aux = [grad(state_last, state_2nd_last[j], grad_outputs=z_T, retain_graph=True, allow_unused=True)[0] for j in range(len(state_2nd_last))]
#     z_T = aux

#     for j in range(len(z_T)):
#       z_T[j] = z_T[j] + grad_state_last[j] if grad_state_last[j] is not None else z_T[j]
#     #z_T = [x + y for x, y in zip(z_T, grad_state_last)]

  
#   z_star = z_T
#   return z_star

# ------------------------------------------------------------------------

def tRBP_feedbacks(state_last, state_2nd_last, grad_state_last, truncate_iter):

  '''
  Suponga el sistema acoplado de la forma:

  h_{l,t} = F_l( h_{l-1,t} ,h_{l,t-1}, h_{l+1,t-1}, ..., h_{N,t-1}) 

El estado oculto h_{l,t} de layer l at time t depends on:
  -feedforward component: h_{l-1,t} desde la capa anterior
  -internal dynamic: h_{l,t-1}
  -feedback of next layers: h_{l+1,t-1}, h_{l+2,t-1}, ..., h_{N,t-1}
  
    Args:
      state_last:             list, last state,                               [h_1,T h_2,T ... h_N,T]
      state_2nd_last:         list, 2nd last state,                           [h_1,T-1 h_2,T-1 ... h_N,T-1]
      grad_state_last:        list, gradient of loss w.r.t. last state,       [dl/dh_1(T) dl/dh_2(T) ... dl/dh_N(T)]
      truncate_iter:          int, truncation iteration

  '''
  z_T = [rand_like(pp) for pp in state_last] # Random initialization for each dimension "i"
  aux = [None for pp in state_last]

  for ii in range(truncate_iter):
    for j in range(len(state_2nd_last)-1):
      aux[j] = grad(state_last[:j+2], state_2nd_last[j], grad_outputs=z_T[:j+2], retain_graph=True, allow_unused=True)[0]

    aux[-1] = grad(state_last, state_2nd_last[-1], grad_outputs=z_T, retain_graph=True, allow_unused=True)[0]

    z_T = aux
    z_T = [x + y for x, y in zip(z_T, grad_state_last)]

  z_star = z_T
  return z_star

# ------------------------------------------------------------------------

def detach_param_with_grad(params):
  params_dst = [pp.detach() for pp in params]

  for pp in params_dst:
    pp.requires_grad = True

  return params_dst

# ------------------------------------------------------------------------

def CLPenalty(state_last, state_2nd_last, tau=0.95, compute_hessian=True):
  norm_1_vect = [ones_like(pp) for pp in state_last]

  vj_prod = grad(state_last, state_2nd_last, norm_1_vect,
         retain_graph=True,create_graph=compute_hessian,allow_unused=True)

  vj_penalty = 0 

  for vj in vj_prod:
    vj_penalty += ((vj-tau).clamp(0) ** 2).sum().sqrt()

  return vj_penalty