import collections as col

import torch
import torch.nn as nn
from torchvision.ops import FrozenBatchNorm2d

from rnn_cells import time_decay_cell, recipgated_cell
from aggregations import aggreg_f1, aggreg_f2, aggreg_f3
from layers import BasicBlockV1, BasicBlockV2, Bottleneck


OPTS_BLOCK = {'V1':BasicBlockV1,'V2':BasicBlockV2, 'Bottleneck':Bottleneck}
OPTS_RNN_CELLS = {'time_decay': time_decay_cell, 'recip_gated':recipgated_cell}
OPTS_AGGRS = {'I':aggreg_f1, 'II':aggreg_f2, 'III':aggreg_f3}

class RecResNet(nn.Module):
    """
    ResNet with Feedback
    """
    # -------------------------------------------------------------

    def __init__(self,
                 input_channels, num_filters,
                 first_layer_kernel_size, first_layer_conv_stride,first_layer_padding,
                 first_pool_size, first_pool_stride, first_pool_padding,
                 blocks_per_layer_list, block_strides_list, block_fn,             
                 growth_factor, frozenBN,
                 time_steps,feedback_connections, cfg_agg,
                 rnn_cell, idxs_cell,
                 lr_alg, bio_computation):


        super(RecResNet, self).__init__()

        self.typeBN = FrozenBatchNorm2d if frozenBN else nn.BatchNorm2d

        # Temporal Evolution  ------------------------------------
        self.time_steps = time_steps
        self.lr_alg = lr_alg

        # Expose attributes for downstream dimension computation -
        self.num_filters = num_filters
        self.growth_factor = growth_factor

        # Modules of 1st-Stage -----------------------------------
        self.inplanes = num_filters

        self.conv1 = nn.Conv2d(
            in_channels=input_channels, out_channels=num_filters,
            kernel_size=first_layer_kernel_size,
            stride=first_layer_conv_stride,
            padding=first_layer_padding,
            bias=False)
        self.bn1 = self.typeBN(num_filters)
        self.relu = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(
            kernel_size=first_pool_size,
            stride=first_pool_stride,
            padding=first_pool_padding)

        # Type of Block
        self.block = OPTS_BLOCK[block_fn]

        # Modules of Layers/Feedforward --------------------------
        self.num_layers = len(blocks_per_layer_list)
        current_num_filters = num_filters

        # Modules of Feedbacks -----------------------------------
        fb_strides_list = [block_strides_list[0]]

        for x in block_strides_list[1:]:
            fb_strides_list.append(fb_strides_list[-1]*x) 
        
        self.rn_dynamics_list = nn.ModuleList()

        # ------------------------------------------------------
        # Create layers and feedbacks

        for i, (num_blocks, stride) in enumerate(zip(blocks_per_layer_list, block_strides_list)):

            # Cfgs of feedbacks 
            channels =  current_num_filters * self.block.expansion
            feedback_connections_per_layer = [(connection,{'in_channels': int(channels * growth_factor**(connection[0]-connection[1])), 
                'out_channels':channels,
                'kernel_size':3,
                'stride':fb_strides_list[connection[0]-connection[1]],
                'padding':1,
                'bias':False}) for connection in feedback_connections if connection[1]==i]

            # Construction of Feedforward layers
            fforward_layer = self._make_layer(
                block=self.block,
                planes=current_num_filters,
                blocks=num_blocks,
                stride=stride,
            )

            # Construction of 'rn_dynamic' modules
            self.rn_dynamics_list.append(RNDynamic(fforward_layer = fforward_layer,
                                                   feedbacks=feedback_connections_per_layer,
                                                   nonlinear=True,
                                                   internal_transformation=False,
                                                   cfg_agg=cfg_agg,
                                                   rnn_cell=rnn_cell))

            if (time_steps == 1) or (i not in idxs_cell):
                for zz, pp in self.rn_dynamics_list[i].rnn_dynamic.named_parameters():
                    pp.requires_grad = False

            current_num_filters *= growth_factor

        # Information for module of Last Stage ----------------------------------
        self.num_features_last_conv = channels

        # Forward function ---------------------------------------
        self.set_forward(bio_computation)


    def _make_layer(self, block, planes, blocks, stride=1):

        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                        nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), 
                        self.typeBN(planes * block.expansion))

        layers_ = [('0',block(self.inplanes, planes, stride, downsample, BN=self.typeBN))]
        self.inplanes = planes * block.expansion

        for i in range(1, blocks):
            layers_.append((str(i),block(self.inplanes, planes, BN=self.typeBN)))
      
        return nn.Sequential(col.OrderedDict(layers_))

    # -------------------------------------------------------------
    def set_forward(self,bio_computation):
        if bio_computation:
            self.forward = self._forward_bio
        else:
            self.forward = self._forward_nobio

    # -------------------------------------------------------------

    def _forward_bio(self,X,batch_idx):
        # Biological computation: Feedforward connection with delay
        '''
        # Input: Tensor (N,Ch,H,W)
        Equation
        Reference --

        '''

        # 1st stage ----------------------------------------------
        h = self.conv1(X)
        h = self.bn1(h)
        h = self.relu(h)
        h = self.pool1(h) #15 x 15

        # 2st stage: Dynamic in layers ---------------------------
        # Definition of hidden states ----------------------------
        prev_states = [h] + [None for x in range(self.num_layers)]
        states = [None for x in range(self.num_layers)]
        prev_memories = [None for x in range(self.num_layers)]
        memories = [None for x in range(self.num_layers)]

        # Temporal evolution -------------------------------------
        for time in range(self.time_steps):
            prev_states[1:] = states.copy()
            prev_memories = memories.copy()

            if time == self.time_steps-1 and 'RBP' in self.lr_alg:
                prev_states = rbp.detach_param_with_grad(prev_states)

            for i in range(min(self.num_layers,time+1)):
                states[i], memories[i] = self.rn_dynamics_list[i](prev_states[i],prev_states[i+1:],prev_memories[i])    # See function RNDynamic

            #filename = './results/temporal_outputs/exp_03_batch_' + "{:02d}".format(batch_idx) + '_t_' + "{:02d}".format(time) +'.pt'
            #torch.save(states[-1], filename)

        return prev_states[1:], states

    # -------------------------------------------------------------

    def _forward_nobio(self,X,batch_idx):
        # Non-biological computation: Feedforward connection without delay

        '''
        # Input: Tensor (N,Ch,H,W)
        Equation
        Reference --

        '''

        # 1st stage ----------------------------------------------
        h = self.conv1(X)
        h = self.bn1(h)
        h = self.relu(h)
        h = self.pool1(h)
        
        # 2st stage: Dynamic in layers ---------------------------
        # Definition of hidden states ----------------------------
        prev_states = [None for x in range(self.num_layers)]
        states = [h] + [None for x in range(self.num_layers)]
        prev_memories = [None for x in range(self.num_layers)]
        memories = [None for x in range(self.num_layers)]

        # Temporal evolution -------------------------------------
        for time in range(self.time_steps):
            prev_states = states[1:].copy()
            prev_memories = memories.copy()

            if time == self.time_steps-1 and 'RBP' in self.lr_alg:
                prev_states = rbp.detach_param_with_grad(prev_states)

            for i in range(self.num_layers):
                states[i+1], memories[i] = self.rn_dynamics_list[i](states[i],prev_states[i:],prev_memories[i])    # See function RNDynamic

            #filename = './results/temporal_outputs/exp_03_batch_' + "{:02d}".format(batch_idx) + '_t_' + "{:02d}".format(time) +'.pt'
            #torch.save(states[-1], filename)

        return prev_states, states[1:]

# =============================================================================


class RNDynamic(nn.Module):
    def __init__(self,fforward_layer,feedbacks,nonlinear,internal_transformation, cfg_agg, rnn_cell):

        # order_integration: 'before', 'after' then FF operation.
        # rsz_type: 'linear', 'conv' (rsz channels)
        # agg_type: 'I', 'II', 'III'
        # agg_function: 'sum'. 'prod'

        super(RNDynamic, self).__init__()

        # Aggregation information 
        order = cfg_agg['order']
        rzs_type = cfg_agg['rsz']
        agg_type = cfg_agg['type']
        agg_function = cfg_agg['function']

        # Operations for h(s_L-1,t) ---------------------------------------------------
        self.first_step, self.last_step = (nn.Identity(),fforward_layer) if order == 'before' else (fforward_layer, nn.Identity())
        planes = fforward_layer[0].conv1.in_channels if order =='before' else fforward_layer[0].conv1.out_channels

        # Feedbacks information (s_L+k, t-1) ------------------------------------------
        self.num_feedbacks    = len(feedbacks)
        self.fb_connections   = [cfg[0] for cfg in feedbacks]
        self.info_connections = [cfg[1] for cfg in feedbacks]
        
        # Aggregator (s_L-1,t and s_L+k,t-1) ------------------------------------------
        self.aggregator = OPTS_AGGRS[agg_type]([x['in_channels'] for x in self.info_connections], planes, rzs_type, agg_function)
        self.activation = nn.ReLU() if nonlinear else nn.Identity()

        # RNN dynamic -----------------------------------------------------------------
        dim_rnn = fforward_layer[0].conv1.out_channels
        self.rnn_dynamic = OPTS_RNN_CELLS[rnn_cell](dim_rnn, dim_rnn, dim_rnn)

    def forward(self,fforward_state,prev_states,prev_memory):

        # Aggregate information of feedbacks ---------------------------------------
        fforward_component = self.first_step(fforward_state)
        Ba,Ch,H,W = fforward_component.shape

        feedback_component = [torch.zeros(Ba,x['in_channels'],H,W).cuda() for x in self.info_connections] if self.num_feedbacks > 0 else []

        for i, connection in enumerate(self.fb_connections):
            distance = connection[0]-connection[1]
            if prev_states[distance] is not None:
                out_fback = self.activation(prev_states[distance])  # Batch x Channels_or x H_or x W_or
                out_fback = torch.nn.functional.interpolate(out_fback, size=fforward_component.size()[2:], mode='bilinear', align_corners=False)  # Batch x Channels_or x H_new x W_new
                feedback_component[i] = out_fback
        
        # ------------------------
        h = self.aggregator(feedback_component,fforward_component)
        h = self.last_step(h)

        # Dynamic of states/memory -------------------------------------------------- 
        internal_state  = prev_states[0] if prev_states[0] is not None else torch.zeros_like(h)
        internal_memory = prev_memory if prev_memory is not None else torch.zeros_like(h)

        new_state, new_memory = self.rnn_dynamic(h,internal_state,internal_memory)
        return new_state, new_memory