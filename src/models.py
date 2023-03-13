import collections as col

import torch.nn as nn

from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection import FasterRCNN

from training_functions import training_rbp, training_bp
import rbp
from torch.autograd import set_detect_anomaly

from RecCNN import RecResNet

OPTS_TRAINING = {'BP':training_bp,'RBP':training_rbp}

# =============================================================================
# ========================== CLASSIFICATION ====================================

class NETWORK_CLASSIFICATION(nn.Module):
    def __init__(self,input_channels,num_filters,
                first_layer_kernel_size,first_layer_conv_stride,first_layer_padding,
                first_pool_size,first_pool_stride,first_pool_padding,
                blocks_per_layer_list,block_strides_list,block_fn,
                growth_factor,frozenBN,
                time_steps,feedback_connections,cfg_agg,
                rnn_cell,idxs_cell,
                num_classes, loss_function, bio_computation,
                lr_alg = 'BP', typeRBP='o', truncate_iter = 50, contractor=False):
        
        super(NETWORK_CLASSIFICATION,self).__init__()

        dim_images = 32
        dim_images = dim_images // first_layer_conv_stride
        dim_images = dim_images // first_pool_stride
        for ii in block_strides_list:
            dim_images = dim_images//ii

        self.backbone = RecResNet(input_channels=input_channels,num_filters=num_filters,
                        first_layer_kernel_size=first_layer_kernel_size,first_layer_conv_stride=first_layer_conv_stride,first_layer_padding=first_layer_padding,
                        first_pool_size=first_pool_size,first_pool_stride=first_pool_stride,first_pool_padding=first_pool_padding,
                        blocks_per_layer_list=blocks_per_layer_list, block_strides_list=block_strides_list,block_fn=block_fn,
                        growth_factor=growth_factor,frozenBN=frozenBN,
                        time_steps = time_steps,feedback_connections = feedback_connections, cfg_agg = cfg_agg, 
                        rnn_cell = rnn_cell, idxs_cell=idxs_cell,
                        lr_alg = lr_alg,
                        bio_computation = bio_computation)

        self.num_features_last_conv = self.backbone.num_features_last_conv
        self.final_bn = nn.BatchNorm2d(self.num_features_last_conv)
        self.relu = nn.ReLU()
        
        self.pooling = nn.MaxPool2d(dim_images)

        self.mlp = nn.Linear(self.num_features_last_conv,num_classes)

        self.loss_function = loss_function
        self.lr_alg = lr_alg
        self.lr_function = OPTS_TRAINING[lr_alg]
        self.typeRBP = typeRBP
        self.truncate_iter = truncate_iter
        self.contractor = contractor
        self.dropout = nn.Dropout(p=0.3)

    def forward(self,x, target, batch_idx=0):
        # RN Stage ----------------
        state_2nd_last, state_last = self.backbone(x,batch_idx)

        if 'RBP' in self.lr_alg:
            new_state_last = rbp.detach_param_with_grad(state_last)
            h=self.final_bn(new_state_last[-1])
        else:
            new_state_last = None
            h=self.final_bn(state_last[-1])

        #filename = './results/temporal_outputs/exp_03_target_' + "{:02d}".format(batch_idx) + '.pt'
        #torch.save(target, filename)

        # Predictor Stage ---------
        h=self.relu(h)
        h=self.pooling(h)
        #h=self.dropout(h)
        h=self.mlp(h.view(-1,self.num_features_last_conv))
        
        # Loss function -------------
        loss = self.loss_function(h,target)

        if self.training and 'RBP' in self.lr_alg:
            with set_detect_anomaly(True):
                grad = self.lr_function(self,loss,state_last,state_2nd_last, new_state_last)
        else:
            grad = None

        return h, loss, grad

# =============================================================================
# =============================================================================

# =============================================================================
# ======================= SEGMENTATION ========================================

class NETWORK_SEGMENTATION(nn.Module):
    def __init__(self,input_channels,num_filters,
                first_layer_kernel_size,first_layer_conv_stride,first_layer_padding,
                first_pool_size,first_pool_stride,first_pool_padding,
                blocks_per_layer_list,block_strides_list,block_fn,
                growth_factor,frozenBN,
                time_steps,feedback_connections,cfg_agg,
                rnn_cell,idxs_cell,
                num_classes, loss_function, bio_computation,
                lr_alg = 'BP', typeRBP='o', truncate_iter = 50, contractor=False):

        super(NETWORK_SEGMENTATION,self).__init__()

        out_channels_backbone = 256

        backbone = RecResNetbackbone(input_channels,num_filters,
                first_layer_kernel_size,first_layer_conv_stride,first_layer_padding,
                first_pool_size,first_pool_stride,first_pool_padding,
                blocks_per_layer_list,block_strides_list, block_fn,
                growth_factor,frozenBN,
                time_steps,feedback_connections,cfg_agg,rnn_cell,idxs_cell, 
                lr_alg, 
                bio_computation,
                out_channels_backbone)

        anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)

        anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)

        roi_pooler = MultiScaleRoIAlign(featmap_names=[str(ii) for ii in range(len(blocks_per_layer_list))],
                                        output_size=7, 
                                        sampling_ratio=2)

        self.segmenter = FasterRCNN(backbone, num_classes=num_classes, rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler)


    def forward(self,x, target, batch_idx=0):
        h = self.segmenter(x,target)
        return h

class RecResNetbackbone(nn.Module):
    def __init__(self,input_channels,num_filters,
                first_layer_kernel_size,first_layer_conv_stride,first_layer_padding,
                first_pool_size,first_pool_stride,first_pool_padding,
                blocks_per_layer_list,block_strides_list,block_fn,
                growth_factor,frozenBN,
                time_steps,feedback_connections,cfg_agg,rnn_cell,idxs_cell, 
                lr_alg, 
                bio_computation,
                out_channels_backbone):

        super(RecResNetbackbone,self).__init__()

        self.body = RecResNet(input_channels=input_channels,num_filters=num_filters,
                        first_layer_kernel_size=first_layer_kernel_size,first_layer_conv_stride=first_layer_conv_stride,first_layer_padding=first_layer_padding,
                        first_pool_size=first_pool_size,first_pool_stride=first_pool_stride,first_pool_padding=first_pool_padding,
                        blocks_per_layer_list=blocks_per_layer_list, block_strides_list=block_strides_list,block_fn=block_fn,
                        growth_factor=growth_factor,frozenBN=frozenBN,
                        time_steps = time_steps,feedback_connections = feedback_connections, cfg_agg = cfg_agg, 
                        rnn_cell = rnn_cell, idxs_cell=idxs_cell,
                        lr_alg = lr_alg,
                        bio_computation = bio_computation)

        self.rz_ch_list = nn.ModuleList([nn.Conv2d(num_filters * self.body.block.expansion *  2**ii,out_channels_backbone,3,1,1) for ii in range(len(blocks_per_layer_list))])
        self.add_new_feat = nn.MaxPool2d(1,2,0)
        self.out_channels = out_channels_backbone

    def forward(self,x):
        _, h = self.body(x,0)
        extra_features = 5 - len(h)

        output = []

        for ii, value in enumerate(h):
            key = str(ii)
            tensor = self.rz_ch_list[ii](value)
            output.append((key,tensor))

        tensor = self.add_new_feat(tensor)

        for jj in range(extra_features):
            key = str(ii+1+jj)
            output.append((key,tensor))
            tensor = self.add_new_feat(tensor)

        return col.OrderedDict(output)


# ---------------------------------------------


# USAR ESTO PARA BN ---> FBN
# import torch
# import torchvision
# n1 = torch.nn.BatchNorm2d(16).eval()
# n2 = torchvision.ops.FrozenBatchNorm2d(16).eval()
# n2.load_state_dict(n1.state_dict(), strict=False)
# x = torch.rand(5,16,2,2)

# print(n1.__dict__)
# print(n2.__dict__)
# print(torch.equal(n1(x), n2(x)))


# CHECK como inicializar los pesos de una resnet sin entrenar
# nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")


# class OutputLayer(nn.Module):
#     def __init__(self, in_features, output_shape):
#         super(OutputLayer, self).__init__()
#         if not isinstance(output_shape, (list, tuple)):
#             output_shape = [output_shape]
#         self.output_shape = output_shape
#         self.flattened_output_shape = int(np.prod(output_shape))
#         self.fc_layer = nn.Linear(in_features, self.flattened_output_shape)

#     def forward(self, x):
#         h = self.fc_layer(x)
#         if len(self.output_shape) > 1:
#             h = h.view(h.shape[0], *self.output_shape)
#         h = F.log_softmax(h, dim=-1)
#         return h