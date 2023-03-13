import torch
from models import NETWORK_CLASSIFICATION, NETWORK_SEGMENTATION
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision

OPTS_TASKS = {'classification': NETWORK_CLASSIFICATION, 'detection':NETWORK_SEGMENTATION}

def initialize_model(parameters):

	model = OPTS_TASKS[parameters['task']](**parameters["cfg"])

	if parameters["device_type"] == 'gpu':
		model = torch.nn.DataParallel(model)
		cudnn.benchmark = True

	epoch = 0
	best_acc = 0
	
	if parameters['model_path']:
		checkpoint = torch.load(parameters["model_path"])
		model.load_state_dict(checkpoint["model"],strict=False)
		epoch = checkpoint['epoch']
		best_acc = checkpoint['acc'] if 'acc' in checkpoint else None

	if (parameters["device_type"] == "gpu") and torch.has_cudnn:
		device = torch.device("cuda:{}".format(parameters["gpu_number"]))
	else:
		device = torch.device("cpu")

	model = model.to(device)
	return model, device, epoch, best_acc

def initialize_optimizer(model,parameters,task):
	name = parameters["name"]
	cfg = parameters["cfg"]
	optimizer = torch.optim.SGD(model.parameters(), **cfg) if name=='SGD' else torch.optim.Adam(model.parameters(), **cfg)

	if task == 'classification':
		scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200) #Classif

	if task == 'detection':
		scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[16, 22], gamma=0.1)  #detection
	
	#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
	if parameters["optimizer_path"]:
		checkpoint = torch.load(parameters["optimizer_path"])
		optimizer.load_state_dict(checkpoint['optimizer'])
		scheduler.load_state_dict(checkpoint["lr_scheduler"])

	return optimizer, scheduler

# -----------------------------------------------------------------
# -----------------------------------------------------------------

def upload_resnet_imagenet(model,device,path_models,version=18):
    parameters = torch.load(path_models + 'resnet' + str(version) + '-imagenet1k.pth')

    model.conv1.weight = nn.Parameter(parameters['conv1.weight'].to(device),requires_grad=False)
    model.bn1.weight = nn.Parameter(parameters['bn1.weight'].to(device),requires_grad=False)
    model.bn1.bias = nn.Parameter(parameters['bn1.bias'].to(device),requires_grad=False)
    model.bn1.running_mean = nn.Parameter(parameters['bn1.running_mean'].to(device),requires_grad=False)
    model.bn1.running_var = nn.Parameter(parameters['bn1.running_var'].to(device),requires_grad=False)

    for ll, layer in enumerate(model.rn_dynamics_list):
    	for bb, block in enumerate(layer.last_step):
    		block.conv1.weight = nn.Parameter(parameters['layer'+ str(ll+1) + '.' + str(bb) + '.conv1.weight'].to(device),requires_grad=False)
    		block.bn1.weight = nn.Parameter(parameters['layer'+ str(ll+1) + '.' + str(bb) + '.bn1.weight'].to(device),requires_grad=False)
    		block.bn1.bias = nn.Parameter(parameters['layer'+ str(ll+1) + '.' + str(bb) + '.bn1.bias'].to(device),requires_grad=False)
    		block.bn1.running_mean = nn.Parameter(parameters['layer'+ str(ll+1) + '.' + str(bb) + '.bn1.running_mean'].to(device),requires_grad=False)
    		block.bn1.running_var = nn.Parameter(parameters['layer'+ str(ll+1) + '.' + str(bb) + '.bn1.running_var'].to(device),requires_grad=False)
    		block.conv2.weight = nn.Parameter(parameters['layer'+ str(ll+1) + '.' + str(bb) + '.conv2.weight'].to(device),requires_grad=False)
    		block.bn2.weight = nn.Parameter(parameters['layer'+ str(ll+1) + '.' + str(bb) + '.bn2.weight'].to(device),requires_grad=False)
    		block.bn2.bias = nn.Parameter(parameters['layer'+ str(ll+1) + '.' + str(bb) + '.bn2.bias'].to(device),requires_grad=False)
    		block.bn2.running_mean = nn.Parameter(parameters['layer'+ str(ll+1) + '.' + str(bb) + '.bn2.running_mean'].to(device),requires_grad=False)
    		block.bn2.running_var = nn.Parameter(parameters['layer'+ str(ll+1) + '.' + str(bb) + '.bn2.running_var'].to(device),requires_grad=False)
    	if ll > 0:
    		layer.last_step[0].downsample[0].weight = nn.Parameter(parameters['layer'+ str(ll+1) + '.0.downsample.0.weight'].to(device),requires_grad=False)
    		layer.last_step[0].downsample[1].weight = nn.Parameter(parameters['layer'+ str(ll+1) + '.0.downsample.1.weight'].to(device),requires_grad=False)
    		layer.last_step[0].downsample[1].bias = nn.Parameter(parameters['layer'+ str(ll+1) + '.0.downsample.1.bias'].to(device),requires_grad=False)
    		layer.last_step[0].downsample[1].running_mean = nn.Parameter(parameters['layer'+ str(ll+1) + '.0.downsample.1.running_mean'].to(device),requires_grad=False)
    		layer.last_step[0].downsample[1].running_var = nn.Parameter(parameters['layer'+ str(ll+1) + '.0.downsample.1.running_var'].to(device),requires_grad=False)

# -----------------------------------------------------------------
# -----------------------------------------------------------------
def upload_original_fasterrcnn(model,device,path_models):
	parameters = torch.load(path_models + 'fasterrcnn_resnet50_fpn_coco.pth')

#	print(model)
#TO-DO:reduce lines
# ESTA FORMA NO FUNCIONA:
#	for name_pp, pp  in model.roi_heads.box_predictor.named_parameters():
#		pp = nn.Parameter(parameters['roi_heads.box_predictor.' + name_pp].to(device),requires_grad=True)

	model.backbone.body.conv1.weight = nn.Parameter(parameters['backbone.body.conv1.weight'].to(device),requires_grad=False)
	model.backbone.body.bn1.weight = nn.Parameter(parameters['backbone.body.bn1.weight'].to(device),requires_grad=False)
	model.backbone.body.bn1.bias = nn.Parameter(parameters['backbone.body.bn1.bias'].to(device),requires_grad=False)
	model.backbone.body.bn1.running_mean = nn.Parameter(parameters['backbone.body.bn1.running_mean'].to(device),requires_grad=False)
	model.backbone.body.bn1.running_var = nn.Parameter(parameters['backbone.body.bn1.running_var'].to(device),requires_grad=False)

	for ll, layer in enumerate(model.backbone.body.rn_dynamics_list):
		for bb, block in enumerate(layer.last_step):
			block.conv1.weight = nn.Parameter(parameters['backbone.body.layer'+ str(ll+1) + '.' + str(bb) + '.conv1.weight'].to(device),requires_grad=False)
			block.bn1.weight = nn.Parameter(parameters['backbone.body.layer'+ str(ll+1) + '.' + str(bb) + '.bn1.weight'].to(device),requires_grad=False)
			block.bn1.bias = nn.Parameter(parameters['backbone.body.layer'+ str(ll+1) + '.' + str(bb) + '.bn1.bias'].to(device),requires_grad=False)
			block.bn1.running_mean = nn.Parameter(parameters['backbone.body.layer'+ str(ll+1) + '.' + str(bb) + '.bn1.running_mean'].to(device),requires_grad=False)
			block.bn1.running_var = nn.Parameter(parameters['backbone.body.layer'+ str(ll+1) + '.' + str(bb) + '.bn1.running_var'].to(device),requires_grad=False)
			block.conv2.weight = nn.Parameter(parameters['backbone.body.layer'+ str(ll+1) + '.' + str(bb) + '.conv2.weight'].to(device),requires_grad=False)
			block.bn2.weight = nn.Parameter(parameters['backbone.body.layer'+ str(ll+1) + '.' + str(bb) + '.bn2.weight'].to(device),requires_grad=False)
			block.bn2.bias = nn.Parameter(parameters['backbone.body.layer'+ str(ll+1) + '.' + str(bb) + '.bn2.bias'].to(device),requires_grad=False)
			block.bn2.running_mean = nn.Parameter(parameters['backbone.body.layer'+ str(ll+1) + '.' + str(bb) + '.bn2.running_mean'].to(device),requires_grad=False)
			block.bn2.running_var = nn.Parameter(parameters['backbone.body.layer'+ str(ll+1) + '.' + str(bb) + '.bn2.running_var'].to(device),requires_grad=False)
			block.conv3.weight = nn.Parameter(parameters['backbone.body.layer'+ str(ll+1) + '.' + str(bb) + '.conv3.weight'].to(device),requires_grad=False)
			block.bn3.weight = nn.Parameter(parameters['backbone.body.layer'+ str(ll+1) + '.' + str(bb) + '.bn3.weight'].to(device),requires_grad=False)
			block.bn3.bias = nn.Parameter(parameters['backbone.body.layer'+ str(ll+1) + '.' + str(bb) + '.bn3.bias'].to(device),requires_grad=False)
			block.bn3.running_mean = nn.Parameter(parameters['backbone.body.layer'+ str(ll+1) + '.' + str(bb) + '.bn3.running_mean'].to(device),requires_grad=False)
			block.bn3.running_var = nn.Parameter(parameters['backbone.body.layer'+ str(ll+1) + '.' + str(bb) + '.bn3.running_var'].to(device),requires_grad=False)
			if ll > 0:
				layer.last_step[0].downsample[0].weight = nn.Parameter(parameters['backbone.body.layer'+ str(ll+1) + '.0.downsample.0.weight'].to(device),requires_grad=False)
				layer.last_step[0].downsample[1].weight = nn.Parameter(parameters['backbone.body.layer'+ str(ll+1) + '.0.downsample.1.weight'].to(device),requires_grad=False)
				layer.last_step[0].downsample[1].bias = nn.Parameter(parameters['backbone.body.layer'+ str(ll+1) + '.0.downsample.1.bias'].to(device),requires_grad=False)
				layer.last_step[0].downsample[1].running_mean = nn.Parameter(parameters['backbone.body.layer'+ str(ll+1) + '.0.downsample.1.running_mean'].to(device),requires_grad=False)
				layer.last_step[0].downsample[1].running_var = nn.Parameter(parameters['backbone.body.layer'+ str(ll+1) + '.0.downsample.1.running_var'].to(device),requires_grad=False)

	model.rpn.head.conv.weight = nn.Parameter(parameters['rpn.head.conv.weight'].to(device),requires_grad=True)
	model.rpn.head.conv.bias = nn.Parameter(parameters['rpn.head.conv.bias'].to(device),requires_grad=True)
	model.rpn.head.cls_logits.weight = nn.Parameter(parameters['rpn.head.cls_logits.weight'].to(device),requires_grad=True)
	model.rpn.head.cls_logits.bias = nn.Parameter(parameters['rpn.head.cls_logits.bias'].to(device),requires_grad=True)
	model.rpn.head.bbox_pred.weight = nn.Parameter(parameters['rpn.head.bbox_pred.weight'].to(device),requires_grad=True)
	model.rpn.head.bbox_pred.bias = nn.Parameter(parameters['rpn.head.bbox_pred.bias'].to(device),requires_grad=True)

	model.roi_heads.box_head.fc6.weight = nn.Parameter(parameters['roi_heads.box_head.fc6.weight'].to(device),requires_grad=True)
	model.roi_heads.box_head.fc6.bias = nn.Parameter(parameters['roi_heads.box_head.fc6.bias'].to(device),requires_grad=True)
	model.roi_heads.box_head.fc7.weight = nn.Parameter(parameters['roi_heads.box_head.fc7.weight'].to(device),requires_grad=True)
	model.roi_heads.box_head.fc7.bias = nn.Parameter(parameters['roi_heads.box_head.fc7.bias'].to(device),requires_grad=True)

	model.roi_heads.box_predictor.cls_score.weight = nn.Parameter(parameters['roi_heads.box_predictor.cls_score.weight'].to(device),requires_grad=True)
	model.roi_heads.box_predictor.cls_score.bias = nn.Parameter(parameters['roi_heads.box_predictor.cls_score.bias'].to(device),requires_grad=True)
	model.roi_heads.box_predictor.bbox_pred.weight = nn.Parameter(parameters['roi_heads.box_predictor.bbox_pred.weight'].to(device),requires_grad=True)
	model.roi_heads.box_predictor.bbox_pred.bias = nn.Parameter(parameters['roi_heads.box_predictor.bbox_pred.bias'].to(device),requires_grad=True)


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):

    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)  
