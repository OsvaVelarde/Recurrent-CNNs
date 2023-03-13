'''
References:
'''
# ===========================================================

import time
from pickle import dump, HIGHEST_PROTOCOL

import torch
from torchvision.models.detection import MaskRCNN, KeypointRCNN

from coco_utils import get_coco_api_from_dataset
from coco_eval import CocoEvaluator
from utils import MetricLogger

# ===========================================================

def _get_iou_types(model):
    model_without_ddp = model
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types

# ===========================================================

@torch.no_grad()
def evaluate(model, data_loader, device, path_preds=None):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    index_preds = 0

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)

        torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images,None)
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]

        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        
        # Save predictions
        if path_preds is not None:
            with open(path_preds + 'prediction_' + str(index_preds) + '.pkl', 'wb') as outp:
                dump(res, outp, HIGHEST_PROTOCOL)
        index_preds+=1

        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator

# ===========================================================
