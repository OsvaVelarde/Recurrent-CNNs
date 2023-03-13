import pickle
import json
import cv2
import matplotlib.pyplot as plt
# ==============================================================
# ==============================================================

COCO_INSTANCE_CATEGORY_NAMES = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
path_imgs = '/media/osvaldo/Seagate Basic/COCO/val2017/'
annfilename = '/media/osvaldo/Seagate Basic/COCO/annotations/instances_val2017.json'
path_preds = '/home/osvaldo/Documents/CCNY/Project_RBP/results_detection/predictions/'

def search_imgpath(idimg, annotations):
    for imgs in annotations['images']:
        if imgs['id'] == idimg:
            return imgs['file_name']
    
    print('ERROR')

# ==============================================================
# ==============================================================

listexps = ['exp_34', 'exp_35' , 'exp_36' , 'exp_37','faster_rcnn_50']
idx_pred = 180 #10, 40, 90, 130, 180

fig,axs = plt.subplots(1,len(listexps),figsize=(15,5))

for idx_exp, exp in enumerate(listexps):
    with open(path_preds + exp + '/' + 'prediction_' + str(idx_pred) + '.pkl', 'rb') as inp:
        predictions = pickle.load(inp)

    annotations = json.load(open(annfilename))

    threshold = 0.5

    for idimg, res in predictions.items():
        img = cv2.imread(path_imgs + search_imgpath(idimg,annotations)) 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 

        pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(res['labels'])]
        pred_boxes = res['boxes']
        pred_scores = list(res['scores'])
        pred_t = [pred_scores.index(x) for x in pred_scores if x>threshold]

        if len(pred_t)>0:
            pred_t = pred_t[-1]
            pred_boxes = pred_boxes[:pred_t+1]
            pred_class = pred_class[:pred_t+1]

            for i in range(len(pred_boxes)):
                cv2.rectangle(img, (int(pred_boxes[i][0]), int(pred_boxes[i][1])),(int(pred_boxes[i][2]), int(pred_boxes[i][3])), color=(0, 255, 0), thickness=3) 
                cv2.putText(img,pred_class[i], (int(pred_boxes[i][0]), int(pred_boxes[i][1]-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, lineType=cv2.LINE_AA)

    axs[idx_exp].imshow(img) 
    axs[idx_exp].set_xticks([])
    axs[idx_exp].set_yticks([])

fig.savefig(path_preds + 'figs/' + 'idx_' + str(idx_pred) + '.svg')
#plt.show()