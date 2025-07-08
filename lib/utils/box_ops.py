import torch
from torchvision.ops.boxes import box_area
import numpy as np


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

def box_cxcywh_to_xywh(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         w, h]
    return torch.stack(b, dim=-1)

def box_xywh_to_xyxy(x):
    x1, y1, w, h = x.unbind(-1)
    b = [x1, y1, x1 + w, y1 + h]
    return torch.stack(b, dim=-1)


def box_xyxy_to_xywh(x):
    x1, y1, x2, y2 = x.unbind(-1)
    b = [x1, y1, x2 - x1, y2 - y1]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
'''Note that this function only supports shape (N,4)'''


def box_iou(boxes1, boxes2):
    """

    :param boxes1: (N, 4) (x1,y1,x2,y2)
    :param boxes2: (N, 4) (x1,y1,x2,y2)
    :return:
    """
    area1 = box_area(boxes1) # (N,)
    area2 = box_area(boxes2) # (N,)

    lt = torch.max(boxes1[:, :2], boxes2[:, :2])  # (N,2)
    rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])  # (N,2)

    wh = (rb - lt).clamp(min=0)  # (N,2)
    inter = wh[:, 0] * wh[:, 1]  # (N,)

    union = area1 + area2 - inter

    iou = inter / union
    return iou, union


'''Note that this implementation is different from DETR's'''


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    boxes1: (N, 4)
    boxes2: (N, 4)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    # try:
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2) # (N,)

    lt = torch.min(boxes1[:, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # (N,2)
    area = wh[:, 0] * wh[:, 1] # (N,)

    return iou - (area - union) / area, iou


def giou_loss(boxes1, boxes2):
    """

    :param boxes1: (N, 4) (x1,y1,x2,y2)
    :param boxes2: (N, 4) (x1,y1,x2,y2)
    :return:
    """
    giou, iou = generalized_box_iou(boxes1, boxes2)
    return (1 - giou).mean(), iou


def clip_box(box: list, H, W, margin=0):
    x1, y1, w, h = box
    x2, y2 = x1 + w, y1 + h
    x1 = min(max(0, x1), W-margin)
    x2 = min(max(margin, x2), W)
    y1 = min(max(0, y1), H-margin)
    y2 = min(max(margin, y2), H)
    w = max(margin, x2-x1)
    h = max(margin, y2-y1)
    return [x1, y1, w, h]

def clip_box_batch(boxes, sizes, margin=0):
    H = sizes[:, 0]
    W = sizes[:, 1]
    
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    w = boxes[:, 2]
    h = boxes[:, 3]
    
    x2 = x1 + w
    y2 = y1 + h
    
    x1 = torch.min(torch.max(torch.tensor(0.0), x1), W - margin)
    x2 = torch.min(torch.max(torch.tensor(margin, dtype=torch.float32), x2), W)
    y1 = torch.min(torch.max(torch.tensor(0.0), y1), H - margin)
    y2 = torch.min(torch.max(torch.tensor(margin, dtype=torch.float32), y2), H)
    
    w = torch.max(torch.tensor(margin, dtype=torch.float32), x2 - x1)
    h = torch.max(torch.tensor(margin, dtype=torch.float32), y2 - y1)
    
    return torch.stack([x1, y1, w, h], axis=1)

#0710
def map_boxes_back(sample_target_box, pred_box, resize_factor, search_size=256):
    cx_prev, cy_prev = sample_target_box[0] + 0.5 * sample_target_box[2], sample_target_box[1] + 0.5 * sample_target_box[3]
    cx, cy, w, h = pred_box
    half_side = 0.5 * search_size / resize_factor
    cx_real = cx + (cx_prev - half_side)
    cy_real = cy + (cy_prev - half_side)
    return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

def map_boxes_back_batch(sample_target_boxes, pred_boxes, resize_factors, search_size=256):
    # 确保输入是PyTorch张量
    sample_target_boxes = torch.tensor(sample_target_boxes)
    pred_boxes = torch.tensor(pred_boxes)
    resize_factors = torch.tensor(resize_factors).view(-1)  # 将 (64, 1) 变为 (64,)
    
    cx_prev = sample_target_boxes[:, 0] + 0.5 * sample_target_boxes[:, 2]
    cy_prev = sample_target_boxes[:, 1] + 0.5 * sample_target_boxes[:, 3]
    
    cx = pred_boxes[:, 0]
    cy = pred_boxes[:, 1]
    w = pred_boxes[:, 2]
    h = pred_boxes[:, 3]
    
    half_side = 0.5 * search_size / resize_factors
    cx_real = cx + (cx_prev - half_side)
    cy_real = cy + (cy_prev - half_side)
    
    return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], axis=1)

#0719
# def convert_to_bbox(cx, cy, w, h):
#     """将中心坐标和宽高转换为边界框坐标 (x_min, y_min, x_max, y_max)"""
#     x_min = cx - w / 2
#     y_min = cy - h / 2
#     x_max = cx + w / 2
#     y_max = cy + h / 2
#     return torch.stack([x_min, y_min, x_max, y_max], dim=-1)

# def convert_to_cxcywh(x_min, y_min, x_max, y_max):
#     """将边界框坐标 (x_min, y_min, x_max, y_max) 转换为中心坐标和宽高 (cx, cy, w, h)"""
#     cx = (x_min + x_max) / 2
#     cy = (y_min + y_max) / 2
#     w = x_max - x_min
#     h = y_max - y_min
#     return torch.stack([cx, cy, w, h], dim=-1)

# def iou(box1, box2):
#     """计算两个边界框的交并比（IoU）。"""
#     x1 = max(box1[0], box2[0])
#     y1 = max(box1[1], box2[1])
#     x2 = min(box1[2], box2[2])
#     y2 = min(box1[3], box2[3])
    
#     intersection = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
#     box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
#     box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
#     union = box1_area + box2_area - intersection
    
#     return intersection / union

# def bbox_voting(bboxes, scores, device, iou_threshold=0.5):
#     """对一组边界框进行bbox voting，考虑分数。"""
#     selected_box = None
#     max_weighted_score = -1
    
#     for i in range(bboxes.shape[0]):
#         cx_i, cy_i, w_i, h_i = bboxes[i]
#         box_i = convert_to_bbox(cx_i, cy_i, w_i, h_i).squeeze()
#         # weighted_sum = np.zeros(4)
#         weighted_sum = torch.zeros((4), device=device)
#         # total_weight = 0
#         total_weight = torch.zeros((1), device=device)
        
#         for j in range(bboxes.shape[0]):
#             cx_j, cy_j, w_j, h_j = bboxes[j]
#             box_j = convert_to_bbox(cx_j, cy_j, w_j, h_j).squeeze()
#             if iou(box_i, box_j) >= iou_threshold:
#                 weight = scores[j]
#                 # weighted_sum += np.array([cx_j, cy_j, w_j, h_j]) * weight
#                 weighted_sum += bboxes[j] * weight
#                 total_weight += weight
        
#         weighted_score = total_weight
#         if weighted_score > max_weighted_score:
#             max_weighted_score = weighted_score
#             selected_box = weighted_sum / total_weight
    
#     return selected_box

# def batch_bbox_voting(batch_bboxes, batch_scores, iou_threshold=0.5):
#     """对批量图像的边界框进行bbox voting，考虑分数。"""
#     batch_size = batch_bboxes.shape[0]
#     device = batch_bboxes.device
#     # result_bboxes = np.zeros((batch_size, 1, 4))
#     result_bboxes = torch.zeros((batch_size, 1, 4), device=device)
    
#     for i in range(batch_size):
#         best_bbox = bbox_voting(batch_bboxes[i], batch_scores[i], device, iou_threshold)
#         result_bboxes[i, 0] = best_bbox
    
#     return result_bboxes

#0720
import multiprocessing as mp

def convert_to_bbox(cx, cy, w, h):
    """将中心坐标和宽高转换为边界框坐标 (x_min, y_min, x_max, y_max)"""
    x_min = cx - w / 2
    y_min = cy - h / 2
    x_max = cx + w / 2
    y_max = cy + h / 2
    return torch.stack([x_min, y_min, x_max, y_max], dim=-1)

def convert_to_cxcywh(x_min, y_min, x_max, y_max):
    """将边界框坐标 (x_min, y_min, x_max, y_max) 转换为中心坐标和宽高 (cx, cy, w, h)"""
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    w = x_max - x_min
    h = y_max - y_min
    return torch.stack([cx, cy, w, h], dim=-1)

def iou(box1, box2):
    """计算两个边界框的交并比（IoU）。"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    union = box1_area + box2_area - intersection
    
    return intersection / union

# def bbox_voting(bboxes, scores, iou_threshold=0.5):
#     """对一组边界框进行bbox voting，考虑分数。"""
#     selected_box = None
#     max_weighted_score = -1
    
#     for i in range(bboxes.shape[0]):
#         cx_i, cy_i, w_i, h_i = bboxes[i]
#         box_i = convert_to_bbox(cx_i, cy_i, w_i, h_i).squeeze()
#         weighted_sum = torch.zeros((4))
#         total_weight = torch.zeros((1))
        
#         for j in range(bboxes.shape[0]):
#             cx_j, cy_j, w_j, h_j = bboxes[j]
#             box_j = convert_to_bbox(cx_j, cy_j, w_j, h_j).squeeze()
#             #0720
#             # if iou(box_i, box_j) >= iou_threshold:
#             if iou(box_i, box_j) >= iou_threshold and scores[i] >=0.2 and scores[j] >=0.2:
#                 weight = scores[j]
#                 weighted_sum += bboxes[j] * weight
#                 total_weight += weight
        
#         weighted_score = total_weight
#         if weighted_score > max_weighted_score:
#             max_weighted_score = weighted_score
#             selected_box = weighted_sum / total_weight
#         #0720
#         #if no box to fuse, choose the first one as result
#         if selected_box == None:
#             selected_box = bboxes[0]
    
#     return selected_box

#0721
def bbox_voting(bboxes, scores, iou_threshold=0.6):
    """对一组边界框进行bbox voting"""
    weight = 0
    selected_box = torch.zeros((4))
    vote = False
    # weight += scores[0]
    # selected_box = bboxes[0] * weight
    # max_weighted_score = -1

    # cx_0, cy_0, w_0, h_0 = bboxes[0]
    # box_0 = convert_to_bbox(cx_0, cy_0, w_0, h_0).squeeze()

    #0722
    bboxes_xyxy = bboxes.clone()
    for i in range(bboxes_xyxy.shape[0]):
        cx_i, cy_i, w_i, h_i = bboxes_xyxy[i]
        bboxes_xyxy[i] = convert_to_bbox(cx_i, cy_i, w_i, h_i).squeeze()
    
    for j in range(bboxes_xyxy.shape[0]):
        # weighted_sum = torch.zeros((4))
        # total_weight = torch.zeros((1))

        # cx_j, cy_j, w_j, h_j = bboxes[j]
        # box_j = convert_to_bbox(cx_j, cy_j, w_j, h_j).squeeze()
        #0720
        # if iou(box_i, box_j) >= iou_threshold:
        #0723_1
        # if iou(bboxes_xyxy[0], bboxes_xyxy[j]) >= iou_threshold and scores[j] >= 0.01:
        #0723_2
        if iou(bboxes_xyxy[0], bboxes_xyxy[j]) >= iou_threshold and scores[j] >= 0.2:
            vote = True
            weight += scores[j]
            selected_box += bboxes_xyxy[j] * scores[j]
            # total_weight += weight

        # weighted_score = total_weight
        # if weighted_score > max_weighted_score:
        #     max_weighted_score = weighted_score
        #     selected_box = weighted_sum / total_weight
        #0720
        #if no box to fuse, choose the first one as result
    if vote == False:
        selected_box = bboxes[0]
    else:
        selected_box = selected_box / weight
        xmin, ymin, xmax, ymax = selected_box
        selected_box = convert_to_cxcywh(xmin, ymin, xmax, ymax)
    
    return selected_box

def process_single_image(bboxes, scores, iou_threshold):
    """处理单张图像的边界框投票"""
    return bbox_voting(bboxes, scores, iou_threshold)

# def batch_bbox_voting(batch_bboxes, batch_scores, iou_threshold=0.5):
#     """对批量图像的边界框进行bbox voting，考虑分数。"""
#     # 将数据复制到CPU
#     batch_bboxes = batch_bboxes.cpu()
#     batch_scores = batch_scores.cpu()
#
#     batch_size = batch_bboxes.shape[0]
#     result_bboxes = torch.zeros((batch_size, 1, 4))
#
#     # 使用多处理进行并行计算
#     num_workers = 8
#     with mp.Pool(num_workers) as pool:
#         results = pool.starmap(process_single_image, [(batch_bboxes[i], batch_scores[i], iou_threshold) for i in range(batch_size)])
#
#     for i, best_bbox in enumerate(results):
#         result_bboxes[i, 0] = best_bbox
#
#     return result_bboxes

def batch_bbox_voting(batch_bboxes, batch_scores, iou_threshold=0.5):
    """对批量图像的边界框进行bbox voting，考虑分数。"""
    # 将数据复制到CPU
    batch_bboxes = batch_bboxes.cpu()
    batch_scores = batch_scores.cpu()
    batch_size = batch_bboxes.shape[0]
    result_bboxes = torch.zeros((batch_size, 1, 4))

    for i in range(batch_size):
        result_bboxes[i, 0] = bbox_voting(batch_bboxes[i], batch_scores[i], iou_threshold)


    return result_bboxes