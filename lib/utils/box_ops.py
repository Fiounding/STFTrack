import torch
from torchvision.ops.boxes import box_area
import numpy as np


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
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
    area1 = box_area(boxes1)  # (N,)
    area2 = box_area(boxes2)  # (N,)

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
    iou, union = box_iou(boxes1, boxes2)  # (N,)

    lt = torch.min(boxes1[:, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # (N,2)
    area = wh[:, 0] * wh[:, 1]  # (N,)

    return iou - (area - union) / area, iou

def clip_box(box: list, H, W, margin=0):
    x1, y1, w, h = box
    x2, y2 = x1 + w, y1 + h
    x1 = min(max(0, x1), W - margin)
    x2 = min(max(margin, x2), W)
    y1 = min(max(0, y1), H - margin)
    y2 = min(max(margin, y2), H)
    w = max(margin, x2 - x1)
    h = max(margin, y2 - y1)
    return [x1, y1, w, h]

def ciou_loss(boxes1, boxes2):
    """
    :param boxes1: (N, 4) (x1,y1,x2,y2)
    :param boxes2: (N, 4) (x1,y1,x2,y2)
    :return: loss: scalar, iou: scalar
    """
    # assert boxes1.shape == boxes2.shape, "boxes1 and boxes2 must have the same shape"
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)  # (N,)

    x1, y1, x2, y2 = boxes1.unbind(dim=-1)
    w1 = x2 - x1
    h1 = y2 - y1
    cx1 = (x1 + x2) / 2
    cy1 = (y1 + y2) / 2

    x1_gt, y1_gt, x2_gt, y2_gt = boxes2.unbind(dim=-1)
    w2 = x2_gt - x1_gt
    h2 = y2_gt - y1_gt
    cx2 = (x1_gt + x2_gt) / 2
    cy2 = (y1_gt + y2_gt) / 2

    rho2 = (cx2 - cx1).pow(2) + (cy2 - cy1).pow(2)

    lt = torch.min(boxes1[:, :2], boxes2[:, :2])  # (N,2)
    rb = torch.max(boxes1[:, 2:], boxes2[:, 2:])  # (N,2)
    c_diag2 = (rb - lt).pow(2).sum(dim=1)  # (N,)
    c_diag2 = torch.clamp(c_diag2, min=1e-6)  # 防止除零

    v = (4 / (math.pi ** 2)) * (torch.atan(w2 / (h2 + 1e-6)) - torch.atan(w1 / (h1 + 1e-6))).pow(2)

    with torch.no_grad():
        S = (iou > 0.5).float()
        alpha = S * v / (1 - iou + v + 1e-6)  # 添加epsilon避免除零

    ciou = iou - (rho2 / c_diag2) - alpha * v
    ciou = torch.clamp(ciou, min=-1.0, max=1.0)

    loss = (1 - ciou).mean()
    iou = iou.mean()

    return loss, iou