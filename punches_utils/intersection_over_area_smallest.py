import torch
from torch import Tensor
from typing import Tuple

def box_area(boxes: Tensor) -> Tensor:
    """Calculate the area of bounding boxes."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    return (x2 - x1) * (y2 - y1)

def _box_inter_area(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """Calculate the intersection area of two sets of bounding boxes."""
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]
    
    wh = rb - lt  # Width and Height of intersection
    wh = wh.clamp(min=0)  # Clamp to avoid negative values
    return wh[:, :, 0] * wh[:, :, 1]  # [N, M] - Area of intersection

def _box_smallest_area(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """Calculate the area of the smallest bounding box for each pair."""
    area1 = box_area(boxes1)  # Shape: [M]
    area2 = box_area(boxes2)  # Shape: [N]
    return torch.min(area1[:, None], area2)  # Shape: [N, M] - Broadcasting

def ioas(boxes1: Tensor, boxes2: Tensor) -> Tuple[Tensor, Tensor]:
    """Calculate Intersection over Area of the Smallest for two sets of boxes."""
    inter_area = _box_inter_area(boxes1, boxes2).float()  # Shape: [N, M]
    smallest_area = _box_smallest_area(boxes1, boxes2).float()  # Shape: [N, M]
    
    # Avoid division by zero by creating a mask
    valid_mask = smallest_area > 0
    ioas_values = torch.zeros(inter_area.shape, device=inter_area.device)
    
    # Compute IoAS only where the smallest area is valid
    ioas_values[valid_mask] = inter_area[valid_mask] / smallest_area[valid_mask]
    
    return ioas_values


