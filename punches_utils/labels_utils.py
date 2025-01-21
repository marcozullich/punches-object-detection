from typing import Tuple, List, Any

import yaml

import torch
from torchvision.ops import box_iou

def label_to_relative(x_center: int, y_center: int, delta_x: int, delta_y: int, img_height: int, img_width: int) -> Tuple[float, float, float, float]:
    x_center /= img_width
    y_center /= img_height
    delta_x /= img_width
    delta_y /= img_height

    return x_center, y_center, delta_x, delta_y

def label_to_absolute(x_center: float, y_center: float, delta_x: float, delta_y: float, img_height: int, img_width: int) -> Tuple[int, int, int, int]:
    x_center *= img_width
    y_center *= img_height
    delta_x *= img_width
    delta_y *= img_height

    return int(x_center), int(y_center), int(delta_x), int(delta_y)


def read_labels(file: str, sep: str = " ") -> List[Tuple[int, float, float, float, float]]:
    labels = []
    with open(file, "r") as f:
        for label_line in f:
            label = label_line.strip().split(sep)
            label[0] = int(label[0])
            label[1] = float(label[1])
            label[2] = float(label[2])
            label[3] = float(label[3])
            label[4] = float(label[4])
            labels.append(label)
        return labels

def save_labels(labels: List[Any], destination_file: str, sep: str = " ", add_row_id: bool = False) -> None:
    i = 0
    with open(destination_file, "w") as f:
        for i, label in enumerate(labels):
            if add_row_id:
                label = [i] + list(label)
            f.write(sep.join(str(lab) for lab in label) + "\n")
            i += 1

def load_labels_names(path_to_yaml):
    with open(path_to_yaml, "r") as f:
        yaml_content = yaml.safe_load(f)
    return yaml_content["names"]

def shift_x2y2_bboxes(
    bboxes: torch.Tensor
) -> torch.Tensor:
    bboxes[:, 2] += bboxes[:, 0]
    bboxes[:, 3] += bboxes[:, 1]
    return bboxes

def yolo_to_absolute(bboxes: torch.Tensor, img_height: int, img_width: int) -> torch.Tensor:
    bboxes[:, (0, 2)] *= img_width
    bboxes[:, (1, 3)] *= img_height
    bboxes[:, 0] = bboxes[:,0] - bboxes[:,2] / 2
    bboxes[:, 1] = bboxes[:,1] - bboxes[:,3] / 2
    bboxes[:, 2] = bboxes[:,0] + bboxes[:,2]
    bboxes[:, 3] = bboxes[:,1] + bboxes[:,3]
    return bboxes

def ground_truth_to_prediction_format(ground_truth: torch.Tensor) -> torch.Tensor:
    '''
    Shifts the ground truth from the format
    CLASS_ID X_TOP Y_TOP X_BOTTOM Y_BOTTOM
    to the format 
    X_TOP Y_TOP X_BOTTOM Y_BOTTOM CLASS_ID
    '''
    perm = (1, 2, 3, 4, 0)
    return ground_truth[:,perm]