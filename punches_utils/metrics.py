import torch
from typing import Dict, List, Union

import os

from torchvision.ops import box_iou

def get_true_positives_false_negatives(
    predictions: torch.Tensor,
    ground_truth: torch.Tensor,
    iou_thresholds: List[float]
) -> torch.Tensor:
    '''
    Computes true positives for the given config of predictions and ground truth
    A match is computed when iou(prediction, ground_truth) > iou_threshold

    Args:
    predictions: [N, 6] tensor. Row = (x1, x2, y1, y2, confidence, class_id)
    ground_truth: [M, 5] tensor. Row = (x1, x2, y1, y2, class_id)
    iou_thresholds: T floats in [0,1]. Used to identify a match between prediction and ground truth boxes.

    Returns:
    a [T, N] tensor specifying whether each prediciton is a true positive for each level of iou
    '''
    iou_val = box_iou(predictions[:, :4], ground_truth[:, :4])
    iou_masks = torch.stack([iou_val > iou_threshold for iou_threshold in iou_thresholds])
    predicted_class = predictions[:, 5]
    ground_truth_class = ground_truth[:, 4]
    classes_match = predicted_class.unsqueeze(1) == ground_truth_class
    tp_mask = torch.bitwise_and(iou_masks, classes_match)
    return tp_mask.any(dim=2), ~tp_mask.any(dim=1)

def precision(
    true_positives: torch.Tensor,
    num_predictions: int
):
    return true_positives.sum(dim=1) / num_predictions

def recall(
    true_positives: torch.Tensor,
    false_negatives: torch.Tensor,
):
    num_true_positives = true_positives.sum(dim=1)
    return num_true_positives / (num_true_positives + false_negatives.sum(dim=1))

def per_class_precision_and_recall(
    classes: List[int],
    true_positives: torch.Tensor,
    false_negatives: torch.Tensor,
    predictions: torch.Tensor,
    ground_truth: torch.Tensor,
    names: List[str],
):
    
    precision_dict = {}
    recall_dict = {}
    for cl in classes:
        predictions_in_class = predictions[:, -1] == cl
        ground_truth_in_class = ground_truth[:, -1] == cl

        precision_dict[names[cl-1]] = precision(true_positives[:, predictions_in_class], predictions_in_class.sum())
        recall_dict[names[cl-1]] = recall(true_positives[:, predictions_in_class], false_negatives[:, ground_truth_in_class])
    return precision_dict, recall_dict


def f1(
    precision_vals: torch.Tensor,
    recall_vals: torch.Tensor
):
    return 2 * (precision_vals * recall_vals) / (precision_vals + recall_vals)

def average_precision(
    precisions: torch.Tensor
):
    return precisions.mean()

def per_class_average_precision(
    per_class_precision_vals: Dict[str, torch.Tensor],
):
    return {cl: average_precision(p) for cl, p in per_class_precision_vals.items()}

def mean_average_precision(
    per_class_average_precision_vals: Dict[str, torch.Tensor]
):
    return sum(per_class_average_precision_vals.values()) / len(per_class_average_precision_vals)

def _print_row(
    cl: str,
    nr: int,
    prec_val: float,
    rec_val: float,
    f1_val: float,
    avg_prec_val: float,
    map_val: Union[float, str],
) -> str:
    if isinstance(map_val, str):
        map_val_f = f"{map_val:<10}"
    else:
        map_val_f = f"{map_val:<10.4f}"
    return f"{cl:<10} {nr:<10} {prec_val:<10.4f} {rec_val:<10.4f} {f1_val:<10.4f} {avg_prec_val:<10.4f} {map_val_f}\n"

def print_metrics(
    num_datapoints: Dict[int, int],
    precision_vals: torch.Tensor,
    recall_vals: torch.Tensor,
    f1_vals: torch.Tensor,
    average_precision_vals: torch.Tensor,
    
    per_class_precision_vals: Dict[str, torch.Tensor],
    per_class_recall_vals: Dict[str, torch.Tensor],
    per_class_f1_vals: Dict[str, torch.Tensor],
    per_class_average_precision_vals: Dict[str, torch.Tensor],
    mean_average_precision_val: torch.Tensor,
    at_iou_val: float = 0.5,
    at_iou_index: int = 0,
    min_iou_val: float = 0.5,
    max_iou_val: float = 0.9,
) -> str:
    prec_label = f"P@{at_iou_val:.2f}"
    rec_label = f"R@{at_iou_val:.2f}"
    f1_label = f"F1@{at_iou_val:.2f}"
    avg_prec_label = f"AP@{min_iou_val:.2f}:{max_iou_val:.2f}"
    
    print_string = ""

    print_string += f"{'Class':<10} {'N':<10} {prec_label:<10} {rec_label:<10} {f1_label:<10} {avg_prec_label:<10} {'mAP':<10}\n"
    print_string +=_print_row("ALL", sum(num_datapoints.values()), precision_vals[at_iou_index].item(), recall_vals[at_iou_index].item(), f1_vals[at_iou_index].item(), average_precision_vals.item(), mean_average_precision_val.item())
    for cl, p in per_class_precision_vals.items():
        print_string += _print_row(str(cl), num_datapoints[cl], p[at_iou_index].item(), per_class_recall_vals[cl][at_iou_index].item(), per_class_f1_vals[cl][at_iou_index].item(), per_class_average_precision_vals[cl].item(), "--")

    return print_string


def run_metrics(
    predictions: torch.Tensor,
    ground_truth: torch.Tensor,
    iou_threshold_precision_recall: float,
    names: List[str],
    save_file: str,
    do_print_metrics: bool = True
) -> Dict[str, torch.Tensor]:
    iou_thresholds = [iou_threshold_precision_recall]

    classes = predictions[:, -1].unique().long()
    num_datapoints = {names[cl.item() -1]: (predictions[:, -1] == cl).sum().item() for cl in classes}

    index_of_iou_threshold_precision_recall = iou_thresholds.index(iou_threshold_precision_recall)
    true_positives, false_negatives = get_true_positives_false_negatives(predictions, ground_truth, iou_thresholds)

    precision_vals = precision(true_positives, predictions.shape[0])
    recall_vals = recall(true_positives, false_negatives)
    f1_vals = f1(precision_vals, recall_vals)
    per_class_precision_vals, per_class_recall_vals = per_class_precision_and_recall(classes, true_positives, false_negatives, predictions, ground_truth, names)
    per_class_f1_vals = {names[cl.item() - 1]: f1(p, r) for cl, p, r in zip(classes, per_class_precision_vals.values(), per_class_recall_vals.values())}

    average_precision_val = average_precision(precision_vals)
    per_class_average_precision_val = per_class_average_precision(per_class_precision_vals)

    mean_average_precision_val = mean_average_precision(per_class_average_precision_val)
    
    metrics_str = print_metrics(
        num_datapoints,
        precision_vals,
        recall_vals,
        f1_vals,
        average_precision_val,
        per_class_precision_vals,
        per_class_recall_vals,
        per_class_f1_vals,
        per_class_average_precision_val,
        mean_average_precision_val,
        at_iou_val = iou_threshold_precision_recall,
        at_iou_index = index_of_iou_threshold_precision_recall,
        min_iou_val = min(iou_thresholds),
        max_iou_val = max(iou_thresholds)
    )

    if do_print_metrics:
        print(metrics_str)
    
    if save_file is not None:
        os.makedirs(os.path.basename(save_file), exist_ok=True)
        with open(save_file, "w") as f:
            f.write(metrics_str)
    
    return average_precision_val






    

