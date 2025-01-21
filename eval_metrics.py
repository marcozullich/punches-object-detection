import torch

import argparse

from punches_utils.predictions import predictions_txt_to_array
from punches_utils.labels_utils import read_labels, yolo_to_absolute, ground_truth_to_prediction_format, load_labels_names
from punches_utils.metrics import run_metrics

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_file", type=str, required=True)
    parser.add_argument("--ground_truth_file", type=str, required=True)
    parser.add_argument("--save_file", type=str, required=True)
    parser.add_argument("--yaml_file", type=str, required=True)
    parser.add_argument("--orig_img_width", type=int, required=True)
    parser.add_argument("--orig_img_height", type=int, required=True)
    parser.add_argument("--iou_threshold", type=float, required=True)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    save_file = args.save_file
    predictions_path = args.predictions_file
    ground_truth_path = args.ground_truth_file
    yaml_file = args.yaml_file
    orig_img_width, orig_img_height = args.orig_img_width, args.orig_img_height
    iou_threshold_precision_recall = args.iou_threshold


    predictions = predictions_txt_to_array(predictions_path)
    ground_truth = torch.Tensor(read_labels(ground_truth_path))
    ground_truth = ground_truth_to_prediction_format(ground_truth)
    ground_truth[:,:4] = yolo_to_absolute(ground_truth[:,:4], orig_img_height, orig_img_width)

    names = load_labels_names(yaml_file)
    
    metrics = run_metrics(
        predictions = predictions,
        ground_truth = ground_truth,
        iou_threshold_precision_recall = iou_threshold_precision_recall,
        names = names,
        save_file = save_file,
        do_print_metrics = True
    )
