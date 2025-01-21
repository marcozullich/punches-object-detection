import punches_utils.predictions as P
from PIL import Image
import cv2

import os

import numpy as np

from psd_tools import PSDImage

import torch
from torchvision.ops import nms

from punches_utils.img_utils import draw_predictions
from punches_utils.labels_utils import load_labels_names, shift_x2y2_bboxes

import argparse


CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD_NMS = .9
DEBUG = True

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_path", type=str, required=True)
    parser.add_argument("--image_info_paths", type=str, nargs="+", required=True)
    parser.add_argument("--conf_threshold", type=float, required=True)
    parser.add_argument("--no_nms_suppression", action="store_true", default=False)
    parser.add_argument("--iou_threshold_nms", type=float, required=True)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    predictions_path = args.predictions_path

    predictions = P.predictions_json_split_by_name(predictions_path)


    for image_info_path in args.image_info_paths:
        image_info = P.load_image_info(image_info_path)
        image_name = os.path.splitext(os.path.basename(image_info_path))[0]
        image_name = image_name[:image_name.rfind("_")]

        
        labels_names = load_labels_names("dataset.yaml")

        predictions_image = P.predictions_json_to_array(predictions[image_name])

        postprocessed_predictions = []

        for image_id, prediction_data in predictions_image.items():
            bboxes = prediction_data["bboxes"]
            scores = prediction_data["scores"]
            labels = prediction_data["labels"]
            
            
            postprocessed_predictions += P.pack_predictions(image_id, bboxes, scores, labels)


        merged_predictions = P.merge_predictions_to_image(image_info, postprocessed_predictions)

        merged_bboxes, merged_scores, merged_labels = P.unpack_predictions(merged_predictions)
        merged_bboxes = torch.from_numpy(np.array(merged_bboxes))
        merged_scores = torch.from_numpy(np.array(merged_scores))
        merged_labels = torch.from_numpy(np.array(merged_labels))

        kept_indices = P.remove_low_confidence_predictions(merged_scores, args.conf_threshold)

        merged_bboxes = merged_bboxes[kept_indices]
        merged_scores = merged_scores[kept_indices]
        merged_labels = merged_labels[kept_indices]

        merged_bboxes = shift_x2y2_bboxes(merged_bboxes)

        discarded_indices = []
        if not args.no_nms_suppression:
            discarded_indices = P.nms_post(merged_bboxes, merged_labels, args.iou_threshold_nms)
        

        
        kept_positions = torch.ones(len(merged_bboxes), dtype=torch.bool)
        if len(discarded_indices) > 0:
            kept_positions[discarded_indices] = False

        kept_bboxes = merged_bboxes[kept_positions]
        kept_scores = merged_scores[kept_positions]
        kept_labels = merged_labels[kept_positions]

        kept_bboxes = kept_bboxes.cpu().numpy()
        kept_scores = kept_scores.cpu().numpy()
        kept_labels = kept_labels.cpu().numpy()

        kept_predictions = P.pack_predictions(0, kept_bboxes, kept_scores, kept_labels)

        predictions_save_path = os.path.join(
            os.path.dirname(predictions_path),
            f"conf_{args.conf_threshold}_nms_{args.iou_threshold_nms}"
        )
        os.makedirs(predictions_save_path, exist_ok=True)
        predictions_save_file = os.path.join(
            predictions_save_path,
            f"{image_name}{'NONMS' if args.no_nms_suppression else ''}.txt"
        )
        with open(predictions_save_file, "w") as f:
            for pred in kept_predictions:
                f.write(" ".join(str(p) for p in pred[1:]) + "\n")
    
        print(f"Predictions for {image_name} saved to {predictions_save_file}")

    
