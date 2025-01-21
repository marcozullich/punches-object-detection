from torchvision.ops.boxes import batched_nms as torch_batched_nms
from torchvision.ops.boxes import nms as torch_nms
from torchvision.ops import box_iou, box_area

import numpy as np

import torch
import json

from punches_utils.intersection_over_area_smallest import ioas

def remove_low_confidence_predictions(scores, conf_threshold):
    return scores >= conf_threshold

def nms(
    bboxes: list[np.ndarray],
    scores: list[float],
    labels: list[int],
    device: str,
    iou: float = 0.5,
    iou_class_agnostic: float = 0.8
) -> tuple[list[np.ndarray], list[float], list[int]]:
    """Performs non-maximum suppression on the given bboxes, scores, and labels.

    Args:
        bboxes (list[np.ndarray]):
        A list of NumPy arrays with the shape [x1,y1,x2,y2].
        scores (list[float]): A list of confidence scores for each bbox,
        in the same order as bboxes.
        labels (list[int]): A list of label indexes for each bbox,
        in the same order as bboxes.
        device (str): the device on which to execute the computation
        iou (float): the intersection over union threshold [default 0.5]
        iou_class_agnostic (float): the class-agnostic intersection over union threshold [default 0.8]

    Returns:
        tuple[list[np.ndarray], list[float], list[int]]:
        A tuple (bboxes, scores, labels) with the same format as the input.
    """
    # convert to tensor first
    # bboxes = torch.from_numpy(np.asarray(bboxes)).to(device)  # [N, 4]
    # scores = torch.from_numpy(np.asarray(scores)).to(device)  # [N]
    # labels = torch.from_numpy(np.asarray(labels)).to(device)  # [N]

    bboxes = bboxes.to(device)
    scores = scores.to(device)
    labels = labels.to(device)
    
    to_keep = torch_batched_nms(
        boxes=bboxes,
        scores=scores,
        idxs=labels,
        iou_threshold=iou,
    )
    
    bboxes = bboxes[to_keep, :]
    scores = scores[to_keep]
    labels = labels[to_keep]
    
    to_keep = torch_nms(bboxes, scores, iou_class_agnostic)
    
    return to_keep

def merge_predictions_to_image(
    frame_info,
    predictions
):
    merged_predictions = []
    
    for prediction in predictions:
        frame_id, x_tl, y_tl, delta_x, delta_y, confidence, class_id = prediction
        
        # Find the frame's position in the larger image
        frame_coords = frame_info[frame_id]
        
        frame_x_tl, frame_y_tl, _, _ = frame_coords
        
        # Adjust the prediction coordinates to the larger image
        image_x_tl = frame_x_tl + x_tl
        image_y_tl = frame_y_tl + y_tl
        
        # Append the adjusted prediction
        merged_predictions.append((class_id, image_x_tl, image_y_tl, delta_x, delta_y, confidence))
    
    return merged_predictions

def predictions_json_split_by_name(json_predictions):
    filtered_predictions = {}
    
    with open(json_predictions, "r") as f:
        predictions = json.load(f)
    
    for prediction in predictions:
        find_underscore = prediction["image_id"].rfind("_")
        image_name = prediction["image_id"][:find_underscore]
        
        if filtered_predictions.get(image_name) is None:
            filtered_predictions[image_name] = []
            
        filtered_predictions[image_name].append(prediction)
    
    return filtered_predictions


def predictions_json_to_array(json_predictions):
    predictions_arrays = {}
    
    for prediction in json_predictions:
        image_id = int(prediction["image_id"].split("_")[-1])
        bbox = np.array(prediction["bbox"])
        score = prediction["score"]
        label = prediction["category_id"]
        
        if predictions_arrays.get(image_id) is None:
            predictions_arrays[image_id] = {
                "bboxes": [], 
                "scores": [], 
                "labels": []
            }
        
        predictions_arrays[image_id]["bboxes"].append(bbox)
        predictions_arrays[image_id]["scores"].append(score)
        predictions_arrays[image_id]["labels"].append(label)
    
    return predictions_arrays

def predictions_txt_to_array(predictions_txt):
    predictions_array = []
    with open(predictions_txt, "r") as f:
        for line in f:
            predictions_array.append([float(elem) for elem in line.strip().split(" ")])

    return torch.Tensor(predictions_array)


def pack_predictions(image_id, bboxes, scores, labels):
    pred = []
    for bb, sc, lab in zip(bboxes, scores, labels):
        pred.append([image_id] + list(bb) + [sc] + [lab])
    return pred

def unpack_predictions(packed_predictions):
    bboxes = []
    scores = []
    labels = []
    
    for prediction in packed_predictions:
        bboxes.append(np.array(prediction[1:5]))
        scores.append(np.array(prediction[5]))
        labels.append(prediction[0])
    
    return bboxes, scores, labels
        

def load_image_info(image_info_path):
    image_info = []
    with open(image_info_path, "r") as f:
        for line in f:
            image_info.append([int(elem) for elem in line.split(" ")[1:]])
    
    return image_info

def nms_post(
    bboxes: torch.Tensor,
    labels: torch.Tensor,
    iou_threshold: float = 0.9
) -> torch.Tensor:
    pairwise_iou = ioas(bboxes, bboxes)
    _ = torch.diagonal(pairwise_iou, 0).zero_()

    to_discard = []

    for i, row in enumerate(pairwise_iou):
        if i not in to_discard:
            large_iou = torch.bitwise_and(row > iou_threshold, labels == labels[i])
            if large_iou.any():
                large_iou[i] = True
                bboxes_selection = bboxes[large_iou]
                bboxes_area = box_area(bboxes_selection)
                argmax_area = bboxes_area.argmax()
                argmax_area_global_index = torch.nonzero(large_iou)[argmax_area].item()

                for j in range(len(large_iou)):
                    if large_iou[j] and j != argmax_area_global_index:
                        to_discard.append(j)

    return torch.Tensor(to_discard).long()

