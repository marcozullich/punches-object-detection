import os
import cv2
import numpy as np

from PIL import Image
from psd_tools import PSDImage
from typing import List, Tuple

from punches_utils.labels_utils import label_to_absolute, read_labels, save_labels

CV2_COLOR_MAP = [(0,0,255), (255,0,0)]

def read_psb(file: str) -> Image.Image:
    """
    Read a .psb file using psd_tools.

    Args:
        file (str): Path to the .psb file.

    Returns:
        PIL.Image: The image read from the .psb file.
    """
    return PSDImage.open(file).composite().convert("RGB")



def split_image_to_frames(image: Image, frame_size: Tuple[int, int], overlap: int, labels: List[Tuple]) -> List[Tuple[Image.Image, List[Tuple[float, float, float, float, int]]]]:
    """
    Split an image into smaller frames of a given size with specified overlap and split the YOLO labels accordingly.
    
    Args:
        image (Image): The input image to be split.
        frame_size (Tuple[int, int]): The size of each frame as (width, height).
        overlap (int): The number of overlapping pixels between frames.
        labels (List[Tuple[float, float, float, float, int]]): A list of YOLO labels for the entire image.
    
    
    Returns:
        Tuple[Tuple[Image, List[Tuple[float, float, float, float, int]]]]: A list of tuples containing the image frame and its corresponding YOLO labels.
    """
    frames = []
    width, height = image.size
    frame_width, frame_height = frame_size

    frames_data = []
    
    for y in range(0, height - frame_height + 1, frame_height - overlap):
        for x in range(0, width - frame_width + 1, frame_width - overlap):
            box = (x, y, x + frame_width, y + frame_height)
            frames_data.append(box)
            frame = image.crop(box)
            
            
            # Split the labels according to the frame
            frame_labels = []
            for label in labels:
                class_id, x_center, y_center, w, h = label
                x_center, y_center, w, h = label_to_absolute(x_center, y_center, w, h, height, width)
                 
                # Calculate the bounding box edges
                x_min = x_center - (w / 2)
                y_min = y_center - (h / 2)
                x_max = x_center + (w / 2)
                y_max = y_center + (h / 2)
                
                # Check if the bounding box is completely outside the frame
                if (x_max < x or x_min > x + frame_width or
                    y_max < y or y_min > y + frame_height):
                    continue  # Skip this label as it's completely outside the frame
                
                # Clip the bounding box to the frame boundaries
                if x_min < x:
                    x_min = x
                if y_min < y:
                    y_min = y
                if x_max > x + frame_width:
                    x_max = x + frame_width
                if y_max > y + frame_height:
                    y_max = y + frame_height
                
                # Calculate new center and dimensions
                new_x_center = (x_min + x_max) / 2 - x
                new_y_center = (y_min + y_max) / 2 - y
                new_w = (x_max - x_min) / frame_width
                new_h = (y_max - y_min) / frame_height
                
                # Normalize center coordinates
                new_x_center /= frame_width
                new_y_center /= frame_height
                
                frame_labels.append((class_id, new_x_center, new_y_center, new_w, new_h))
            frames.append((frame, frame_labels))
    return frames, frames_data

def preprocess_images(img_origin_path: str, labels_origin_path: str, destination_path: str, frame_size: Tuple[int, int], frame_overlap: int, files_ext: Tuple[str] = (".psb",), save_ext: str = ".png") -> None:
    destination_path_images = os.path.join(destination_path, "images")
    destination_path_labels = os.path.join(destination_path, "labels")
    _ = [os.makedirs(folder, exist_ok=True) for folder in (destination_path_images, destination_path_labels)]

    list_of_images = os.listdir(img_origin_path)

    for img_path in list_of_images:
        file_path = os.path.join(img_origin_path, img_path)
        file_name = os.path.splitext(os.path.basename(img_path))[0]
        labels_path = os.path.join(labels_origin_path, f"{file_name}.txt")
        if os.path.splitext(img_path)[1] in files_ext:
            image = read_psb(file_path)
            labels = read_labels(labels_path)
            frames, frames_data = split_image_to_frames(image, frame_size, frame_overlap, labels)
            print(frames_data)
            for i, (frame, labels) in enumerate(frames):
                frame.save(os.path.join(destination_path_images, f"{file_name}_{i}{save_ext}"))
                save_labels(labels, os.path.join(destination_path_labels, f"{file_name}_{i}.txt"), add_row_id=False)
            save_labels(frames_data, os.path.join(destination_path, f"{file_name}_frames.txt"), add_row_id=True)
            print(f"File {file_name}: saved {len(frames)} to {destination_path}")


def draw_yolo_labels(image_pil, labels, save_file):
    # Load the image
    image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    height, width, _ = image.shape
    
    for label in labels:
        class_id, x_c, y_c, delta_x, delta_y = label
        
        # Convert YOLO format to pixel coordinates
        x_center = int(x_c * width)
        y_center = int(y_c * height)
        width_box = int(delta_x * width)
        height_box = int(delta_y * height)
        
        # Calculate top-left corner of the bounding box
        x1 = int(x_center - width_box / 2)
        y1 = int(y_center - height_box / 2)
        x2 = int(x_center + width_box / 2)
        y2 = int(y_center + height_box / 2)
        
        # Draw the rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # Put class label
        cv2.putText(image, f'Class: {class_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    cv2.imwrite(save_file, image)

def draw_predictions(image, predictions, color_array=None, names=None, debug=False, x2y2_is_delta=False):   
    for i, pred in enumerate(predictions):
        _, x1, y1, x2, y2, score, label = pred

        if names is not None:
            label = names[label]

        if x2y2_is_delta:
            x2 += x1
            y2 += y1

        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

        if color_array is not None:
            color = CV2_COLOR_MAP[color_array[i]]
        else:
            color = (255, 0, 0)
        
        # Draw the rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Put class label
        cv2.putText(image, f'Cl: {label} ({score})', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    if debug:
        image = image[:1088*4, :1088*4]
    return image


