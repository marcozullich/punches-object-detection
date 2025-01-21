# Code for the paper "Large-image Object Detection for Fine-grained Recognition of Punches Patterns in Medieval Panel Painting", by Josh Bruegger et al. Published at EvoMUSART, March 2025

## Environment setup

Install YOLOv10 as in [this repository](https://github.com/THU-MIG/yolov10) and set up an environment as there indicated.

After installation, you should be able to run the `yolo` command from your shell, as shown in the repository.

Additionally, install psd_tools via pip

```
pip install psd_tools
```

All the procedure indicated in this repository will run in the yolo environment.

## Running our code

### Dataset

We are currently trying to work out a solution for hosting the dataset. Please email us for accessing the data.

Set up the datasets folder in using YOLO format:

```
split // e.g., train, val, test
  |
   - images
   - labels
```

Modify the `dataset.yaml` file by indicating the path of the `images` folder for each of the three splits.

### Training

Operate training using the following command

`yolo train model=yolov10n.yaml pretrained=true data=dataset.yaml epochs=100 imgsz=1088 seed=<insert seed>`

You can further personalize the destination of the weights by toggling the `project` and `name` arguments.

The seeding can be deactivated by passing the `deterministic=false` arg.

### Testing

The model is tested on the full image(s) decomposed into windows.
The evaluation on these images is operated with the `val` option:

`yolo detect val model=<path/to/weights.pt> data=dataset.yaml imgsz=1088 save_json=true iou=0.5 split=test`

Again, personalize the destination of the files toggling the `project` and `name` arguments.

In the destination folder, a `predictions.json` will be created containing all of the predictions for all of the windows.

#### Merging the per-window predictions

The predictions can be recomposed using the `merge_predictions.py` script in this repo.
This will merge the prediction using the custom NMS explained in the paper.

Run it in the following way:

```bash
python merge_predictions.py \
    --predictions_path <path/to/predictions.json> \
    --image_info_paths <path/to/test/dataset/07_Traino_S_Domenico_2_frames.txt> \
    --conf_threshold <confidence_threshold> \
    --iou_threshold_nms <iou threshold custom NMS>
```

The script will create a subfolder of `--predictions_path` called `conf_{conf_threshold}_nms_{iou_threshold_nms}` (replace the content of `{}` with the appropriate value of the arguments).
Within this folder, a file will be created containing the merged predictions in the following format:
```
x_top_left y_top_left x_bottom_right y_bottom_right class_id
```

#### Calculating the metrics

The metrics can finally be calculating by running the `eval_metrics.py` script in the following way:

```bash
python eval_metrics.py \
    --predictions_file <path/to/merged_predictions.py> \
    --ground_truth_file <path/to/source/dataset/labels/07_Traino_S_Domenico_2.txt> \
    --save_file <path/to/metrics/save/file.txt> \
    --yaml_file dataset.yaml \
    --orig_img_width 27274 --orig_img_height 36451 \
    --iou_threshold 0.5
```

### Testing on custom dataset

For testing on a custom dataset, the images need first to be split in windows.
This can be achieved by running the `split_img.py` script for each image.
The file is designed to be ran on high-quality .psd Photoshop images:

```bash
python split_img.py \
    --img_path <path/to/image.psd> \
    --labels_path <path/to/labels_current_image.txt> \
    --destination_path <folder/for/cropped/windows> \
    --frame_size 1088 1088 # modify according to dataset \
    --frame_overlap 324 # modify according to dataset
```

The script will create two subfolders of `--destination_path`, one named "images", the other named "labels", as per yolov10 requirements; the former contains the cropped images, the latter contains the labels.
Both images and labels are named as `<image_name>_frame_id.png/txt`


In addition, an extra file will be created in the `--destination_path` folder, indicating the position of each window in the original image.
The file has name `<image_name>_frames.txt` and is formatted as follows:
```
frame_id x_top_left y_top_left x_bottom_right y_bottom_right
```

This is the file that has to be indicated in the `--image_info_paths` arg in `merge_predictions.py`.

# Citing this work

This work is to be presented at EvoMUSART 2025.

Cite using the following BibTeX:

```
@inproceedings {bruegger2025largeimage,
    author    = "Bruegger, Josh and Catana, Diana and Macovaz, Vanja and Sabatelli, Matthia, and Valdenegro-Toro, Matias and Zullich, Marco",
    title     = "Large-image Object Detection for Fine-grained Recognition of Punches Patterns in Medieval Panel Painting",
    booktitle = "EvoMUSART",
    year      = "2025"
}
```
