from punches_utils import img_utils
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", type=str, required=True)
    parser.add_argument("--labels_path", type=str, required=True)
    parser.add_argument("--destination_path", type=str, required=True)
    parser.add_argument("--frame_size", type=tuple, required=True)
    parser.add_argument("--frame_overlap", type=int, required=True)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    img_path = args.img_path
    labels_path = args.labels_path
    destination_path = args.destination_path
    frame_size = args.frame_size
    overlap = args.frame_overlap

    img_utils.preprocess_images(
        img_origin_path = img_path,
        labels_origin_path = labels_path,
        destination_path = destination_path,
        frame_size = frame_size,
        frame_overlap = overlap,
    )