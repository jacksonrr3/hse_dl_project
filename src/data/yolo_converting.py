import os
import cv2
import random
import shutil

import file_utils as futils


def convert_to_yolo(x1, y1, x2, y2, img_width, img_height):
    """
    Convert bounding box coordinates to YOLO format.

    Args:
        x1, y1, x2, y2 (float): Bounding box coordinates.
        img_width, img_height (int): Dimensions of the image.

    Returns:
        tuple: (center_x, center_y, width, height) in YOLO format.
    """
    box_width = x2 - x1
    box_height = y2 - y1
    center_x = x1 + box_width / 2
    center_y = y1 + box_height / 2
    return (
        center_x / img_width,
        center_y / img_height,
        box_width / img_width,
        box_height / img_height,
    )


def process_dataset_to_yolo(raw_image_path, annotation_path, output_path, no_detection_fraction=0.2):
    """
    Convert dataset to YOLO format.

    Args:
        raw_image_path (str): Path to raw images.
        annotation_path (str): Path to annotation files.
        output_path (str): Output path for YOLO dataset.
        no_detection_fraction (float): Fraction of no-detection images to include.
    """
    images_output = os.path.join(output_path, "images")
    labels_output = os.path.join(output_path, "labels")
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(images_output, exist_ok=True)
    os.makedirs(labels_output, exist_ok=True)

    for video_folder in os.listdir(raw_image_path):
        video_folder_path = os.path.join(raw_image_path, video_folder)
        annotation_file = os.path.join(annotation_path, f"{video_folder}.txt")

        if not os.path.isdir(video_folder_path) or not os.path.exists(annotation_file):
            print("Incorrect path")
            continue
        
        annotations = futils.read_annotations(annotation_file)
        all_frames = futils.read_files_as_map(video_folder_path)
        no_detections = {}

        for idx, frame_path in all_frames.items():
            img = cv2.imread(frame_path)
            if img is None:
                continue
            img_height, img_width = img.shape[:2]

            if idx in annotations:
                # Convert annotations
                yolo_annotations = []
                for x1, y1, x2, y2 in annotations[idx]:
                    yolo_annotations.append(convert_to_yolo(x1, y1, x2, y2, img_width, img_height))

                # Save annotations
                label_file = os.path.join(labels_output, f"{video_folder}_{idx}.txt")
                with open(label_file, "w") as label:
                    for bbox in yolo_annotations:
                        label.write(f"0 {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")

                # Save image
                image_file = os.path.join(images_output, f"{video_folder}_{idx}.jpg")
                shutil.copy(frame_path, image_file)
            else:
                # Frame with no detections
                no_detections[idx] = frame_path

    # Randomly include 20% of no-detection images
    selected_no_detections_keys = random.sample(list(no_detections.keys()), int(len(no_detections) * no_detection_fraction))
    for idx in selected_no_detections_keys:
        # Save empty annotation file
        label_file = os.path.join(labels_output, f"{video_folder}_{idx}.txt")
        open(label_file, "w").close()
        # Save image
        image_file = os.path.join(images_output, f"{video_folder}_{idx}.jpg")
        shutil.copy(no_detections[idx], image_file)