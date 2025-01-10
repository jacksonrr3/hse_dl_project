import os
import random
import shutil
import re

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


def read_annotations(file_path):
    """
    Read annotations from the text file
    """
    annotations = {}
    with open(file_path, "r") as f:
        for line in f:
            values = line.strip().split(",")
            frame_num = int(values[0])
            num_of_detections = int(values[1])
            bboxes = []
            for i in range(num_of_detections):
                x1 = int(values[2 + i * 4])
                y1 = int(values[3 + i * 4])
                x2 = int(values[4 + i * 4])
                y2 = int(values[5 + i * 4])
                bboxes.append((x1, y1, x2, y2))
            annotations[frame_num] = bboxes
    return annotations


def read_files_as_map(directory, prefix='frame_', extension='.jpg'):
    """
    Reads files in a directory and returns a dictionary mapping numeric indices to file paths.

    Args:
        directory (str): Path to the folder containing the files.
        prefix (str): The prefix of the filenames to filter. Default is 'frame_'.
        extension (str): The file extension to filter. Default is '.jpg'.

    Returns:
        dict: A dictionary where keys are numeric indices and values are file paths.
    """
    # Define a regex pattern to extract the number from filenames
    pattern = re.compile(rf"{prefix}(\d+){extension}")

    # Initialize the result map
    file_map = {}

    # Iterate through files in the directory
    for file in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, file)):
            match = pattern.match(file)
            if match:
                idx = int(match.group(1))  # Extract the numeric index
                file_map[idx] = os.path.join(directory, file)

    # Return the dictionary sorted by numeric keys
    return dict(sorted(file_map.items()))
