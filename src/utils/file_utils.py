import os
import re

import cv2
import numpy as np

def rename_files_in_folder(folder_path):
    """
    Renames all files in the folder from the format 'Video_001.txt' to 'Video_1.txt'.

    Args:
        folder_path (str): Path to the folder containing files.

    Returns:
        None
    """
    for file_name in os.listdir(folder_path):
        if file_name.startswith("Video_") and file_name.endswith(".txt"):
            # Extract the numeric part, strip leading zeros, and reformat the name
            prefix, num_with_zeros = file_name.split('_')
            num = int(num_with_zeros.split('.')[0])  # Convert to int to strip leading zeros
            new_name = f"{prefix}_{num}.txt"
            
            # Rename the file
            old_path = os.path.join(folder_path, file_name)
            new_path = os.path.join(folder_path, new_name)
            os.rename(old_path, new_path)
            print(f"Renamed: {file_name} -> {new_name}")


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


def categorize_files_by_detections(dataset_labels_dir):
    """
    Split images with and without detections and store their names to two lists.

    Parameters:
    - dataset_labels_dir: path to labels folder.

    Returns:
    - has_detections: list of filenames with detections.
    - no_detections: list of filenames with no detections.
    """
    has_detections = []
    no_detections = []

    # Iterate through all label files in the dataset directory
    for filename in os.listdir(dataset_labels_dir):
        # Process only .txt files in the 'labels' subdirectory
        if filename.endswith('.txt'):
            label_file_path = os.path.join(dataset_labels_dir, filename)

            # Check if the label file has any detections (non-empty file)
            with open(label_file_path, 'r') as f:
                if f.read().strip():  # If file is not empty, it has detections
                    has_detections.append(os.path.splitext(filename)[0])
                else:  # No detections in the file
                    no_detections.append(os.path.splitext(filename)[0])

    return has_detections, no_detections


def get_number_of_files(folder_path, extension):
    """
    Counts the number of files in a folder with the given extension.

    Args:
    - folder_path (str): Path to the folder to search.
    - extension (str): The file extension to search for (e.g., '.jpg', '.txt').

    Returns:
    - int: The number of files with the specified extension in the folder.
    """
    # List all files in the folder and count those with the given extension
    file_count = sum(1 for filename in os.listdir(folder_path)
    if filename.lower().endswith(extension.lower()))

    return file_count

def load_image(path):
 
    image = cv2.imread(path)
     
    # Convert image in BGR format to RGB.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
     
    # Add a batch dimension which is required by the model.
    image = np.expand_dims(image, axis=0)
     
    return image

def create_dir(path):
    os.makedirs(path, exist_ok=True)

