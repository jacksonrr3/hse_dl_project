import subprocess
import os
import shutil
import random
import yaml

import file_utils as futils
from config import DRONE_DS_VIDEOS_FOLDER, YOLO_ANNOTATIONS_PATH, YOLO_IMAGES_PATH

def download_drones_dataset():
    # Check if dataset folder exist
    if os.path.exists(DRONE_DS_VIDEOS_FOLDER):
        print("Dataset folder already exists")
        return

    # Download the file using wget
    subprocess.run(["wget", "https://drive.switch.ch/index.php/s/3b3bdbd6f8fb61e05d8b0560667ea992/download"])

    # Unzip the downloaded file
    subprocess.run(["unzip", "download"])

    # Move dataset to the project
    shutil.move('../cvpr15', '.')

def download_fl_drones_annotations():
    # Clone the repository
    subprocess.run(["git", "clone", "https://github.com/mwaseema/Drone-Detection.git", "../../Drone-Detection"])
    
    # Copy annotaion files to the dataset folder
    source_folder = '../../Drone-Detection/annotations/FL-Drones-Dataset/'
    destination_folder = '../dataset/annotations/'

    # Copy the folder and its contents to the new location
    shutil.copytree(source_folder, destination_folder)


def create_yolo_dataset_folders():
    os.makedirs(os.path.dirname(YOLO_ANNOTATIONS_PATH), exist_ok=True)
    os.makedirs(YOLO_IMAGES_PATH, exist_ok=True)
    os.makedirs(YOLO_ANNOTATIONS_PATH, exist_ok=True)


def split_dataset(image_folder, annotation_folder, train_fraction, test_fraction, val_fraction):
    # Ensure fractions sum to 1
    assert train_fraction + test_fraction + val_fraction == 1.0, "Fractions must sum to 1."

    # Categorize files with and without detections
    has_detections, no_detections = futils.categorize_files_by_detections(annotation_folder)

    # Full image paths
    has_detections_images = [os.path.join(image_folder, img_name + '.jpg') for img_name in has_detections]
    no_detections_images = [os.path.join(image_folder, img_name + '.jpg') for img_name in no_detections]
    
    # Shuffle the images to distribute them evenly across the splits
    random.shuffle(has_detections_images)
    random.shuffle(no_detections_images)

    # Calculate the split indices for images with detections
    total_has_detection_images = len(has_detections_images)
    train_has_det_size = int(train_fraction * total_has_detection_images)
    test_has_det_size = int(test_fraction * total_has_detection_images)
    val_has_det_size = total_has_detection_images - train_has_det_size - test_has_det_size

    # Calculate the split indices for images without detections
    total_no_detection_images = len(no_detections_images)
    train_no_det_size = int(train_fraction * total_no_detection_images)
    test_no_det_size = int(test_fraction * total_no_detection_images)
    val_no_det_size = total_no_detection_images - train_no_det_size - test_no_det_size

    # Split images with detections
    train_has_det_images = has_detections_images[:train_has_det_size]
    test_has_det_images = has_detections_images[train_has_det_size:train_has_det_size + test_has_det_size]
    val_has_det_images = has_detections_images[train_has_det_size + test_has_det_size:]

    # Split images without detections
    train_no_det_images = no_detections_images[:train_no_det_size]
    test_no_det_images = no_detections_images[train_no_det_size:train_no_det_size + test_no_det_size]
    val_no_det_images = no_detections_images[train_no_det_size + test_no_det_size:]

    # Combine the images with detections and without detections for each set
    train_images = train_has_det_images + train_no_det_images
    test_images = test_has_det_images + test_no_det_images
    val_images = val_has_det_images + val_no_det_images

    # Create necessary directories for train, test, val
    for folder in ['train', 'test', 'val']:
        os.makedirs(os.path.join(image_folder, folder), exist_ok=True)
        os.makedirs(os.path.join(annotation_folder, folder), exist_ok=True)

    # Move images and annotations to their respective folders
    def move_files(image_list, set_name):
        for img in image_list:
            # Move image
            shutil.move(img, os.path.join(image_folder, set_name, os.path.basename(img)))
            # Move corresponding annotation file
            annotation_file = os.path.splitext(os.path.basename(img))[0] + '.txt'
            shutil.move(os.path.join(annotation_folder, annotation_file), os.path.join(annotation_folder, set_name, annotation_file))

    # Move the files
    move_files(train_images, 'train')
    move_files(test_images, 'test')
    move_files(val_images, 'val')

    # Print the number of files in each set
    print(f"Number of training images: {len(train_images)}")
    print(f"Number of testing images: {len(test_images)}")
    print(f"Number of validation images: {len(val_images)}")


def create_yaml_file(dataset_root):
    data_yaml = {
        'path': 'yolo',
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'names': {0: 'drone'}
    }

    with open(os.path.join(dataset_root, 'data.yaml'), 'w') as yaml_file:
        yaml.dump(data_yaml, yaml_file)