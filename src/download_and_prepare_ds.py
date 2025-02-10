import argparse
import os
import sys

import utils.cv_utils as cv_utils
import utils.file_utils as futils
import data.data_loader as dload
import data.yolo_converting as yolo_conv
from config import DRONE_DS_VIDEOS_FOLDER, DATASET_RAW_FOLDER, ANNOTATIONS_FL_DRONES_DS_FOLDER, \
                    YOLO_ANNOTATIONS_PATH, YOLO_IMAGES_PATH


def set_current_dir(project_path):
<<<<<<< HEAD
    """Set project_path dir as current directory"""
    os.chdir(project_path)
=======
    """Set src dir as current directory"""
    os.chdir(os.path.join(project_path, "src"))
>>>>>>> 6ab15b5 (Update some misfunctions)

def main():
    """Main function to download and prepare datasets."""
    
    # Step 1: Parse command-line arguments
    parser = argparse.ArgumentParser(description='Download and prepare dataset for the project.')
    parser.add_argument('project_path', type=str, help='Path to the project directory')
    args = parser.parse_args()

    # Check if project_path is provided
    if not args.project_path:
        print("Error: project_path argument is required.")
        sys.exit(1)  # Stop the program with an error code
    
    # Step 2: Set up directories
    set_current_dir(args.project_path)

    # Step 3: Download Drones dataset
    print("Downloading dataset...")
    dload.download_drones_dataset()
    print("Downloading is complete")

    # Step 4: Split videos to frames and save as images
    print("Spliting videos to frames...")
    cv_utils.split_videos_to_frames(DRONE_DS_VIDEOS_FOLDER, DATASET_RAW_FOLDER)
    print(f"Frames saved at {DATASET_RAW_FOLDER}")

    # Step 5: Download annotations for the dataset
    print("Downloading drones annotations...")
    dload.download_fl_drones_annotations()
    print("Downloading is complete")
    print("Renaming annotations files")
    futils.rename_files_in_folder(ANNOTATIONS_FL_DRONES_DS_FOLDER)

    # Step 6: Transform annotations to YOLO format
    print("Creating dataset folders")
    dload.create_yolo_dataset_folders()
    print("Transform annotations to YOLO format")
    yolo_conv.process_dataset_to_yolo(DATASET_RAW_FOLDER, ANNOTATIONS_FL_DRONES_DS_FOLDER, os.path.dirname(YOLO_ANNOTATIONS_PATH))
    print("Moving images and annotations")
    print(f"Number of images in dataset: {futils.get_number_of_files(YOLO_IMAGES_PATH)}")
    print(f"Number of labels in dataset: {futils.get_number_of_files(YOLO_ANNOTATIONS_PATH)}")

    # Step 7: Crop images to 480x480 size and update bbox coordinates in annotations files
    print("Cropping images...")
    cv_utils.crop_images_and_update_annotations(YOLO_IMAGES_PATH, YOLO_ANNOTATIONS_PATH, 480)
    has_detections, no_detections = futils.categorize_files_by_detections(YOLO_ANNOTATIONS_PATH)
    print(f"Percentage of files with detections: {(len(has_detections) / (len(has_detections) + len(no_detections)) * 100):.2f}%")

    # Step 8: Split dataset to train/test/val with 0.7/0.15/0.15
    print("Spliting dataset to train/test/val")
    dload.split_dataset(YOLO_IMAGES_PATH, YOLO_ANNOTATIONS_PATH, 0.7, .15, .15)

    # Step 9: Create data.yaml file
    print("Creating YAML file")
    dload.create_yaml_file(os.path.dirname(YOLO_IMAGES_PATH))

    print("Dataset preparation is complete.")

if __name__ == "__main__":
    main()
