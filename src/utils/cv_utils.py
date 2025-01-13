import os
import cv2
import random
import numpy as np
from matplotlib import pyplot as plt

from . import file_utils as futils
from config import ANNOTATIONS_FL_DRONES_DS_FOLDER, DATA_ANALYSIS_FOLDER


def calculate_bbox_area(x1, y1, x2, y2):
    """Calculate the area of a bounding box given its coordinates."""
    return np.abs(x2 - x1) * np.abs(y2 - y1)


def read_annotations_and_calculate_areas(directory):
    """Read annotation files in YOLO format and calculate bounding box areas."""
    areas_dict = dict()
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            video_annotations = futils.read_annotations(filepath)
            for bboxes in video_annotations.values():
                for bbox in bboxes:
                    x1, y1, x2, y2 = bbox
                    area = calculate_bbox_area(x1, y1, x2, y2)
                    if filename in areas_dict:
                        areas_dict[filename].append(area)
                    else:
                        areas_dict[filename] = [area]

    return areas_dict


def normalize_coordinates(x_center, bbox_width):
    """
    Normalize x coordinate in range [0; 1].

    Parameters:
    - x_center: the x-coordinate of the center of the bounding box (normalized [0, 1]).
    - bbox_width: the width of the bounding box (normalized [0, 1]).

    Returns:
    - Adjusted (x_center, bbox_width) ensuring the bounding box is within [0, 1].
    """
    bbox_left = x_center - bbox_width / 2
    bbox_right = x_center + bbox_width / 2

    # If bbox_left < 0, move x_center to make bbox_left = 0
    if bbox_left < 0:
        bbox_width = bbox_width + bbox_left
        x_center = bbox_width / 2

    # If bbox_right > 1, move x_center to make bbox_right = 1
    if bbox_right > 1:
        bbox_width = bbox_width + (1 - bbox_right)
        x_center = 1 - bbox_width / 2

    return x_center, bbox_width


def split_videos_to_frames(video_folder, output_folder):
    """
    Splits all video files in the specified folder into frames and saves them as individual images.

    Args:
        video_folder (str): Path to the folder containing video files (e.g., .mp4, .avi, .mkv, .mov, .wmv).
        output_folder (str): Path to the folder where extracted frames will be saved.

    The function processes all video files in the `video_folder`, extracts individual frames, and saves them as `.jpg` 
    images in the `output_folder`. Each video will have its own subfolder in the output directory, named after the video.
    
    The function also prints progress for each video and the total number of frames processed.
    
    Example:
        split_videos_to_frames('videos/', 'frames_output/')
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    total_frames = 0

    for video_file in os.listdir(video_folder):
        video_path = os.path.join(video_folder, video_file)

        if not video_file.lower().endswith(('.mp4', '.avi', '.mkv', '.mov', '.wmv')):
            print(f"Skipping non-video file: {video_file}")
            continue

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video file: {video_file}")
            continue

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Processing {video_file} ({frame_count} frames)")

        video_name = os.path.splitext(video_file)[0]
        video_output_folder = os.path.join(output_folder, video_name)

        if not os.path.exists(video_output_folder):
            os.makedirs(video_output_folder)

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_filename = os.path.join(video_output_folder, f"frame_{frame_idx:06d}.jpg")
            cv2.imwrite(frame_filename, frame)
            frame_idx += 1

        cap.release()

        print(f"{video_file}: {frame_idx} frames saved.")
        total_frames += frame_idx

    print(f"Total frames processed: {total_frames}")


def draw_bboxes_on_the_video(video_number, video_path, output_video_path):
    """
    Processes a video, drawing bounding boxes around annotated objects for each frame, 
    and saves the output to a new video file.

    Parameters:
    video_number (int): The identifier for the video to be processed (used for naming the video file).
    video_path (str): The path where the input video is stored.
    output_video_path (str): The directory path where the output video will be saved.

    Steps:
    1. Reads the video and its annotations (in YOLO format).
    2. Extracts video properties such as frames per second (fps), width, and height.
    3. Iterates through each frame of the video.
    4. For each frame, checks if annotations exist and draws the corresponding bounding boxes.
    5. Writes the processed frame to an output video file.
    6. Releases resources after processing is complete.

    Returns:
    None

    Example:
    draw_bboxes_on_the_video(1, '/path/to/videos', '/path/to/output')
    """
    # Open the video
    filename = f"Video_{video_number}"
    # Load annotations
    annotations = futils.read_annotations(ANNOTATIONS_FL_DRONES_DS_FOLDER + '/' + filename + '.txt')

    cap = cv2.VideoCapture(video_path + '/' + filename + '.avi')
    if not cap.isOpened():
        print("Error: Cannot open video.")
        exit()

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    full_output_path = output_video_path + '/' + filename + '_inference.mp4'
    out = cv2.VideoWriter(full_output_path, fourcc, fps, (width, height))

    # Process the video
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Draw bounding boxes if annotations exist for the current frame
        if frame_idx in annotations:
            for bbox in annotations[frame_idx]:
                x1, y1, x2, y2 = bbox
                color = (0, 255, 0)  # Green color for bounding boxes
                thickness = 2
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # Write the frame to the output video
        out.write(frame)

        frame_idx += 1
        if frame_idx >= frame_count:
            break

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("Output video saved to:", full_output_path)


def get_frames_and_show(video_number, video_path, frame_numbers):
    """
    Extracts specific frames from a video and displays them in a grid using Matplotlib.
    
    Args:
        video_path (str): Path to the video file.
        frame_numbers (list of int): List of frame numbers to extract (0-based index).
    
    Returns:
        None
    """

    full_video_path = f"{video_path}/Video_{video_number}_inference.mp4"
    # Open the video file
    cap = cv2.VideoCapture(full_video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    frames = []
    for frame_number in frame_numbers:
        # Set the frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        
        # Read the frame
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Could not read frame {frame_number}.")
            continue
        
        # Convert BGR to RGB and store the frame
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # Release the video capture object
    cap.release()
    
    # Plot the frames in a grid
    num_frames = len(frames)
    cols = 2  # Number of columns
    rows = (num_frames + cols - 1) // cols  # Calculate rows based on frames
    fig, axes = plt.subplots(rows, cols, figsize=(10, 5))
    axes = axes.flatten()  # Flatten the axes array for easier indexing
    
    for i in range(len(axes)):
        if i < num_frames:
            axes[i].imshow(frames[i])
            axes[i].set_title(f"Frame {frame_numbers[i]}")
            axes[i].axis('off')  # Hide axes for a cleaner look
        else:
            axes[i].axis('off')  # Hide any unused axes
    
    fig_name = f"Video_{video_number}_frames.png"
    save_path = os.path.join(DATA_ANALYSIS_FOLDER, fig_name)
    plt.savefig(save_path)
    
    plt.tight_layout()
    plt.show()


def get_video_resolutions(directory):
    """
    Read all .avi video files in a directory and return a map of filenames to their resolutions.
    """
    resolutions = {}  # Dictionary to store video resolutions
    for filename in os.listdir(directory):
        if filename.endswith(".avi"):
            filepath = os.path.join(directory, filename)
            cap = cv2.VideoCapture(filepath)
            if cap.isOpened():
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                resolutions[filename] = (width, height)
            cap.release()
    return resolutions


def draw_yolo_annotations(image, annotations):
    """
    Draws YOLO annotations on the image.
    
    Args:
        image_path (str): Path to the image file.
        annotations (list): List of YOLO annotations, where each annotation is in the format:
                            [class_id, x_center, y_center, width, height]
    
    Returns:
        None
    """
    image_height, image_width, _ = image.shape
    
    for annotation in annotations:
        _, x_center, y_center, width, height = annotation
        print(x_center, y_center, width, height)
        # Convert from normalized coordinates to pixel coordinates
        x1 = int((x_center - width / 2) * image_width)
        y1 = int((y_center - height / 2) * image_height)
        x2 = int((x_center + width / 2) * image_width)
        y2 = int((y_center + height / 2) * image_height)
        print(x1,y1,x2,y2)
        # Draw bounding box on the image
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Display the image
    cv2.imshow("Image with YOLO Annotations", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def obj_out_of_image(x_center, bbox_width):
    """
    Check if the bounding box is out of the image by width.

    Parameters:
    - x_center: the x-coordinate of the center of the bounding box (normalized [0, 1]).
    - bbox_width: the width of the bounding box (normalized [0, 1]).

    Returns:
    - 'left' if the bounding box is out on the left side.
    - 'right' if the bounding box is out on the right side.
    - None if the bounding box is fully inside the image width.
    """
    bbox_left = x_center - bbox_width / 2
    bbox_right = x_center + bbox_width / 2
    
    # Check if the bounding box is outside the image bounds
    if bbox_left < 0:
        return 'left'
    elif bbox_right > 1:
        return 'right'
    return None


def crop_images_and_update_annotations(image_folder, annotation_folder, target_width=480):
    """
    Crop images by width with random crop_x coordinate and update annotations coordinates.

    Parameters:
    - image_folder: path to images folder.
    - annotation_folder: path to labels folder.
    - target_width: target size to crop images.
    """
    # Rename all files with its index
    img_idx = 0
    detections_num = 0

    # Iterate over each image in the dataset
    for image_name in os.listdir(image_folder):
        if image_name.endswith('.jpg'):
            # Read the image and annotation
            image_path = os.path.join(image_folder, image_name)
            annotation_path = os.path.join(annotation_folder, image_name.replace('.jpg', '.txt'))
            
            image = cv2.imread(image_path)
            _, w = image.shape[:2]
            
            # Read annotations for the current image
            with open(annotation_path, 'r') as f:
                annotations = f.readlines()

            # Randomly crop the image horizontally
            crop_x = random.randint(0, w - target_width)
            cropped_image = image[:, crop_x:crop_x + target_width]
            _, new_w = cropped_image.shape[:2]
            
            # Adjust annotations based on crop
            new_annotations = []
            for ann in annotations:
                class_id, x_center, y_center, bbox_width, bbox_height = map(float, ann.split())

                # Convert bbox coordinates
                new_x_center = (x_center * w - crop_x) / new_w
                new_bbox_width = bbox_width * w / new_w

                # Check coordinates inside cropped image
                if obj_out_of_image(new_x_center, new_bbox_width) is None:
                    new_annotations.append([class_id, new_x_center, y_center, new_bbox_width, bbox_height])
                    detections_num += 1
                else:
                    norm_x_center, norm_bbox_width = normalize_coordinates(new_x_center, new_bbox_width)
                    if norm_bbox_width > 0 and obj_out_of_image(norm_x_center, norm_bbox_width) is None:
                        new_annotations.append([class_id, norm_x_center, y_center, norm_bbox_width, bbox_height])
                        detections_num += 1
            
            # Save the cropped image
            new_image_name = f"img_{img_idx}.jpg"
            new_image_path = os.path.join(image_folder, new_image_name)
            cv2.imwrite(new_image_path, cropped_image)

            # Save the new annotations for the cropped image
            new_annotation_path = os.path.join(annotation_folder, new_image_name.replace('.jpg', '.txt'))
            with open(new_annotation_path, 'w') as f:
                for bbox in new_annotations:
                        f.write(f"{bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]} {bbox[4]}\n")

            # Delete the original image and annotation if you no longer need them
            os.remove(image_path)
            os.remove(annotation_path)
            
            # Increase image idx
            img_idx += 1
    
    print(f"Number of images in dataset: {img_idx}")
    print(f"Number of detections in dataset: {detections_num}")