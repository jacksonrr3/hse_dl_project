import os
import numpy as np
import cv2
import seaborn as sns
from matplotlib import pyplot as plt
from collections import Counter

from config import DATA_ANALYSIS_FOLDER

def count_frames_in_dataset(dataset_folder):
    """
    Counts the number of frames (image files) in each video subfolder within the dataset.

    Args:
        dataset_folder (str): Path to the folder containing subfolders of videos. Each subfolder should contain frame images (e.g., .jpg files).

    Returns:
        dict: A dictionary where the keys are video subfolder names and the values are the number of frames (image files) in each subfolder.

    The function iterates over each subfolder in the provided dataset folder, counting the number of `.jpg` image files in each subfolder.
    It returns a dictionary with the video folder names as keys and the number of frames as values.

    Example:
        frame_counts = count_frames_in_dataset('dataset/')
        print(frame_counts)
    """
    video_frame_dict = {}

    for video_folder in os.listdir(dataset_folder):
        video_path = os.path.join(dataset_folder, video_folder)

        if not os.path.isdir(video_path):
            print(f"Skipping non-folder item: {video_folder}")
            continue

        frame_files = [f for f in os.listdir(video_path) if f.endswith('.jpg')]
        video_frame_dict[video_folder] = len(frame_files)

    print(video_frame_dict)
    return video_frame_dict


def get_detections_summary(folder_path):
    """
    Reads all .txt files in a folder and calculates the total number of detections 
    and frames with detections for each file.

    Args:
        folder_path (str): Path to the folder containing .txt files.

    Returns:
        dict: A dictionary where keys are file names (without .txt) and values are 
              tuples (num_of_detections, num_of_frames_with_detections).
    """
    summary = {}

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            file_path = os.path.join(folder_path, file_name)
            num_of_detections = 0
            num_of_frames_with_detections = 0
            
            with open(file_path, 'r') as file:
                for line in file:
                    parts = line.strip().split(',')
                    if len(parts) >= 2:
                        try:
                            detections = int(parts[1])  # Second number: number of detections on the frame
                            num_of_detections += detections
                            if detections > 0:
                                num_of_frames_with_detections += 1
                        except ValueError:
                            print(f"Invalid line in {file_name}: {line}")
                            continue
            
            # Store results using the file name without the .txt extension
            file_base_name = os.path.splitext(file_name)[0]
            summary[file_base_name] = (num_of_detections, num_of_frames_with_detections)
    
    return summary


def plot_overall_percentage_of_frames_with_detection(data):
    """
    Plots the overall percentage of frames with detections in a dataset.

    Parameters:
    data (dict): A dictionary where the keys are video identifiers, and the values are 
                 tuples containing:
                    - The number of frames with detections (int)
                    - The number of frames with detections (int)
                    - The total number of frames in the video (int)
    
    Returns:
    None
    """
    # Calculate total frames with detections and total frames across all videos
    total_frames_with_detections = sum(value[1] for value in data.values())  # num_of_frames_with_detections
    total_frames = sum(value[2] for value in data.values())  # total_frames
    
    # Calculate the overall percentage of frames with detections
    percentage_with_detection = (total_frames_with_detections / total_frames) * 100
    
    # Plotting the percentage as a bar chart
    labels = ['Frames with Detection', 'Frames without Detection']
    sizes = [percentage_with_detection, 100 - percentage_with_detection]
    colors = ['blue', 'gray']

    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    plt.title('Overall Percentage of Frames with Detection in Dataset')
    plt.axis('equal')  # Equal aspect ratio ensures the pie is drawn as a circle.

    fig_name = "Overall_percentage_of_detections.png"
    save_path = os.path.join(DATA_ANALYSIS_FOLDER, fig_name)
    plt.tight_layout()
    plt.savefig(save_path)

    plt.show()


def plot_frame_distribution(data):
    """
    Plots a stacked bar chart showing the distribution of frames with and without detections 
    for each video in the dataset.

    Parameters:
    data (dict): A dictionary where the keys are video names, and the values are tuples containing:
                 - The number of frames with detections (int)
                 - The total number of frames in the video (int)

    Returns:
    None
    """
    video_names = list(data.keys())
    frames_with_detections = [value[1] for value in data.values()]  # num_off_frames_with_detections
    frames_without_detections = [value[2] - value[1] for value in data.values()]  # total - with detections

    x = np.arange(len(video_names))
    width = 0.6

    plt.figure(figsize=(10, 6))
    plt.bar(x, frames_with_detections, width, label='Frames with Detections', color='green')
    plt.bar(x, frames_without_detections, width, bottom=frames_with_detections, label='Frames without Detections', color='gray')
    
    plt.xticks(x, video_names, rotation=45, ha='right')
    plt.xlabel('Video Name')
    plt.ylabel('Number of Frames')
    plt.title('Frame Distribution per Video')
    plt.legend()
    fig_name = "Frame_distribution.png"
    save_path = os.path.join(DATA_ANALYSIS_FOLDER, fig_name)
    plt.savefig(save_path)

    plt.tight_layout()
    plt.show()


def plot_detection_density_heatmap(data):
    """
    Plots a heatmap showing the detection density (detections per frame) for each video.

    Parameters:
    data (dict): A dictionary where the keys are video names, and the values are tuples containing:
                 - The number of detections in the video (int)
                 - The total number of frames in the video (int)

    Returns:
    None
    """
    video_names = list(data.keys())
    densities = [value[0] / value[2] for value in data.values()]  # num_of_detections / num_of_frames

    plt.figure(figsize=(10, 1))
    sns.heatmap([densities], annot=True, fmt=".2f", cmap="coolwarm", xticklabels=video_names, yticklabels=[])
    plt.xticks(rotation=45, ha='right')
    plt.title('Detection Density Heatmap (Detections per Frame)')
    fig_name = "Overall_percentage_of_detections.png"
    save_path = os.path.join(DATA_ANALYSIS_FOLDER, fig_name)
    plt.savefig(save_path)
    plt.tight_layout()
    plt.show()


def draw_resolution_histogram(resolutions):
    """
    Plots a histogram showing the distribution of video resolutions.

    Parameters:
    resolutions (dict): A dictionary where the keys are video names, and the values are tuples
                         representing the resolution (width, height) of the video.

    Returns:
    None
    """
    # Count occurrences of each resolution
    resolution_counts = Counter(resolutions.values())

    # Prepare data for plotting
    labels = [f"{res[0]}x{res[1]}" for res in resolution_counts.keys()]
    counts = resolution_counts.values()

    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.bar(labels, counts, color="skyblue", alpha=0.7)
    plt.title("Resolution Distribution")
    plt.xlabel("Resolution")
    plt.ylabel("Number of Videos")
    plt.xticks(rotation=45, ha="right")
    fig_name = "Resolutions.png"
    save_path = os.path.join(DATA_ANALYSIS_FOLDER, fig_name)
    plt.savefig(save_path)
    plt.tight_layout()
    plt.show()


def visualize_drones_areas(area_data):
    """
    Visualizes the distribution of areas for each file and prints a summary of area statistics across all files.

    Parameters:
    area_data (dict): A dictionary where the keys are filenames and the values are lists of area values corresponding to drones.

    Returns:
    None
    """
    # Prepare for summary
    all_areas = []

    # Plot area distributions for each file
    plt.figure(figsize=(10, 6))
    for filename, areas in area_data.items():
        all_areas.extend(areas)
        plt.hist(areas, bins=10, alpha=0.6, label=filename)

    # Summary statistics
    all_areas = np.array(all_areas)
    overall_summary = {
        "Minimum": np.min(all_areas),
        "Maximum": np.max(all_areas),
        "Mean": np.mean(all_areas),
        "Median": np.median(all_areas),
    }

    # Print overall summary
    print("Summary Across All Files:")
    for key, value in overall_summary.items():
        print(f"{key}: {value:.2f}")

    # Add details to the plot
    plt.title("Area Distribution Per File")
    plt.xlabel("Area")
    plt.ylabel("Frequency")
    plt.legend()
    fig_name = "Drones_areas.png"
    save_path = os.path.join(DATA_ANALYSIS_FOLDER, fig_name)
    plt.savefig(save_path)
    plt.tight_layout()
    plt.show()


def analyze_image_sizes(base_path):
    """
    Traverse all folders in the given path, read image sizes, and plot size distribution.

    Args:
        base_path (str): Path to the directory containing image files.
    """
    image_sizes = []

    # Traverse directories and gather image sizes
    for root, _, files in os.walk(base_path):
        for file in files:
            file_path = os.path.join(root, file)
            # Check if the file is an image
            if file.lower().endswith(('.jpg', '.jpeg')):
                image = cv2.imread(file_path)
                if image is not None:
                    height, width = image.shape[:2]
                    image_sizes.append((width, height))
    
    # Convert sizes to numpy array for easier processing
    image_sizes = np.array(image_sizes)

    # Plot distribution of widths and heights
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.hist(image_sizes[:, 0], bins=20, color='blue', alpha=0.7, edgecolor='black')
    plt.title('Width Distribution')
    plt.xlabel('Width (pixels)')
    plt.ylabel('Frequency')

    plt.subplot(1, 2, 2)
    plt.hist(image_sizes[:, 1], bins=20, color='green', alpha=0.7, edgecolor='black')
    plt.title('Height Distribution')
    plt.xlabel('Height (pixels)')
    plt.ylabel('Frequency')
    fig_name = "Image_size_distribution.png"
    save_path = os.path.join(DATA_ANALYSIS_FOLDER, fig_name)
    plt.savefig(save_path)
    plt.tight_layout()
    plt.show()

    # Print summary statistics
    widths = image_sizes[:, 0]
    heights = image_sizes[:, 1]
    print("Summary Statistics:")
    print(f"Width - Min: {np.min(widths)}, Max: {np.max(widths)}")
    print(f"Height - Min: {np.min(heights)}, Max: {np.max(heights)}")

