import os
import shutil
import cv2
import json
import numpy as np
import pandas as pd
from collections import Counter
from PIL import Image
import matplotlib.pyplot as plt

import pandas as pd

def expand_csv_views(source_file_path, target_file_path):
    '''
    Expand action_label csv to all views because in we only annotated one view
    '''
    data = pd.read_csv(source_file_path)
    all_views = range(1, 9)
    new_rows = []

    for index, row in data.iterrows():
        current_view = row['view']
        for view in all_views:
            if view != current_view:
                new_row = row.copy()
                new_row['view'] = view
                new_rows.append(new_row)
    new_rows_df = pd.DataFrame(new_rows)

    # Combine the original data with the new rows DataFrame
    expanded_data = pd.concat([data, new_rows_df], ignore_index=True)
    expanded_data.to_csv(target_file_path, index=False)


def jpg_to_mp4_seq(data_path, frame_rate=24, verbose=1):
    """
    Combine image frames into video files for each view in each sequence in the data folder.
    """
    # Navigate through each sequence folder
    for seq in sorted(os.listdir(data_path)):
        seq_path = os.path.join(data_path, seq)
        if os.path.isdir(seq_path):
            # Navigate through each view folder within the sequence
            for view in sorted(os.listdir(seq_path)):
                view_path = os.path.join(seq_path, view)
                if os.path.isdir(view_path):
                    # List all jpg files in the current view folder
                    files = sorted([f for f in os.listdir(view_path) if f.endswith('.jpg')])
                    if not files:
                        continue
                    
                    # Determine the path for the first image to get dimensions
                    first_frame_path = os.path.join(view_path, files[0])
                    frame = cv2.imread(first_frame_path)
                    height, width, layers = frame.shape
                    
                    # Define the codec and create VideoWriter object
                    output_path = os.path.join(seq_path, f"{view}.mp4")
                    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (width, height))
                    
                    # Convert each frame to RGB and write it to the video file
                    for file in files:
                        frame_path = os.path.join(view_path, file)
                        frame = cv2.imread(frame_path)
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        out.write(frame)
                    
                    # Release the video writer
                    out.release()

                    if verbose:
                        # Print the output path of the video
                        print(f"Video written: {output_path}")

def convert_images_to_video(image_folder, output_video_file, fps=24):
    """
    Convert all images in image_folder to a video
    """
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg") or img.endswith(".png")]
    # Sort files according to the digits included in the filename
    images.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))

    # Get path of the first image to obtain frame size
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or you can use 'XVID'
    video = cv2.VideoWriter(output_video_file, fourcc, fps, (width, height))
<<<<<<< HEAD

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

=======

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

>>>>>>> origin/image_segmentation
    cv2.destroyAllWindows()
    video.release()
    
def convert_save_bgr_to_rgb(root_dir):
    """
    Convert all images in the root_dir from bgr to rgb
    """
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                file_path = os.path.join(dirpath, filename)
                # Read the image in BGR format
                bgr_image = cv2.imread(file_path)
                # Convert from BGR to RGB
                rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
                # Overwrite the original image
                cv2.imwrite(file_path, rgb_image)
                print(f"Converted and saved {file_path}")

<<<<<<< HEAD

def replace_ids_with_categories(json_file, npy_file, image_name):
    """
    Replace all ids in segmented images to their respective categories
    """
    # Load JSON data
    with open(json_file, 'r') as file:
        data = json.load(file)
    
    # Find the image data in the JSON
    image_data = next((item for item in data['annotations'] if item['file_name'] == image_name), None)
    if not image_data:
        raise ValueError(f"No data found for image {image_name}")
    
    # Load the corresponding NumPy matrix
    matrix = np.load(npy_file)
    
    # Create a dictionary mapping IDs to their category IDs
    id_to_category = {segment['id']: segment['category_id'] for segment in image_data['segments_info']}
    
    # Replace each ID in the matrix with its category ID
    unique_ids = np.unique(matrix)
    for obj_id in unique_ids:
        if obj_id in id_to_category:
            matrix[matrix == obj_id] = id_to_category[obj_id]
        else:
            # Optional: Handle the case where an ID in the matrix is not found in the JSON
            print(f"Warning: ID {obj_id} not found in JSON. Leaving unchanged.")
=======
def combine_images(source_path, dest_path):
    """
    Take all images from all the subfolders of source_path and store them in one folder in the dest_path
    """
    # Ensure the destination directory exists
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
>>>>>>> origin/image_segmentation
    
    # Walk through the source directory
    for root, dirs, files in os.walk(source_path):
        # Determine if current directory contains image files
        if files:
            # Split the path to get different parts
            parts = root.split(os.sep)
            # Extract sequence and view assuming the folder structure /Data/seq_x/view_x/
            seq = parts[-2]
            view = parts[-1]
            
            # Process each file in the directory
            for file in files:
                if file.endswith(".jpg"):
                    # Create a new filename based on the sequence, view, and original filename
                    new_filename = f"{seq}_{view}_{file}"
                    # Full path for the source and destination
                    source_file = os.path.join(root, file)
                    dest_file = os.path.join(dest_path, new_filename)
                    
                    # Copy the file to the new location with the new name
                    shutil.copy2(source_file, dest_file)


<<<<<<< HEAD
def combine_images(source_path, dest_path):
    """
    Take all images from all the subfoldes of source_path and store them in dest_path
    """
    # Ensure the destination directory exists
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    
    # Walk through the source directory
    for root, dirs, files in os.walk(source_path):
        # Determine if current directory contains image files
        if files:
            # Split the path to get different parts
            parts = root.split(os.sep)
            # Extract sequence and view assuming the folder structure /Data/seq_x/view_x/
            seq = parts[-2]
            view = parts[-1]
            
            # Process each file in the directory
            for file in files:
                if file.endswith(".jpg"):
                    # Create a new filename based on the sequence, view, and original filename
                    new_filename = f"{seq}_{view}_{file}"
                    # Full path for the source and destination
                    source_file = os.path.join(root, file)
                    dest_file = os.path.join(dest_path, new_filename)
                    
                    # Copy the file to the new location with the new name
                    shutil.copy2(source_file, dest_file)


def organize_images(source_dir, target_root_dir):
    """
    Take all images from source_dir and store them in target_root_dir in seq_x/view_x structure
    """
    # Ensure the target directory exists
    os.makedirs(target_root_dir, exist_ok=True)

    # List all files in the source directory
    for filename in os.listdir(source_dir):
        if filename.endswith(".jpg") or filename.endswith(".npz"):  # check for npz images
            # Example filename: 'seq_1_seq_1_view_1_left0000.jpg'
            parts = filename.split('_')
            
            # Extracting parts based on the expected filename format
            sequence = f"{parts[0]}_{parts[1]}"
            view = f"{parts[2]}_{parts[3]}_{parts[4]}_{parts[5]}"
            image_name = f"{parts[6]}"
            
            # Create directory path
            new_dir_path = os.path.join(target_root_dir, sequence, view)
            os.makedirs(new_dir_path, exist_ok=True)
            
            # Construct source and target file paths
            source_file_path = os.path.join(source_dir, filename)
            target_file_path = os.path.join(new_dir_path, image_name)
            
            # Move the file
            shutil.copy2(source_file_path, target_file_path)

    print("All files have been reorganized.")


# Standardize framenames of seq_11, seq_12, seq_13
def format_filename(filename):
    # Split the filename into parts based on underscore
    parts = filename.split('_')
    if len(parts) >= 4 and parts[0].startswith('seq') and parts[1].isdigit() and parts[2].startswith('Seq') and parts[3].startswith('view'):
        # Add the missing underscore after 'Seq' and 'view'
        seq_num = parts[1]
        seq_name = parts[2]
        view = parts[3]
        rest = '_'.join(parts[4:])
        
        # Reformat 'Seq' and 'view' parts to include underscores
        new_seq_name = f"{seq_name[:3]}_{seq_name[3:]}"
        new_view = f"{view[:4]}_{view[4:]}"
        
        # Combine all parts into the new filename
        new_filename = f"{parts[0]}_{seq_num}_{new_seq_name}_{new_view}_{rest}"
        return new_filename
    return filename

def rename_files(directory):
    # List all files in the directory
    for filename in os.listdir(directory):
        # Check if filename starts with the desired sequence
        if filename.startswith('seq_11') or filename.startswith('seq_12') or filename.startswith('seq_13'):
            new_filename = format_filename(filename)
            # Check if the filename was changed
            if new_filename != filename:
                # Rename the file
                os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))
                print(f"Renamed '{filename}' to '{new_filename}'")


# Funtion to check the most frequent frame counts among all the views in a sequence
def determine_most_frequent_frame_count_in_sequence(sequence_path):
=======
def organize_images(source_dir, target_root_dir):
    """
    Take all images from source_dir and store them in target_root_dir in seq_x/view_x structure
    """
    # Ensure the target directory exists
    os.makedirs(target_root_dir, exist_ok=True)

    # List all files in the source directory
    for filename in os.listdir(source_dir):
        if filename.endswith(".jpg") or filename.endswith(".npz"):  # check for npz images
            # Example filename: 'seq_1_seq_1_view_1_left0000.jpg'
            parts = filename.split('_')
            
            # Extracting parts based on the expected filename format
            sequence = f"{parts[0]}_{parts[1]}"
            view = f"{parts[2]}_{parts[3]}_{parts[4]}_{parts[5]}"
            image_name = f"{parts[6]}"
            
            # Create directory path
            new_dir_path = os.path.join(target_root_dir, sequence, view)
            os.makedirs(new_dir_path, exist_ok=True)
            
            # Construct source and target file paths
            source_file_path = os.path.join(source_dir, filename)
            target_file_path = os.path.join(new_dir_path, image_name)
            
            # Move the file
            shutil.copy2(source_file_path, target_file_path)

    print("All files have been reorganized.")


def determine_most_frequent_frame_count_in_sequence(sequence_path):
    """
    Check the most frequent frame counts among all the views in a sequence
    """
>>>>>>> origin/image_segmentation
    frame_counts = []
    for view in os.listdir(sequence_path):
        view_path = os.path.join(sequence_path, view)
        if os.path.isdir(view_path):
            frames = [f for f in os.listdir(view_path) if f.endswith('.jpg') or f.endswith('.npz')]
            frame_counts.append(len(frames))
    most_common_count = Counter(frame_counts).most_common(1)[0][0] if frame_counts else 0
    return most_common_count

<<<<<<< HEAD
# Function to resample all the videos to make the frame count in all views equal to the most frequent frame counts among all the views in a sequence
def adjust_frames_to_target(root_folder, pad_with_black=False):
=======

def adjust_frames_to_target(root_folder, pad_with_black=False):
    """
    Resample all the videos to make the frame count in all views equal to the most frequent frame counts among all the views in a sequence
    """
>>>>>>> origin/image_segmentation
    for seq in os.listdir(root_folder):
        seq_path = os.path.join(root_folder, seq)
        if os.path.isdir(seq_path):
            most_frequent_frame_count = determine_most_frequent_frame_count_in_sequence(seq_path)
            for view in os.listdir(seq_path):
                view_path = os.path.join(seq_path, view)
                if os.path.isdir(view_path):
                    frames = sorted([f for f in os.listdir(view_path) if f.endswith('.jpg') or f.endswith('.npz')])
                    num_frames = len(frames)
                    
                    # Handle fewer frames than the most frequent count
                    if num_frames < most_frequent_frame_count:
                        last_frame_path = os.path.join(view_path, frames[-1]) if frames else None
                        extension = last_frame_path.split('.')[-1] if last_frame_path else 'jpg'
                        for i in range(num_frames, most_frequent_frame_count):
                            new_frame_name = f"left{i:04d}.{extension}"
                            if pad_with_black and extension == 'jpg':
                                black_frame = Image.new('RGB', (640, 480))
                                black_frame.save(os.path.join(view_path, new_frame_name))
                            elif last_frame_path:
                                shutil.copy(last_frame_path, os.path.join(view_path, new_frame_name))

                    # Handle more frames than the most frequent count
                    elif num_frames > most_frequent_frame_count:
                        indices = np.round(np.linspace(0, num_frames - 1, most_frequent_frame_count)).astype(int)
                        selected_frames = {frames[i] for i in indices}

                        # Remove unselected frames
                        for frame in frames:
                            if frame not in selected_frames:
                                os.remove(os.path.join(view_path, frame))
                        
                        # Rename selected frames sequentially to eliminate numbering gaps
                        for i, frame in enumerate(sorted(selected_frames)):
                            os.rename(os.path.join(view_path, frame), os.path.join(view_path, f"left{i:04d}.{frame.split('.')[-1]}"))                       

    folder_name = os.path.basename(root_folder)
    print(f"Resampling for folder '{folder_name}' complete.")

def count_frames_in_views(root_folder):
<<<<<<< HEAD
=======
    """
    Count number of frames in each view in each sequence
    """
>>>>>>> origin/image_segmentation
    results = []
    for seq in os.listdir(root_folder):
        seq_path = os.path.join(root_folder, seq)
        if os.path.isdir(seq_path):
            for view in os.listdir(seq_path):
                view_path = os.path.join(seq_path, view)
                if os.path.isdir(view_path):
                    frames = [f for f in os.listdir(view_path) if f.endswith('.jpg') or f.endswith('.npz')]
                    frame_count = len(frames)
                    results.append((seq, view, frame_count))
                    print(f"Sequence: {seq}, View: {view}, Frame count: {frame_count}")
<<<<<<< HEAD

# Function to organize actions in folders like /action_x/seq_x
def organize_actions(source_dir, target_root_dir, action_labels_csv, stride=24, max_frame_diff=48):
=======
                    

def organize_actions(source_dir, target_root_dir, action_labels_csv, stride=24, max_frame_diff=48):
    '''
    Function to organize actions in folders like /action_x/seq_x.
    Reads csv and builds an action seq for each row.
    '''
>>>>>>> origin/image_segmentation
    # Read CSV
    df = pd.read_csv(action_labels_csv)

    # Dictionary to track the sequence index for each action label
    action_sequence_count = {}

    # Process each row
    for index, row in df.iterrows():
        action_label = row['action_label'].replace(" ", "_")
        if action_label not in action_sequence_count:
            action_sequence_count[action_label] = 0

        # Use extended max_frame_diff and stride for mistakes as mistakes are usually a longer sequence
        current_max_frame_diff = int(max_frame_diff * 2.5) if action_label == "mistakes" else max_frame_diff
        current_stride = int(stride * 2) if action_label == "mistakes" else stride

        # Source folder to copy files from
        source_folder = os.path.join(source_dir, row['sequence_number'], row['sequence_number'] + "_view_" + str(row['view']))

<<<<<<< HEAD
=======
        # Skip source_folder seq_13/view 3 and seq_11/view 2 because we don't have data for them
        if (source_folder == os.path.join(source_dir,'seq_11/seq_11_view_2')) or (source_folder == os.path.join(source_dir,'seq_13/seq_13_view_3')):
            continue
>>>>>>> origin/image_segmentation
        # Frame clipping and striding
        total_frames = row['frame_diff']
        start_frame = row['start frame']

        num_subgroups = (total_frames - current_stride) // current_stride + 1

        if total_frames > current_max_frame_diff:
            total_frames = current_max_frame_diff

        for subgroup_index in range(num_subgroups):
            action_sequence_count[action_label] += 1
            seq_folder_name = f"seq_{action_sequence_count[action_label]}"
            action_folder = os.path.join(target_root_dir, action_label, seq_folder_name)

            # Ensure the action folder exists
<<<<<<< HEAD
            if not os.path.exists(action_folder):
                os.makedirs(action_folder)
=======
            #if not os.path.exists(action_folder):
            #   os.makedirs(action_folder)
>>>>>>> origin/image_segmentation

            subgroup_start = start_frame + subgroup_index * current_stride
            subgroup_end = subgroup_start + current_max_frame_diff
            # Ensure the end frame does not exceed the maximum frame in the original sequence
            subgroup_end = min(subgroup_end, start_frame + row['frame_diff'])

<<<<<<< HEAD
            # Copy files for this subgroup, supporting both 'npz' and 'jpg' files
            for frame in range(subgroup_start, subgroup_end + 1):
                for extension in ['npz', 'jpg']:
=======
            if (subgroup_end - subgroup_start) < current_stride:
                continue
            # Ensure the action folder exists
            if not os.path.exists(action_folder):
                os.makedirs(action_folder)

            # Copy files for this subgroup, supporting both 'npz' and 'jpg' files
            for frame in range(subgroup_start, subgroup_end + 1):
                for extension in ['jpg']:
>>>>>>> origin/image_segmentation
                    source_file = os.path.join(source_folder, f"left{frame:04d}.{extension}")
                    target_file = os.path.join(action_folder, f"left{frame:04d}.{extension}")
                    if os.path.exists(source_file):
                        shutil.copy(source_file, target_file)
<<<<<<< HEAD
=======
                    else:
                        print('File does not exist: ', source_file)
>>>>>>> origin/image_segmentation

        if num_subgroups == 0:
            print(f"Skipping action due to insufficient frames after applying stride for row {index}")

    # Inform user about completion
    print("Organized actions into separate folders.")

<<<<<<< HEAD
def count_sequences_by_action(action_dir):
=======
def plot_sequence_length_histogram(root_dir):
    '''
    Generates a histogram of length of sequences and prints the sequence with the minimum number of frames.
    Required data structure: root_dir/action_x/seq_x
    '''
    sequence_lengths = []
    min_length = float('inf')
    min_seq_path = None

    # Traverse the root directory and record the number of files in each sequence folder
    for action in os.listdir(root_dir):  # List each action directory
        action_dir = os.path.join(root_dir, action)
        if os.path.isdir(action_dir):  # Ensure it is a directory
            for seq in os.listdir(action_dir):  # List each sequence directory within the action directory
                seq_dir = os.path.join(action_dir, seq)
                if os.path.isdir(seq_dir):
                    # Count files (assuming all entries in these directories are relevant files)
                    num_files = len([name for name in os.listdir(seq_dir) if os.path.isfile(os.path.join(seq_dir, name))])
                    sequence_lengths.append(num_files)

                    # Update minimum sequence info if current sequence has fewer files
                    if num_files < min_length:
                        min_length = num_files
                        min_seq_path = seq_dir

    # Plotting the histogram of sequence lengths
    plt.figure(figsize=(10, 6))
    plt.hist(sequence_lengths, bins=30, color='blue', edgecolor='black')
    plt.title('Histogram of Sequence Lengths')
    plt.xlabel('Length of Sequence (number of files)')
    plt.ylabel('Frequency')
    plt.show()

    # Print the sequence with the minimum number of frames
    if min_seq_path:
        print(f"The sequence with the minimum number of frames is located at '{min_seq_path}' with {min_length} frames.")
    else:
        print("No sequences found.")

def count_sequences_by_action(action_dir):
    '''
    Returns number of seqs in each action
    '''
>>>>>>> origin/image_segmentation
    # Dictionary to store count of sequences for each action
    action_counts = {}
    total_sequences = 0

    # Iterate over each subdirectory in the given action directory
    for action in os.listdir(action_dir):
        action_path = os.path.join(action_dir, action)
        if os.path.isdir(action_path):
            # Count sequences within this action
            seq_count = 0
            for item in os.listdir(action_path):
                if os.path.isdir(os.path.join(action_path, item)) and item.startswith('seq_'):
                    seq_count += 1
            
            action_counts[action] = seq_count
            total_sequences += seq_count

    return action_counts, total_sequences

# Extract labels from action folders names
def get_vids(path2ajpgs):
    listOfCats = os.listdir(path2ajpgs)
    ids = []
    labels = []
    for catg in listOfCats:
        path2catg = os.path.join(path2ajpgs, catg)
        listOfSubCats = os.listdir(path2catg)
        path2subCats = [os.path.join(path2catg, los) for los in listOfSubCats]
        ids.extend(path2subCats)
        labels.extend([catg] * len(listOfSubCats))
    return ids, labels, listOfCats

<<<<<<< HEAD

def resize_image(image_path, output_size=(224, 224)):
    with Image.open(image_path) as img:
        # Resize the image and apply the ANTIALIAS filter.
        resized_img = img.resize(output_size, Image.LANCZOS)
        resized_img.save(image_path)

# Resize all the images in the Action Dataset to 224 x 224 to match ResNet input dimensions
def resize_images_in_directory(directory_path):
    for subdir, dirs, files in os.walk(directory_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                full_path = os.path.join(subdir, file)
                resize_image(full_path)
                print(f"Resized {full_path}")

=======
>>>>>>> origin/image_segmentation
#TODO: Add mean, std logic for npz arrays as well
def compute_mean_std(root_dir):
    # Initialize variables to accumulate sums and squared sums
    num_samples = 0
<<<<<<< HEAD
    # channel_sum = np.zeros(3)
    # channel_squared_sum = np.zeros(3)
    if 'Data_Action' in root_dir:
        channel_sum = np.zeros(3)
        channel_squared_sum = np.zeros(3)
    else:
        channel_sum = np.zeros(1)
        channel_squared_sum = np.zeros(1)
=======
    channel_sum = np.zeros(3)
    channel_squared_sum = np.zeros(3)
>>>>>>> origin/image_segmentation

    # Walk through all files in the directory structure
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            # Check if the file is an image
            if file.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                file_path = os.path.join(subdir, file)
                # Open the image and convert it to RGB format
                image = Image.open(file_path).convert('RGB')
                # Convert the image to a NumPy array and normalize to [0, 1]
                image_np = np.array(image) / 255.0
                # Accumulate sums and squared sums
                channel_sum += image_np.mean(axis=(0, 1))
                channel_squared_sum += (image_np ** 2).mean(axis=(0, 1))
                num_samples += 1
<<<<<<< HEAD
            else:
                # Check if the file is an npz file
                if 'Data_Segmented_Action/Annotations' in root_dir:
                    if file.endswith('.npz'):
                        file_path = os.path.join(subdir, file)
                        # Load the npz file and extract the array
                        npz_data = np.load(file_path)
                        image_np = npz_data['arr_0']/np.max(npz_data['arr_0'])
                        # Accumulate sums and squared sums
                        channel_sum += image_np.mean(axis=(0, 1))
                        channel_squared_sum += (image_np ** 2).mean(axis=(0, 1))
                        num_samples += 1
                else:
                    if file.endswith('.npz'):
                        file_path = os.path.join(subdir, file)
                        # Load the npz file and extract the array
                        npz_data = np.load(file_path)
                        image_np = npz_data['arr_0'][0][0]
                        # Extract the tracking points from the frame
                        # Org Image dimensions
                        height = 480
                        width = 640
                        # Create a black background image of specified dimensions
                        background = np.zeros((height, width))
                        # Overlay coordinates onto the background
                        for point in image_np:
                            y, x = point  # Extract coordinates
                            if 0 <= x < width and 0 <= y < height:
                                background[int(x), int(y)] = 1  # Set pixel value to 1

                        # Accumulate sums and squared sums
                        channel_sum += background.mean(axis=(0, 1))
                        channel_squared_sum += (background ** 2).mean(axis=(0, 1))
                        num_samples += 1

=======
>>>>>>> origin/image_segmentation

    # Compute mean and standard deviation
    mean = channel_sum / num_samples
    std = np.sqrt(channel_squared_sum / num_samples - mean ** 2)

<<<<<<< HEAD
    return mean, std
=======
    return mean, std
>>>>>>> origin/image_segmentation
