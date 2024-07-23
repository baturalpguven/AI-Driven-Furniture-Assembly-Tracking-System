import os
import json
import numpy as np
import cv2
from pathlib import Path
import pandas as pd
import argparse
import json
import matplotlib.pyplot as plt

def display_frame_from_viewpoint(video_chunk, seq_name, frame_idx, view_idx):
    """
    Displays a specific frame from a specific viewpoint in a multi-view video array.

    Args:
        video_chunk (np.ndarray): A multi-view video array with dimensions (frames,views, height, width, channels ).
        seq_name (str): Name of the sequence.
        frame_idx (int): Index of the frame to display.
        view_idx (int): Index of the viewpoint to display.

    Returns:
        None: This function only visualizes the frame and does not return any values.
    """
    # Validate indices
    if frame_idx >= video_chunk.shape[0]:
        raise ValueError(f"frame_idx is out of bounds. Maximum allowed is {video_chunk.shape[0] - 1}")
    if view_idx >= video_chunk.shape[1]:
        raise ValueError(f"view_idx is out of bounds. Maximum allowed is {video_chunk.shape[1] - 1}")

    # Extract the specific frame from the specific viewpoint
    frame_to_show = video_chunk[frame_idx,view_idx, ... ]

    # Determine if the image has color channels
    if frame_to_show.shape[-1] == 3:  # Assumes the third dimension in frame_to_show is channels
        plt.imshow(frame_to_show.cpu().detach().numpy().astype(np.uint8) )
    else:
        plt.imshow(frame_to_show.cpu().detach().numpy().astype(np.uint8), cmap='gray')  # For grayscale images

    plt.title(f'Frame {frame_idx + 1} from Viewpoint {view_idx + 1}')
    plt.axis('off')  # Hide the axis to focus on the image
    plt.savefig(f'{seq_name}.png')
    plt.show()


def print_array_memory_usage(array, name="Array"):
    """Prints the memory usage of a NumPy array in a human-readable format.
    
    Args:
        array (np.ndarray): The NumPy array whose memory usage is to be checked.
        name (str): The name of the array (for display purposes).
    """
    size_bytes = array.nbytes
    size_kb = size_bytes / 1024
    size_mb = size_kb / 1024
    size_gb = size_mb / 1024

    if size_gb > 1:
        print(f"{name} uses approximately {size_gb:.2f} GB")
    elif size_mb > 1:
        print(f"{name} uses approximately {size_mb:.2f} MB")
    elif size_kb > 1:
        print(f"{name} uses approximately {size_kb:.2f} KB")
    else:
        print(f"{name} uses approximately {size_bytes} bytes")

def create_memmap(filename, dtype, shape):
    """ Create a memory-mapped array with the given dtype and shape. """
    mmap = np.memmap(filename, dtype=dtype, mode='w+', shape=shape)
    return mmap

def save_data(seq_name, multi_view_video, labels, directory='dataset_processed'):
    """ Save the video and labels data to disk using memory mapping. """
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Determine data types and shapes from the inputs
    video_dtype = multi_view_video.dtype
    video_shape = multi_view_video.shape
    
    label_dtype = labels.dtype
    label_shape = labels.shape
    
    # Create memory-mapped arrays
    video_mmap = create_memmap(f'{directory}/multi_view_video_{seq_name}.dat', video_dtype, video_shape)
    label_mmap = create_memmap(f'{directory}/labels_{seq_name}.dat', label_dtype, label_shape)
    
    # Copy data to memory-mapped arrays
    video_mmap[:] = multi_view_video[:]
    label_mmap[:] = labels[:]
    
    # Ensure data is written to disk
    del video_mmap, label_mmap


def read_video_from_path(video_path):
    """
    Reads a video file from the specified path using OpenCV and extracts all frames,
    converting them to RGB format for consistency in image processing.
    
    Args:
    video_path (str): The file path to the video.

    Returns:
    tuple: A tuple containing a numpy array of frames and the frame rate of the video.
    """
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return np.stack(frames), frame_rate

def get_length(video_path):
    """
    Reads a video file from the specified path using OpenCV and extracts all frames,
    converting them to RGB format for consistency in image processing.
    
    Args:
    video_path (str): The file path to the video.

    Returns:
    tuple: A tuple containing a numpy array of frames and the frame rate of the video.
    """
    
    cap = cv2.VideoCapture(video_path)
    # get me the length of the video
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return  length

def read_csv_from_path(csv_path):
    """
    Loads a CSV file into a pandas DataFrame. Assumes the first row is a header.

    Args:
    csv_path (str): The file path to the CSV file.

    Returns:
    DataFrame: A pandas DataFrame containing the data from the CSV file.
    """
    return pd.read_csv(csv_path, skiprows=1)

import re

def get_seq_name(video_path):
    """
    Extracts the sequence name from the video file path using regex to capture 'seq_' followed by digits.

    Args:
    video_path (str): The file path of the video.

    Returns:
    str: The extracted sequence name like 'seq_4'.
    """
    # Get the last part of the file path after the last '/'
    file_name = video_path.split("/")[-1]
    # Search for a pattern 'seq_' followed by any digits
    match = re.search(r'seq_\d+', file_name)
    if match:
        return match.group()  # Returns the matched part of the string
    return None  # Return None if no match is found

def get_view_num(video_path):
    """
    Extracts the view number from the video file path.

    Args:
    video_path (str): The file path of the video.

    Returns:
    int: The view number subtracted by one, as an index.
    """
    return int(video_path.split('.')[0][-1])

def resample_video_frames(video, target_length):
    """Resample or pad video frames to match the target length."""
    current_length = video.shape[0]
    if target_length > current_length:
        # Pad video by repeating the last frame
        last_frame = video[-1]
        padding = np.repeat(last_frame[np.newaxis, :], target_length - current_length, axis=0)
        video = np.concatenate((video, padding), axis=0)
    elif target_length < current_length:
        indices = np.linspace(0, current_length - 1, num=target_length, dtype=int)
        video = video[indices]
    return video

def get_files_for_sequence(df, sequence_name):
    """
    Retrieves file names for a given sequence number.

    Args:
    df (pd.DataFrame): The DataFrame containing the file information.
    sequence_name (str): The sequence name to filter by, e.g., 'seq_14'.

    Returns:
    list: A list of file names corresponding to the sequence name.
    """
    # Filter the DataFrame by the sequence number
    filtered_df = df[df['sequence_number'] == sequence_name]
    # Get the unique file names from the filtered DataFrame
    file_names = filtered_df['file_list'].unique().tolist()
    return file_names[0]


def parse_file_list(file_list_str):
    """
    Parses a JSON string representing a file list and returns the first element.

    Args:
        file_list_str (str): The JSON string representing the file list.

    Returns:
        The first element of the parsed file list, or None if the JSON string is invalid.
    """
    try:
        # Convert the JSON string into a list and extract the first element
        return json.loads(file_list_str.replace("'", '"'))[0]
    except json.JSONDecodeError:
        return None
    
# Define a function to parse the JSON-like string
def parse_meta_data(file_list_str):
    """
    Parse the meta data from a JSON string.

    Args:
        file_list_str (str): The JSON string containing the file list.

    Returns:
        dict or None: The parsed meta data as a dictionary, or None if the JSON string is invalid.
    """
    try:
        # Convert the JSON string into a list and extract the first element
        return json.loads(file_list_str.replace("{", '}'))[0]
    except json.JSONDecodeError:
        return None
def extract_temporal_segments(metadata_str):
    """
    Extracts the value of 'TEMPORAL-SEGMENTS' from a JSON string.

    Args:
        metadata_str (str): The JSON string containing the metadata.

    Returns:
        str: The value of 'TEMPORAL-SEGMENTS' if found, otherwise 'Not specified'.
             If the JSON string is invalid, returns 'Invalid JSON'.
    """
    try:
        # Load the string as a JSON object
        metadata_json = json.loads(metadata_str)
        # Extract the 'TEMPORAL-SEGMENTS' value
        return metadata_json.get('TEMPORAL-SEGMENTS', 'Not specified')  # Default to 'Not specified' if key not found
    except json.JSONDecodeError:
        return 'Invalid JSON'  # Return a placeholder if JSON is invalid
    
def load_data_labels(seq_name, dataset_path, csv_path, class_path):
    """
    Integrates video data with annotations and class indices. This function loads the video,
    reads annotations from a CSV file, and assigns class indices to each video frame based on the annotations.

    Args:
    dataset_path (str): The file path to the video.
    csv_path (str): The file path to the CSV annotations.
    class_path (str): The file path to the JSON file containing class indices.

    Returns:
    tuple: A dict containing a multi-view video array and a corresponding label array.
    """
    with open(class_path, 'r') as file:
        classes = json.load(file)["classes"]

    # video, frame_rate = read_video_from_path(dataset_path)
    annotations = read_csv_from_path(csv_path)
    annotations['file_list'] = annotations['file_list'].apply(parse_file_list)
    annotations['metadata'] = annotations['metadata'].apply(extract_temporal_segments)
    annotations.drop('# CSV_HEADER = metadata_id', axis=1, inplace=True)
    # Apply the function and create new columns
    annotations['sequence_number'] = annotations['file_list'].apply(get_seq_name)
    annotations['view'] = annotations['file_list'].apply(get_view_num)

    multi_view_video = []
    for current_name in os.listdir(dataset_path):

        #### Process the video
        if seq_name == get_seq_name(current_name):

            view_number = get_view_num(current_name)-1
            video, frame_rate = read_video_from_path(dataset_path / current_name)
            label_length = get_length(dataset_path / get_files_for_sequence(annotations,seq_name))

            if len(video) != label_length: # Resample or pad video frames to match the target length
                video = resample_video_frames(video, label_length)
            multi_view_video.append((view_number,video ))

            
             #### Process the labels
            label = np.zeros(label_length)
            # get these values from the annotation based on the seq input 
            filtered_df  = annotations[annotations['sequence_number']==seq_name] 
            # Iterate over each row in the filtered DataFrame
            for _, row in filtered_df.iterrows():

                start_second, end_second = row['temporal_segment_start'], row['temporal_segment_end']
                class_name = row['metadata']

                start_frame, end_frame = int(float(start_second) * frame_rate), int(float(end_second) * frame_rate)
                label[start_frame:end_frame + 1] = classes[class_name]
                    
    multi_view_video_sorted = sorted(multi_view_video, key=lambda x: x[0])
    # Extract just the arrays in their new sorted order
    multi_view_video_final = [array for _, array in multi_view_video_sorted]
    return np.stack(multi_view_video_final,axis=4), label



def get_class_counts(csv_path, class_path):
    """
    Calculates the count of each class based on annotations in a CSV file,
    aligning these counts with class indices specified in a JSON file.
    
    Args:
        csv_path (str): Path to the annotations CSV file.
        class_path (str): Path to the JSON file that contains class definitions.
        
    Returns:
        class_counts list: A list where indices represent class indices and values represent the count of each class.
    """
    # Load class definitions from JSON
    with open(class_path, 'r') as file:
        classes = json.load(file)["classes"]

    # Load annotations from the CSV file
    annotations = read_csv_from_path(csv_path)
    annotations['file_list'] = annotations['file_list'].apply(parse_file_list)
    annotations['metadata'] = annotations['metadata'].apply(extract_temporal_segments)
    annotations.drop('# CSV_HEADER = metadata_id', axis=1, inplace=True)
    annotations['sequence_number'] = annotations['file_list'].apply(get_seq_name)
    annotations['view'] = annotations['file_list'].apply(get_view_num)
    
    # Map metadata to class indices and count occurrences
    class_counts = annotations['metadata'].map(classes).value_counts().sort_index()

    # Initialize a series with zeros for all class indices to ensure all classes are represented
    all_class_counts = pd.Series(index=range(len(classes)), data=0)  # Create a series with all zeros
    all_class_counts.update(class_counts)  # Update with actual counts
    all_class_counts[0]=50 # add arbitrary big value to the first class to show that it is the most frequent class

    return list(all_class_counts)

if __name__ == "__main__":

    directory = Path(__file__).parent 

    csv_path = directory / "dataset/annotations.csv"
    dataset_path = directory / "dataset"
    class_path = directory / "classes.json"

    argparse = argparse.ArgumentParser()
    argparse.add_argument('--seq_num', type=int, default=1,help='Sequence number to process')
    args = argparse.parse_args()
    num = args.seq_num

    # Note dataset is very big do not try to read all the data at once it occupues 200G
    seq_name = f'seq_{num}'

    print('Extracing class counts...')
    class_counts = get_class_counts(csv_path, class_path)
    print("Class counts extracted successfully.")
    print(class_counts)

    print('Loading data and labels...')

    multi_view_video, label = load_data_labels(seq_name, dataset_path, csv_path, class_path)
    print('Data and labels loaded successfully.')
    print_array_memory_usage(multi_view_video, name='Multi-view video')
    print("Shape of the multi-view video is: ", multi_view_video.shape)
    print('dtype of the multi-view video is: ', multi_view_video.dtype)

    # DO NOT SAVE Unless you have extra 200G of space!!!!

    # print('Saving data and labels to memory-mapped files...')
    # save_data(seq_name, multi_view_video, label, directory='dataset_processed')

    # print(f'Data for {seq_name} has been saved.')
