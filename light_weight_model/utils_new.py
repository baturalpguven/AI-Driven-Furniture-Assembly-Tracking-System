import numpy as np
import pandas as pd
import os
from pathlib import Path
import json
from tqdm import tqdm
import time
import cv2

def load_npz_frame_co_tracker(npz_path, max_length=10000):
    """Load a frame from a .npz file, pad it by repeating the first point, and append a repeat count.

    Args:
        npz_path (str): Path to the .npz file containing the frame data.
        max_length (int): Maximum length to which the frame should be padded.

    Returns:
        np.ndarray: The padded frame with an additional dimension for repeat count.

    Example:
        In training use padded_frame[padded_frame[0,0,-1]] to get the actual tracking points
    """
    with np.load(npz_path) as data:
        frame = data['arr_0'][0, 0]
        current_length = len(frame)
        repeat_count = max_length - current_length
        padding = np.tile(frame[0], (repeat_count, 1))
        padded_frame = np.vstack([frame, padding])
        repeat_count_array = np.full((padded_frame.shape[0], 1), repeat_count)
        padded_frame = np.hstack([padded_frame, repeat_count_array])
    return padded_frame

def load_npz_frame_deva(npz_path):
    """Load a single frame from a .npz file.

    Args:
        npz_path (str): Path to the .npz file containing the frame data.

    Returns:
        np.ndarray: The frame data extracted from the .npz file.
    """
    with np.load(npz_path) as data:
        frame = data['arr_0'][0, 0]
    return frame

def resample_or_pad_video_frames(video, target_length):
    """Resample or pad video frames to a uniform length, either by repeating the last frame or using linear sampling.

    Args:
        video (np.ndarray): Array of video frames.
        target_length (int): Target length to achieve for the video frames.

    Returns:
        np.ndarray: Video data adjusted to the target length.
    """
    current_length = len(video)
    if target_length > current_length:
        last_frame = video[-1]
        padding = np.repeat(last_frame[np.newaxis, ...], target_length - current_length, axis=0)
        video = np.concatenate((video, padding), axis=0)
    elif target_length < current_length:
        indices = np.linspace(0, current_length - 1, num=target_length, dtype=int)
        video = video[indices]
    return video

def load_and_process_views(sequence_path, sequence_number, label_view_frame_count, type='co_tracker', max_length=10000):
    """Load and adjust all views within a sequence to the same frame count, handling different types of datasets.

    Args:
        sequence_path (str): Base path to the sequence data.
        sequence_number (str): Identifier for the specific sequence.
        label_view_frame_count (int): Target frame count determined by the labeled view.
        type (str): Type of dataset being processed ('co_tracker' or 'deva').
        max_length (int): Maximum length for padding in 'co_tracker' dataset type.

    Returns:
        np.ndarray: Stacked adjusted video data from all views.
    """
    views_data = []
    for view_num in tqdm(range(1, 9), desc='Processing Views', leave=False):
        view_path = os.path.join(sequence_path, f"{sequence_number}_view_{view_num}")
        frames = sorted(os.listdir(view_path))
        if type == 'co_tracker':
            video = np.array([load_npz_frame_co_tracker(os.path.join(view_path, frame), max_length=max_length) for frame in frames])
        elif type == 'deva':
            video = np.array([load_npz_frame_deva(os.path.join(view_path, frame)) for frame in frames])
        adjusted_video = resample_or_pad_video_frames(video, label_view_frame_count)
        views_data.append(adjusted_video)
    return np.stack(views_data, axis=-1)  # Different shapes based on dataset type

def chunk_actions(video_data, max_chunk_size=50, stride=25):
    """Generate chunks from video data with potential overlap, discarding undersized chunks.

    Args:
        video_data (np.ndarray): Array of video data to be chunked.
        max_chunk_size (int): Maximum number of frames per chunk.
        stride (int): Step size to advance for each new chunk (determines overlap).

    Returns:
        list: List of video data chunks that meet the size requirements.
    """
    num_frames = video_data.shape[0]
    chunks = []
    for start in range(0, num_frames, stride):
        end = start + max_chunk_size
        if end > num_frames:
            end = num_frames
        if end - start >= stride:
            chunks.append(video_data[start:end])
        if num_frames - (start + stride) < stride:
            break
    return chunks

def load_data_labels_only_actions(label_csv, base_path,class_path,d_type='co_tracker',max_length_co_tracker=10000,stride=25,max_chunk_size=50):
    """
    Load and process action data from labeled sequences, returning chunks of actions based on specified parameters.

    This function reads a CSV file containing labels for actions in video sequences and processes each sequence 
    to extract chunks of frames corresponding to these actions. The frames are processed to ensure uniform length
    and are chunked into overlapping segments if specified. The function supports handling of different types 
    of data processing based on the dataset type ('co_tracker' or 'deva').

    Args:
        label_csv (str): Path to the CSV file containing action labels and metadata.
        base_path (str): The base directory where the sequence data is stored.
        class_path (str): Path to the JSON file containing class definitions for actions.
        d_type (str, optional): The type of dataset processing to apply ('co_tracker' or 'deva'). Defaults to 'co_tracker'.
        max_length_co_tracker (int, optional): The maximum length to which frame sequences should be padded or trimmed. Defaults to 10000.
        stride (int, optional): The number of frames to step forward to create a new chunk, allowing overlap. Defaults to 25.
        max_chunk_size (int, optional): The maximum number of frames per chunk. Defaults to 50.

    Returns:
        np.ndarray: An array of tuples, where each tuple contains a chunk of video data and its corresponding action class.

    Notes:
        - The function dynamically adjusts to process different views within each sequence and manages overlapping chunks.
        - Each action frame range is processed separately and can support different sequence lengths within the same batch of data.
        - Chunks smaller than the stride are discarded to maintain uniform chunk sizes.
    """
    labels_df = pd.read_csv(label_csv)
    all_actions = []
    # Load class definitions from JSON
    with open(class_path, 'r') as file:
        classes = json.load(file)["classes"]

    prev_sequence_number = None

    for iter, row in tqdm(labels_df.iterrows(),desc='Loading data', total = 126, leave=False):
        sequence_number = row['sequence_number']
        start_frame = row['start frame']
        end_frame = row['end frame']
        view_name = row['view']
        sequence_path = os.path.join(base_path, f"{sequence_number}")

        # Determine the target frame count based on the labeled view
        if prev_sequence_number != sequence_number:
            label_view_frame_count = len(os.listdir(os.path.join(sequence_path,f"{sequence_number}_view_{view_name}")))
            video_data = load_and_process_views(sequence_path,sequence_number, label_view_frame_count,type=d_type,  max_length=max_length_co_tracker)

        # Extract and chunk the action frames across all views
        action_frames = video_data[start_frame:end_frame+1]
        action_chunks = chunk_actions(action_frames,stride=stride,max_chunk_size=max_chunk_size)
        for chunk in action_chunks:
            all_actions.append((chunk, classes[row['action_label']]))
        prev_sequence_number = sequence_number
        print(sequence_number,'Action:',row['action_label'],'up to line',iter,'has been loaded',end='\r')

    return np.array(all_actions, dtype=object)

def read_points_for_demo(base_dir="/root/maaf/Data_Segmented_Tracked_equal/seq_3/",view_numbers=[1,2,4,7]):
    """
    Reads and processes .npz files from multiple view directories within a specified base directory.

    This function navigates through specified subdirectories (views) of a base directory, loads all .npz files
    found within each view directory, and compiles the data into a structured numpy array.

    Parameters:
    - base_dir (str): The root directory where view subdirectories are located. Default is a pre-set path.
    - view_numbers (list of int): A list of view numbers to process, corresponding to subdirectories in the base directory.

    Returns:
    - numpy.ndarray: A numpy array containing the processed data from all .npz files in the specified view directories.
    
    Each element of the outer array corresponds to a different view and contains an array of data loaded from
    the .npz files of that view.
    """

    data = []
    seq_name = base_dir.split('/')[-2]
    for view in view_numbers:   
        selected_view_path = os.path.join(base_dir, f"{seq_name}_view_{view}")
        for subdir,dirs,files in (os.walk(selected_view_path)):
            view_data = []
            for file in sorted(files):
                view_data.append(load_npz_frame_co_tracker(os.path.join(subdir,file)))
            data.append(np.array(view_data))
    data = np.transpose(data, axes=(1,2,3,0))
    return np.array(data)

def read_imgs_for_demo(base_dir="/root/maaf/Data_Comb/seq_3/",view_numbers=[1,2,4,7]):
    """
    Reads and processes .npz files from multiple view directories within a specified base directory.

    This function navigates through specified subdirectories (views) of a base directory, loads all .npz files
    found within each view directory, and compiles the data into a structured numpy array.

    Parameters:
    - base_dir (str): The root directory where view subdirectories are located. Default is a pre-set path.
    - view_numbers (list of int): A list of view numbers to process, corresponding to subdirectories in the base directory.

    Returns:
    - numpy.ndarray: A numpy array containing the processed data from all .npz files in the specified view directories.
    
    Each element of the outer array corresponds to a different view and contains an array of data loaded from
    the .npz files of that view.
    """
    
    data = []
    seq_name = base_dir.split('/')[-2]
    for view in view_numbers:   
        selected_view_path = os.path.join(base_dir, f"{seq_name}_view_{view}")
        for subdir,dirs,files in (os.walk(selected_view_path)):
            view_data = []
            for file in sorted(files):
                frame = cv2.imread(os.path.join(subdir,file))
                # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                view_data.append(frame)
            data.append(np.array(view_data))

    return np.array(data)


# Usage example
if __name__ == "__main__":

    directory = Path(__file__).parent 

    # # test for co-tracker dataset
    # start = time.time()
    # base_path = directory / 'Data_Comb_Tracked'
    # label_csv = directory / 'Data_Comb_Tracked/action_labels.csv'
    # class_path = directory / 'classes.json'
    # processed_actions = load_data_labels_only_actions(label_csv, base_path,class_path,d_type='co_tracker',max_length_co_tracker=10000,stride=25)
    # np.savez_compressed('/root/maaf/processed_datas/processed_actions_co_tracker.npz', processed_actions=processed_actions)
    # print(processed_actions.shape)
    # print('Time taken(s):',time.time()-start)

    # test for DEVA dataset
    start = time.time()
    base_path = directory / 'Data_Comb_Seg/Annotations'
    label_csv = directory / 'Data_Comb_Tracked/action_labels.csv'
    class_path = directory / 'classes.json'
    processed_actions = load_data_labels_only_actions(label_csv, base_path,class_path,d_type='deva',max_length_co_tracker=10000,stride=25)
    np.savez_compressed('/root/maaf/processed_datas/processed_actions_deva.npz', processed_actions=processed_actions)
    print(processed_actions.shape)
    print('Time taken(s):',time.time()-start)
