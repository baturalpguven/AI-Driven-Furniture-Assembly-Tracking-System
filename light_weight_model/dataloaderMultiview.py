import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from pathlib import Path
from maaf.utils import load_data_labels,print_array_memory_usage
from tqdm import tqdm

class MultiVideoDataset(Dataset):
    def __init__(self, seq_names, dataset_path, csv_path, class_path, frame_chunk_size=10):
        """
        Initializes a dataset that handles multiple video sequences.

        Args:
            seq_names (list): List of sequence names to load.
            dataset_path (Path): Path to the dataset directory containing videos.
            csv_path (Path): Path to the CSV file with annotations.
            class_path (Path): Path to the JSON file with class mappings.
            frame_chunk_size (int): Number of frames per chunk.
        """
        self.dataset_path = dataset_path
        self.seq_names = seq_names
        self.frame_chunk_size = frame_chunk_size
        self.load_data_labels = load_data_labels  # Function to load data and labels
        self.videos = []
        self.labels = []
        self.num_chunks_per_video = []

        # Load each sequence
        for seq_name in seq_names:
            video, labels = self.load_data_labels(seq_name, dataset_path, csv_path, class_path)
            self.videos.append(video)
            self.labels.append(labels)
            self.num_chunks_per_video.append(video.shape[0] // self.frame_chunk_size)

    def __len__(self):
        # Total chunks across all videos
        return max(self.num_chunks_per_video)

    def __getitem__(self, idx):

        video_chunks=[]
        label_chunks=[]

        for video,label in zip(self.videos,self.labels): 
            start_frame = idx%len(video) * self.frame_chunk_size
            end_frame = start_frame + self.frame_chunk_size
            video_chunk = video[start_frame:end_frame]  # Shape: (10, 340, 400, 3, 8)
            label_chunk = label[start_frame:end_frame]  # Shape: (10,)
            
            video_chunks.append(video_chunk)
            label_chunks.append(label_chunk)

        # Convert to PyTorch tensors
        video_chunks = torch.from_numpy(np.stack(video_chunks)).float().permute(0,1,5,4,2,3)  # (batch_size,10,8,3,480,640)
        label_chunks = torch.from_numpy(np.stack(label_chunks)).long()

    
        return video_chunks, label_chunks


class MultiVideoDatasetStride(Dataset):
    def __init__(self, seq_names, dataset_path, csv_path, class_path, frame_chunk_size=10, stride=5):
        """
        Initializes a dataset that handles multiple video sequences with striding logic.
        
        Args:
            seq_names (list): List of sequence names to load.
            dataset_path (Path): Path to the dataset directory containing videos.
            csv_path (Path): Path to the CSV file with annotations.
            class_path (Path): Path to the JSON file with class mappings.
            frame_chunk_size (int): Number of frames per chunk.
            stride (int): Number of frames to stride between chunks.
        """
        self.dataset_path = dataset_path
        self.seq_names = seq_names
        self.frame_chunk_size = frame_chunk_size
        self.load_data_labels = load_data_labels  # Function to load data and labels
        self.stride = stride
        self.videos = []
        self.labels = []
        self.num_chunks_per_video = []



        # Load each sequence
        for seq_name in seq_names:
            video, labels = self.load_data_labels(seq_name, dataset_path, csv_path, class_path)
            self.videos.append(video)
            self.labels.append(labels)
            num_chunks = (video.shape[0] - frame_chunk_size) // stride + 1
            self.num_chunks_per_video.append(num_chunks)

    def __len__(self):
        # Total chunks across all videos
        return min(self.num_chunks_per_video)

    def __getitem__(self, idx):
        video_chunks = []
        label_chunks = []

        for video, label in zip(self.videos, self.labels):
            start_frame = idx%len(video) * self.stride
            end_frame = start_frame + self.frame_chunk_size
            video_chunk = video[start_frame:end_frame]  # Shape: (10, 340, 400, 3, 8)
            label_chunk = label[start_frame:end_frame]  # Shape: (10,)
            
            video_chunks.append(video_chunk)
            label_chunks.append(label_chunk)

        # Convert to PyTorch tensors
        video_chunks = torch.from_numpy(np.stack(video_chunks)).float().permute(0,1,5,4,2,3)  # (batch_size,10,8,3,480,640)
        label_chunks = torch.from_numpy(np.stack(label_chunks)).long()
        
        return video_chunks, label_chunks


class MultiVideoDatasetStrideAllVideosSequentiallyLSTM(Dataset):
    def __init__(self, seq_names, dataset_path, csv_path, class_path, frame_chunk_size=10, stride=10):
        self.dataset_path = dataset_path
        self.seq_names = seq_names
        self.frame_chunk_size = frame_chunk_size
        self.stride = stride
        self.load_data_labels = load_data_labels  # Assuming this is defined elsewhere
        self.chunks = []

        # Load each sequence and prepare chunks
        for seq_name in tqdm(seq_names, desc='loading sequences'):
            video, labels = self.load_data_labels(seq_name, dataset_path, csv_path, class_path) # (full_frames,8,3,480,640)
            num_frames = video.shape[0]

            # Process all full chunks and potentially partial chunks
            start = 0
            while start < num_frames:
                end = start + frame_chunk_size
                if end > num_frames:
                    # If the end exceeds the number of frames, use what is available
                    end = num_frames

                # Extract the chunk and append to list
                video_chunk = video[start:end]
                label_chunk = labels[start:end]
                self.chunks.append((video_chunk, label_chunk))

                start += frame_chunk_size

            # Append a single frame of zeros at the end of each video to indicate the end of the sequence
            zero_frame = np.zeros((1,) + video.shape[1:], dtype=video.dtype)  # Adjust shape to match video frame dimensions
            zero_label = np.array([-1], dtype=labels.dtype)  # Corresponding zero label
            self.chunks.append((zero_frame, zero_label))

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        video_chunk, label_chunk = self.chunks[idx]
        video_chunk = torch.from_numpy(video_chunk).float().permute(0, 4, 3, 1, 2)  # (10,8,3,480,640)
        label_chunk = torch.from_numpy(label_chunk).long()
        return video_chunk, label_chunk


class MultiVideoDatasetStrideAllVideosSequentiallyTransformer(Dataset):
    def __init__(self, seq_names, dataset_path, csv_path, class_path, frame_chunk_size=10, stride=5):
        self.dataset_path = dataset_path
        self.seq_names = seq_names
        self.frame_chunk_size = frame_chunk_size
        self.stride = stride
        self.load_data_labels = load_data_labels  # Assuming this is defined elsewhere
        self.chunks = []

        # Load each sequence and prepare chunks
        for seq_name in tqdm(seq_names, desc='loading sequences'):
            video, labels = self.load_data_labels(seq_name, dataset_path, csv_path, class_path) # (full_frames,8,3,480,640)
            num_frames = video.shape[0]

            # Calculate the number of chunks needed to ensure each video_chunk has the same number of frames
            num_chunks = (num_frames - frame_chunk_size) // stride + 1

            # Process all full chunks and potentially partial chunks
            for i in range(num_chunks):
                start = i * stride
                end = start + frame_chunk_size

                # Extract the chunk and append to list
                video_chunk = video[start:end]
                label_chunk = labels[start:end]
                self.chunks.append((video_chunk, label_chunk))

                
    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        video_chunk, label_chunk = self.chunks[idx]
        video_chunk = torch.from_numpy(video_chunk).float().permute(0, 4, 3, 1, 2)  # (10,8,3,480,640)
        label_chunk = torch.from_numpy(label_chunk).long()
        return video_chunk, label_chunk



# Example of how to use this Dataset with DataLoader
if __name__ == "__main__":
    # directory = Path(__file__).parent
    # csv_path = directory / "dataset/annotations.csv"
    # dataset_path = directory / "dataset"
    # class_path = directory / "classes.json"

    # # seq_names = ['seq_1', 'seq_2', 'seq_3', 'seq_4']  # Example sequence names
    # seq_names = ['seq_4']  # Example sequence names
    # video_dataset = MultiVideoDataset(seq_names, dataset_path, csv_path, class_path, frame_chunk_size=10)
    # video_loader = DataLoader(video_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=True)

    # for video_chunk, label_chunk in video_loader:
    #     print(f'Video chunk shape: {video_chunk.shape}')  # shape should be (1,batch_size, frame_chunk_size, 8, 480, 640, 3)
    #     print(f'Label chunk shape: {label_chunk.shape}')
    #     # print(f'Sample from each video: {video_chunk[0][0:2],video_chunk[1][0:2]}')
    #     print_array_memory_usage(video_chunk)
    #     break # Process data or perform training steps

    # # Example of how to use this Dataset with DataLoader for MultiVideoDatasetStride

    # directory = Path(__file__).parent
    # csv_path = directory / "dataset/annotations.csv"
    # dataset_path = directory / "dataset"
    # class_path = directory / "classes.json"

    # # seq_names = ['seq_1', 'seq_2', 'seq_3', 'seq_4']  # Example sequence names
    # seq_names = ['seq_4','seq_2']  # Example sequence names
    # video_dataset = MultiVideoDatasetStride(seq_names, dataset_path, csv_path, class_path, frame_chunk_size=10,stride=5)
    # video_loader = DataLoader(video_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=True)

    # for iter,(video_chunk, label_chunk) in enumerate(video_loader):
    #     print(f'Video chunk shape: {video_chunk.shape}')  # shape should be (1,batch_size, frame_chunk_size, 8, 480, 640, 3)
    #     print(f'Label chunk shape: {label_chunk.shape}')
    #     # print(f'Sample from each video: {video_chunk[0][0:2],video_chunk[1][0:2]}')
    #     print_array_memory_usage(video_chunk)
    #     if iter==2:
    #         break # Process data or perform training steps

    # Example of how to use this Dataset with DataLoader for MultiVideoDatasetStrideAllVideosSequentially
    directory = Path(__file__).parent
    csv_path = directory / "dataset/annotations.csv"
    dataset_path = directory / "dataset"
    class_path = directory / "classes.json"

    seq_names = ['seq_1']  # Example sequence names
    # seq_names = ['seq_1', 'seq_2', 'seq_3', 'seq_4', 'seq_5', 'seq_6', 'seq_7',
            #   'seq_8', 'seq_9', 'seq_10', 'seq_11', 'seq_12', 'seq_13']
    
    video_dataset = MultiVideoDatasetStrideAllVideosSequentiallyLSTM(seq_names, dataset_path, csv_path, class_path, frame_chunk_size=10,stride=5)
    video_loader = DataLoader(video_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)

    for iter,(video_chunk, label_chunk) in enumerate(video_loader):
        print('data:',iter)
        print(f'Video chunk shape: {video_chunk.shape}')  # shape should be (1,batch_size, frame_chunk_size, 8, 480, 640, 3)
        print(f'Label chunk shape: {label_chunk.shape}')

        # print(f'Sample from each video: {video_chunk[0][0:2],video_chunk[1][0:2]}')
        print_array_memory_usage(video_chunk)
        # if iter==120:
        #     break # Process data or perform training steps