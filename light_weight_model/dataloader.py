from utils import load_data_labels,display_frame_from_viewpoint,print_array_memory_usage
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from pathlib import Path

class VideoDataset(Dataset):
    def __init__(self, seq_name, dataset_path, csv_path, class_path, frame_chunk_size=10):
        """
        Args:
            dataset_path (Path): Path to the dataset directory containing videos.
            seq_name (str): Sequence name, such as 'seq_1', indicating which sequence to load.
            frame_chunk_size (int): Number of frames per chunk.
        """
        self.dataset_path = dataset_path
        self.seq_name = seq_name
        self.frame_chunk_size = frame_chunk_size
        self.load_data_labels = load_data_labels
        self.video, self.labels = self.load_data_labels(seq_name, dataset_path, csv_path, class_path)
        
        # Calculate the total number of chunks
        self.num_chunks = self.video.shape[0] // self.frame_chunk_size

    def __len__(self):
        return self.num_chunks

    def __getitem__(self, idx):
        # print("idx",idx)
        start_frame = idx * self.frame_chunk_size
        end_frame = start_frame + self.frame_chunk_size
        video_chunk = self.video[start_frame:end_frame]  # Shape: (10, 340, 400, 3, 8)
        label_chunk = self.labels[start_frame:end_frame]  # Shape: (10,)
        
        # Convert to PyTorch tensors
        video_chunk = torch.from_numpy(video_chunk).float().permute(0,4,3,1,2)  # (10,8,3,480,640)
        label_chunk = torch.from_numpy(label_chunk).long()
        # # ensure the video is not corrupted
        # if os.getenv('DEBUG_MODE') == 'True':
        #     display_frame_from_viewpoint(video_chunk, seq_name, frame_idx=0, view_idx=4)
        
        return video_chunk, label_chunk

# Example of how to use this Dataset with DataLoader
if __name__ == "__main__":
    directory = Path(__file__).parent 

    csv_path = directory / "dataset/annotations.csv"
    dataset_path = directory / "dataset"
    class_path = directory / "classes.json"

    seq_name = 'seq_4'  # Example sequence name

    video_dataset = VideoDataset(seq_name, dataset_path, csv_path, class_path, frame_chunk_size=50) # due to memory issues we are using a frame_chunk_size of 1 !!! Generate volume and increase the frame chunk size
    video_loader = DataLoader(video_dataset, batch_size=2, shuffle=False, num_workers=0, drop_last=True)

    for _,(video_chunk, label_chunk) in enumerate(video_loader):
        print(f'Video chunk shape: {video_chunk.shape}')  # Should be (1, frame_chunk_size, 8, 480, 640, 3)
        print(f'Label chunk shape: {label_chunk.shape}')  # Should be (1, frame_chunk_size)
        print(len(video_dataset))
        break  # Process data or perform training steps
