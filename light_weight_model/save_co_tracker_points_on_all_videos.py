import torch
import numpy as np
from pathlib import Path
import os
import argparse
from tqdm import tqdm
import cv2

import sys

# Add the parent directory of `maaf` to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from maaf.co_tracker.cotracker.predictor import CoTrackerPredictor

def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.sel_GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model
    model = CoTrackerPredictor(checkpoint='/root/maaf/co_tracker/checkpoints/cotracker2.pth')
    model = model.to(device)

    # Prepare directories
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    # Traverse directories and process files
    base_dir = Path(args.read_dir)
    rgb_dir = Path(args.rgb_dir)
    for seq_dir in tqdm(sorted(base_dir.glob('seq_*')), desc='Processing sequences', unit='seqs', leave=False):
        for view_dir in tqdm(sorted(seq_dir.glob('*')), desc='Processing views', unit='views', leave=False):
            video = []
            masks = []
            for npz_file in tqdm(sorted(view_dir.glob('*.npz')), desc='Processing files', unit='frames', leave=False):
                
                if view_dir.name == 'seq_1_view_1' or view_dir.name == 'seq_1_view_2' or view_dir.name == 'seq_1_view_3':
                    continue
                # Load data
                data = np.load(npz_file)
                segm_mask = data['arr_0']

                frame_path = rgb_dir / seq_dir.name / view_dir.name / f'{npz_file.stem}.jpg'
                bgr_frame = cv2.imread(str(frame_path))
                
                if bgr_frame is None:
                    print(f"Warning: Frame {frame_path} could not be loaded.")
                    continue
                
                rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
                
                if int(npz_file.stem.split('t')[-1]) >= 100:
                    video.append(rgb_frame)
                    masks.append(segm_mask)
                
            if not video:
                print(f"Warning: No valid frames were loaded for processing in directory {view_dir}.")
                continue
            
            # Convert the video and segmentation mask to tensors
            video = torch.from_numpy(np.array(video))
            segm_mask = masks[0]

            # Define the block size
            block_size = 50
            num_blocks = (video.shape[0] + block_size - 1) // block_size  # Calculate the number of blocks

            # Loop through each block
            for i in tqdm(range(num_blocks), desc='Processing blocks', unit='blocks', leave=False):
                start_frame = i * block_size
                end_frame = min((i + 1) * block_size, video.shape[0])  # Ensure we don't go out of bounds
                video_block = video[start_frame:end_frame]

                # Prepare video and segmentation mask inputs for the model
                video_input = video_block[None].permute(0, 1, 4, 2, 3).float().to(device)
                segm_mask_input = torch.from_numpy(np.array(segm_mask))[None, None].float().to(device)

                # Get prediction
                pred_tracks, pred_vis = model(video_input, grid_size=args.grid_size, segm_mask=segm_mask_input)

                # Save output in corresponding structure in new base directory
                save_path = Path(args.save_dir) / seq_dir.name / view_dir.name
                save_path.mkdir(parents=True, exist_ok=True)

                # Save each part separately
                part_name = f"part{i+1}.npz"
                np.savez_compressed(save_path / part_name, tracks=pred_tracks.cpu().numpy(), vis=pred_vis.cpu().numpy())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process and track video segments.')
    parser.add_argument('--sel_GPU', type=int, default=0, help='GPU index to use.')
    parser.add_argument('--save_dir', type=str, default='/root/maaf/Data_Comb_Tracked_new', help='Directory to save processed videos.')
    parser.add_argument('--grid_size', type=int, default=60, help='Grid size for the model.')
    parser.add_argument('--read_dir', type=str, default='/root/maaf/Data_Comb_Seg/Annotations', help='Directory to read .npz files.')
    parser.add_argument('--rgb_dir', type=str, default='/root/maaf/Data_Comb', help='Directory to read RGB frames.')
    args = parser.parse_args()
    main(args)
