import torch
import numpy as np
from pathlib import Path
import os
import argparse
from tqdm import tqdm
from tqdm import tqdm

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
    for seq_dir in tqdm(base_dir.glob('seq_*'), desc='Processing sequences', unit = 'seqs',leave=False):
        for view_dir in tqdm(seq_dir.glob('*'), desc='Processing views', unit = 'views',leave=False):
            for npz_file in tqdm(view_dir.glob('*.npz'), desc='Processing files', unit = 'files',leave=False):
                # Load data
                data = np.load(npz_file)
                segm_mask = data['arr_0']
                
                # Prepare input for model
                video = torch.zeros_like(torch.from_numpy(segm_mask))
                video_input = video[None, None, None].repeat(1, 1, 3, 1, 1).float().to(device)
                segm_mask_input = torch.from_numpy(np.array(segm_mask))[None, None].float().to(device)
                
                # Get prediction
                pred_tracks, _ = model(video_input, grid_size=args.grid_size, segm_mask=segm_mask_input)
                
                # Save output in corresponding structure in new base directory
                save_path = Path(args.save_dir) / seq_dir.name / view_dir.name
                save_path.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(save_path / npz_file.name, pred_tracks.cpu().numpy())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process and track video segments.')
    parser.add_argument('--sel_GPU', type=int, default=1, help='GPU index to use.')
    parser.add_argument('--save_dir', type=str, default='/root/maaf/Data_Comb_Tracked', help='Directory to save processed videos.')
    parser.add_argument('--grid_size', type=int, default=60, help='Grid size for the model.')
    parser.add_argument('--read_dir', type=str, default='/root/maaf/Data_Comb_Seg/Annotations', help='Directory to read .npz files.')
    args = parser.parse_args()
    main(args)
