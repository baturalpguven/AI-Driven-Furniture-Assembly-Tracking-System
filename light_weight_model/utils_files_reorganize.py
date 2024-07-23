import torch
import numpy as np
from pathlib import Path
import os
import argparse
from tqdm import tqdm
import sys

# Add the parent directory of `maaf` to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Function to extract the numerical part of the filename
def extract_part_number(path):
    filename = path.stem  # get the filename without the extension
    part_number = filename.split('part')[-1]  # extract the part number
    return int(part_number)

# Directory path
dir_path = Path('/root/maaf/Data_Comb_Tracked_new/seq_1/seq_1_view_1')

# Get all .npz files and sort them using the custom key
sorted_files = sorted(dir_path.glob('*.npz'), key=extract_part_number)

def main(args):
    # Prepare directories
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    # Traverse directories and process files
    base_dir = Path(args.read_dir)
    for seq_dir in tqdm(sorted(base_dir.glob('seq_*')), desc='Processing sequences', unit='seqs', leave=False):
        for view_dir in tqdm(sorted(seq_dir.glob('*')), desc='Processing views', unit='views', leave=False):
            global_frame_num = 0
            for npz_file in tqdm(sorted(view_dir.glob('*.npz'),key=extract_part_number), desc='Processing files', unit='frames', leave=False):
                # Load data
                data = np.load(npz_file)
                pred_tracks = data['tracks']
                pred_vis = data['vis']


                for frame_num in range(pred_tracks.shape[1]):
                    part_name = f'left{str(global_frame_num).zfill(4)}.npz'  # Change 1 to 0001

                    # Save output in corresponding structure in new base directory
                    save_path = Path(args.save_dir) / seq_dir.name / view_dir.name
                    save_path.mkdir(parents=True, exist_ok=True)

                    np.savez_compressed(save_path / part_name, tracks=pred_tracks[:,frame_num,...], vis=pred_vis[:,frame_num,...])
                    global_frame_num += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Change_file_structure')
    parser.add_argument('--save_dir', type=str, default='/root/maaf/Data_Comb_Tracked_new_organized', help='Directory to save processed videos.')
    parser.add_argument('--read_dir', type=str, default='/root/maaf/Data_Comb_Tracked_new', help='Directory to read .npz files.')
    args = parser.parse_args()
    main(args)
