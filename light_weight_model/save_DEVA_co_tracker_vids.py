import torch
import os
from maaf.co_tracker.cotracker.predictor import CoTrackerPredictor
from pathlib import Path
import numpy as np
import random
import argparse
import json
from tqdm import tqdm
import numpy as np
from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt

# initilizations
print("initilizations started ...")
parser = argparse.ArgumentParser(description='Save DEVA+co-tracker videos')
parser.add_argument('--sel_GPU', type=int, default=1, help='Select the gpu to run the model on')
parser.add_argument('--save_dir', type=str, default='/maaf/Data_Comb_Tracked', help='where to save the videos')
parser.add_argument('--grid_size', type=int, default=60, help='Size of the feature vector')
parser.add_argument('--read_dir', type=str, default='/root/maaf/Data_Comb_Tracked', help='Folder to read dataset files')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = ".".join(str(args.sel_GPU))

checkpoint_dir = args.save_dir
os.makedirs(checkpoint_dir, exist_ok=True)


# Load the co-tracker model
feature_model = CoTrackerPredictor(
    checkpoint=os.path.join(
        '/root/maaf/co_tracker/checkpoints/cotracker2.pth'
    )
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# if torch.cuda.is_available():
feature_model = feature_model.to(device)


file_names_list = get_frame_names(args.read_dir)

for file in file_names_list:
    data = np.load(file)

    segm_mask = data['arr_0']

    video = torch.zeros_like(torch.from_numpy(segm_mask))

    pred_tracks, _ = feature_model(video[None,None,None].repeat(1, 1, 3, 1, 1).to(torch.float).to(device), grid_size=60,segm_mask=torch.from_numpy(np.array(segm_mask))[None,None].to(torch.float).to(device))

    np.savez_compressed(pred_tracks)
# data = np.load('/root/maaf/Data_Comb_Seg/Annotations/seq_1/seq_1_view_4/left0100.npz')

# segm_mask = data['arr_0']

# video = torch.zeros_like(torch.from_numpy(segm_mask))

# pred_tracks, _ = feature_model(video[None,None,None].repeat(1, 1, 3, 1, 1).to(torch.float).to(device), grid_size=60,segm_mask=torch.from_numpy(np.array(segm_mask))[None,None].to(torch.float).to(device))

# np.savez_compressed(pred_tracks)