import sys
import os

# Add the parent directory of `maaf` to the Python path
path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(path)

import torch
import os
from maaf.handmade_transformer.model_Transformer_new import ViewAwareTransformer
import numpy as np
import json
import torch.nn as nn
from argparse import Namespace
from maaf.utils_new import read_points_for_demo,read_imgs_for_demo
import cv2
import pandas as pd
from collections import defaultdict
from scipy.stats import mode
from tqdm import tqdm

## load the necessat files and the classes

class_path = "/root/maaf/classes_only_actions.json"

with open(class_path, 'r') as file:
    classes = json.load(file)["classes"]

# Load the saved files
# model_results_dir = '/root/maaf/checkpoint/2024-07-15-09-40-53_transfomer_action_onlyViewAwareTransformerDifferentViews'
model_results_dir = '/root/maaf/checkpoint/2024-07-15-10-25-02_transfomer_action_onlyViewAwareTransformerDifferentViewsWithMistakes'

# Read the args.json file
args_file_path = os.path.join(model_results_dir, 'args.json')
with open(args_file_path, 'r') as f:
    args = json.load(f)
    args = Namespace(**args)

# get the selected views
view_numbers = np.array(args.selected_views) +1

data = read_points_for_demo(base_dir="/root/maaf/Data_Segmented_Tracked_equal/seq_6/",view_numbers=view_numbers)
data = data[:,args.num_features:,:2,:]
frames = read_imgs_for_demo(base_dir="/root/maaf/Data_Comb/seq_6/",view_numbers=[5])
frames = frames[0]


# Set up the path and ensure directory exists
save_path = '/root/maaf/demo'
os.makedirs(save_path, exist_ok=True)

output_video_path = os.path.join(save_path, 'rgb.mp4')
# Define the codec and initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Change codec to XVID for testing
output_video = cv2.VideoWriter(output_video_path, fourcc, 24, (640, 480))

# Check if the VideoWriter has been successfully opened
if not output_video.isOpened():
    print("Error: VideoWriter not initialized properly")
else:
    # Assuming 'frames' is a list of frames already in BGR format
    for frame in frames:
        output_video.write(frame)

    # Release the video writer to ensure the file is closed properly
    output_video.release()
    print("Video saved successfully.")



os.environ["CUDA_VISIBLE_DEVICES"] = ".".join(str(args.sel_GPU))
os.environ['TORCH_USE_CUDA_DSA'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
print("Number of GPUs: ", torch.cuda.device_count())


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load the saved weights
weights_file_path = os.path.join(model_results_dir, 'best_model_weights.pth')
saved_weights = torch.load(weights_file_path)


# init the model
temporal_model = ViewAwareTransformer(num_features=(10000 - args.num_features), num_views=args.num_views, num_classes=len(classes), num_layers=args.num_layers, nhead=args.num_heads,device=device,dropout=args.dr_rate).to(device)
temporal_model.load_state_dict(saved_weights)
temporal_model.eval()


# Process the data
stride = 5
window_size = 50
num_frames = data.shape[0]
results = []
inv_label_dict = {v: k for k, v in classes.items()}

for start in tqdm(range(0, num_frames - window_size + 1, stride),desc='Processing frames'):
    end = start + window_size
    chunk = data[start:end]
    chunk_tensor = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0).to(device)
    pred = temporal_model(chunk_tensor).argmax(dim=1).item() 
    results.append((start, pred))  # Store the start index and the prediction

# Now, aggregate the predictions
# Assume the last complete window ends at `num_frames - window_size`
# We will use a list to compile all predictions and apply majority vote at the end
compiled_predictions = [None] * num_frames

# Fill the compiled_predictions list with all predictions, overlapping will overwrite
for start, pred in results:
    end = start + window_size
    for i in range(start, min(end, num_frames)):
        if compiled_predictions[i] is None:
            compiled_predictions[i] = []
        compiled_predictions[i].append(pred)

# Apply majority voting for each position
final_predictions = [mode(preds).mode if preds else None for preds in compiled_predictions]

# Clean up to ensure no gaps in the final predictions
# Optionally fill gaps with the nearest valid prediction or a default value
for i in range(len(final_predictions)):
    if final_predictions[i] is None:
        final_predictions[i] = final_predictions[i-1]  # Fill with the previous prediction, or use another method



string_predictions = [inv_label_dict[pred] if pred in inv_label_dict else "no action" for pred in final_predictions]

# Convert to DataFrame and save to CSV
df = pd.DataFrame(string_predictions, columns=['Predicted_Label'])

df.to_csv(os.path.join(save_path, 'pred_labels.csv'), index=False)


### save the pred video
# Set up the save path and ensure the directory exists
save_path = '/root/maaf/demo_light_weight_model'
os.makedirs(save_path, exist_ok=True)
output_video_path = os.path.join(save_path, 'pred_video.mp4')

# Define the codec and initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Using XVID codec
output_video = cv2.VideoWriter(output_video_path, fourcc, 24, (640, 480))

# Check if the VideoWriter has been successfully opened
if not output_video.isOpened():
    print("Error: VideoWriter not initialized properly")
else:
    # Assuming 'frames' is a list of frames already in BGR format
    # Assuming 'string_predictions' is the list with predictions for each frame
    for idx, frame in enumerate(frames):
        if idx < len(string_predictions):
            text = string_predictions[idx]
            # Put text on the frame
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        
        # Write the frame to the video
        output_video.write(frame)

    # Release the video writer to ensure the file is closed properly
    output_video.release()
    print("Video saved successfully.")