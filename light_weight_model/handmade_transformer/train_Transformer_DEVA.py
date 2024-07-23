from dataloaderMultiview import MultiVideoDataset,MultiVideoDatasetStride,MultiVideoDatasetStrideAllVideosSequentially
from torch.utils.data import DataLoader
import torch
import os
from model_Transformer import ViewAwareTransformer
from co_tracker.cotracker.predictor import CoTrackerPredictor
from pathlib import Path
import numpy as np
import random
import argparse
from utils import get_class_counts
import wandb
from sklearn.metrics import f1_score
import datetime
import json
import time
from tabulate import tabulate
import json
import torch.nn as nn
from DEVA_segmantation_mask import get_segmentation_mask
from tqdm import tqdm

# initilizations
print("initilizations started ...")
parser = argparse.ArgumentParser(description='Transfomer model parameters')
parser.add_argument('--num_views', type=int, default=8, help='Number of multiview')
parser.add_argument('--num_heads', type=int, default=10, help='Number of heads in the transformer')
parser.add_argument('--num_layers', type=int, default=10, help='Number of encoder layers in the transformer')
parser.add_argument('--num_features', type=int, default=500, help='Number of track points for co-tracker')
parser.add_argument('--sel_GPU', type=int, default=1, help='Select the gpu to run the model on')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train the model')
parser.add_argument('--early_stopping_threshold',type=float,default=0.001,help='Early stopping threshold')
parser.add_argument('--use_wandb',type=bool,default=False,help='Save the results to wandb')
parser.add_argument('--name',type=str,default='co_tracker+transformer',help='Name of the experiment')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = ".".join(str(args.sel_GPU))

# Convert args to a dictionary
args_dict = vars(args)


# Create a table with the args
table = tabulate(args_dict.items(), headers=["Argument", "Value"], tablefmt="grid")

# Print the table
print(table)

name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
checkpoint_dir = f'/root/maaf/checkpoint/{name}'
os.makedirs(checkpoint_dir, exist_ok=True)


# Save args as a JSON file with spaces
args_file_path = os.path.join(checkpoint_dir, 'args.json')
with open(args_file_path, 'w') as args_file:
    json.dump(args_dict, args_file, indent=4)


# Load the co-tracker model
feature_model = CoTrackerPredictor(
    checkpoint=os.path.join(
        '/root/maaf/co_tracker/checkpoints/cotracker2.pth'
    )
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# if torch.cuda.is_available():
feature_model = feature_model.to(device)


    
# # Define a function for processing a batch of videos
def pre_process(video_batch):
    processed_frames = []
    max_frames = video_batch.shape[1]
    max_views = video_batch.shape[2]
    for view in tqdm(range(max_views), desc='Processing views'):
        masks = []
        for frame in range(max_frames):
            one_frame = video_batch.squeeze()[frame, view, :, :].permute(1, 2, 0)
            mask = get_segmentation_mask(one_frame.cpu().numpy().astype(np.uint8))
            masks.append(mask)
        pred_tracks, _ = feature_model(video_batch[:,:,view,...], grid_size=40,segm_mask=torch.from_numpy(np.array(masks))[None].to(torch.float))
        processed_frames.append(pred_tracks)
    
    # Pad outputs with zeros to ensure same size at -2
    for i in range(len(processed_frames)):
        pad_frames = args.num_features - processed_frames[i].shape[-2]
        if pad_frames > 0:
            padding = torch.zeros((processed_frames[i].shape[0],processed_frames[i].shape[1], pad_frames, processed_frames[i].shape[-1]), dtype=processed_frames[i].dtype)
            processed_frames[i] = torch.cat([(processed_frames[i]).to(device), padding.to(device)], dim=-2)
    
    return torch.stack(processed_frames, dim=-1)



# Load the sequences
directory = Path(__file__).parent 

csv_path = directory / "dataset/annotations.csv"
dataset_path = directory / "dataset"
class_path = directory / "classes.json"

with open(class_path, 'r') as file:
    classes = json.load(file)["classes"]

seq_names = ['seq_1', 'seq_2', 'seq_3', 'seq_4', 'seq_5', 'seq_6', 'seq_7',
              'seq_8', 'seq_9', 'seq_10', 'seq_11', 'seq_12', 'seq_13']

# seq_names = ['seq_4','seq_4', 'seq_4']

random.seed(42)
random.shuffle(seq_names)

# shuffle the dataset
train_size = int(0.8 * len(seq_names))
train_seq_names = seq_names[:train_size]
validation_seq_names = seq_names[train_size:]

print("dataset is loading ...")
# Load the dataset
print(f"Training sequences: {train_seq_names}"
      f"\nValidation sequences: {validation_seq_names}")
# video_dataset_train = MultiVideoDataset(train_seq_names, dataset_path, csv_path, class_path, frame_chunk_size=10)
# video_dataset_train = MultiVideoDatasetStride(train_seq_names, dataset_path, csv_path, class_path, frame_chunk_size=10,stride=5)
video_dataset_train = MultiVideoDatasetStrideAllVideosSequentially(train_seq_names, dataset_path, csv_path, class_path, frame_chunk_size=10,stride=5)
video_loader_train = DataLoader(video_dataset_train, batch_size=1, shuffle=False, num_workers=0, drop_last=True)

# video_dataset_val = MultiVideoDataset(validation_seq_names, dataset_path, csv_path, class_path, frame_chunk_size=10)
# video_dataset_val = MultiVideoDatasetStride(validation_seq_names, dataset_path, csv_path, class_path, frame_chunk_size=10,stride=5)
video_dataset_val = MultiVideoDatasetStrideAllVideosSequentially(validation_seq_names, dataset_path, csv_path, class_path, frame_chunk_size=10,stride=5)
video_loader_val = DataLoader(video_dataset_val, batch_size=1, shuffle=False, num_workers=0, drop_last=True)

# Define your models and optimizer
temporal_model = ViewAwareTransformer(num_features=args.num_features, num_views=args.num_views, num_classes=len(classes), num_layers=args.num_layers, nhead=args.num_heads,device=device)

optimizer = torch.optim.Adam(temporal_model.parameters(), lr=0.001)

# Define your loss function
class_counts = get_class_counts(csv_path, class_path)
class_weights = torch.tensor([1.0 / x for x in class_counts], dtype=torch.float32)
# if torch.cuda.is_available():
class_weights = class_weights.cuda().to(device)
loss_function = nn.CrossEntropyLoss(weight=class_weights)

# Import the necessary libraries
# Initialize wandb

if args.use_wandb:
    wandb.init(project="maaf", entity="btrgvn", name=name)

# Define variables to track the lowest validation loss and corresponding epoch
lowest_val_loss = float('inf')
best_epoch = -1

print("training starts ...")

# can you pass me th model shape to print int terminal and show me the input shape to the model
# Print model shape
print("Model shape:", temporal_model)
print("Input shape to the model:", video_loader_train.dataset[0][0].shape)

# Training and Validation Loop
num_epochs = args.epochs # Define the number of epochs
early_stopping_threshold = args.early_stopping_threshold  # Set the early stopping threshold

for epoch in range(num_epochs):
    # Training Phase
    temporal_model.train()
    train_loss = 0
    train_f1s = []
    for iter,(video, labels) in enumerate(video_loader_train):
        video = video.to(device)  
        labels = labels.to(device)
        
        
        # Forward pass
        start_time = time.time()

        processed_video = pre_process(video)
        pred_labels = temporal_model(processed_video.to(device))

        forward_pass_time = time.time() - start_time

        
        # Backward and optimize
        loss = loss_function(pred_labels.view(-1,len(classes)), labels.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        # Collect predictions and labels for F1 score calculation
        train_pred = (torch.argmax(pred_labels.squeeze(), dim=-1).view(-1).cpu().numpy())
        true_label = (labels.squeeze().view(-1).cpu().numpy())

        # Calculate F1 score
        train_f1 = f1_score(train_pred, true_label, average='weighted')
        train_f1s.append(train_f1)

        print(f'Epoch [{epoch+1}/{num_epochs}],Iteration,{iter}, Train Loss: {train_loss/len(video_loader_train)}, Train F1 Score: { np.mean(train_f1s)} , Forward pass time:, {forward_pass_time}')


    # Validation Phase
    temporal_model.eval()
    val_loss = 0
    val_f1s = []
    with torch.no_grad():
        for video, labels in video_loader_val:
            video = video.to(device)  
            labels = labels.to(device)
            processed_video = pre_process(video)
            pred_labels = temporal_model(processed_video.to(device))
            loss = loss_function(pred_labels.view(-1,len(classes)), labels.view(-1))
            val_loss += loss.item()

            # Collect predictions and labels for F1 score calculation
            val_pred = (torch.argmax(pred_labels.squeeze(), dim=-1).view(-1).cpu().numpy())
            true_label = (labels.squeeze().view(-1).cpu().numpy())

            # Calculate F1 score
            val_f1 = f1_score(val_pred, true_label, average='weighted')
            val_f1s.append(val_f1)
            print(f'Epoch [{epoch+1}/{num_epochs}], Val Loss: {val_loss/len(video_loader_val)}, Val F1 Score: { np.mean(val_f1s)}')

    # Save weights based on lowest validation loss
    if val_loss < lowest_val_loss:


        lowest_val_loss = val_loss
        best_epoch = epoch

        checkpoint_path = os.path.join(checkpoint_dir, 'best_model_weights.pth')
        torch.save(temporal_model.state_dict(), checkpoint_path)


    # Log metrics to wandb
    if args.use_wandb:
        wandb.log({"epoch": epoch+1, "train_loss": train_loss/len(video_loader_train), "val_loss": val_loss/len(video_loader_val), "Train f1_score": np.mean(train_f1s), "Val f1_score": np.mean(val_f1s)})

    # Print Epoch Summary
    print('\n')
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss/len(video_loader_train)},Validation Loss: {val_loss/len(video_loader_val)}, Train F1 Score: { np.mean(train_f1s)} , Val F1 Score: { np.mean(val_f1s)}, Forward pass time:, {forward_pass_time} s')


    # Check if the current validation loss is below the early stopping threshold
    if val_loss/len(video_loader_val) < early_stopping_threshold:
        break  # Stop training if the threshold is reached

# Print best epoch and lowest validation loss
print(f'Best Epoch: {best_epoch+1}, Lowest Validation Loss: {lowest_val_loss}')

