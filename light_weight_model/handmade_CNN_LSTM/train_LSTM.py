import sys
import os

# Add the parent directory of `maaf` to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from maaf.light_weight_model.dataloaderMultiview import MultiVideoDatasetStrideAllVideosSequentiallyLSTM
from torch.utils.data import DataLoader
import torch
from model_LSTM import ViewAwareLSTM
from maaf.light_weight_model.co_tracker.cotracker.predictor import CoTrackerPredictor
from pathlib import Path
import numpy as np
import random
import argparse
from maaf.light_weight_model.utils import get_class_counts
import wandb
from sklearn.metrics import f1_score
import datetime
import json
import time
from tabulate import tabulate
import json
import torch.nn as nn
from tqdm import tqdm



# initilizations
print("initilizations started ...")
parser = argparse.ArgumentParser(description='Transfomer model parameters')
parser.add_argument('--hidden_size', type=int, default=256, help='Number of heads in the transformer')
parser.add_argument('--num_layers', type=int, default=2, help='Number of encoder layers in the transformer')
parser.add_argument('--grid_size', type=int, default=20, help='Number of track points for co-tracker')
parser.add_argument('--sel_GPU', type=int, default=1, help='Select the gpu to run the model on')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train the model')
parser.add_argument('--early_stopping_threshold',type=float,default=0.001,help='Early stopping threshold')
parser.add_argument('--use_wandb',type=bool,default=False,help='Save the results to wandb')
parser.add_argument('--name',type=str,default='co_tracker+lstm',help='Name of the experiment')
parser.add_argument('--frames',type=int,default=10,help='Number of frames to consider for the model')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = ".".join(str(args.sel_GPU))

if os.environ.get("DEBUG_MODE") != "True":
    # Convert args to a dictionary
    args_dict = vars(args)


    # Create a table with the args
    table = tabulate(args_dict.items(), headers=["Argument", "Value"], tablefmt="grid")

    # Print the table
    print(table)

    name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")+'_LSTM'
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

# Define a function for processing a batch of videos
def pre_process(video_batch):
    if video_batch.mean() != 0:
        processed_frames = []
        for view in tqdm(range(video_batch.shape[2]), desc='Processing views'):
            pred_tracks, _ = feature_model(video_batch[:,:,view,...], grid_size=args.grid_size)
            processed_frames.append(pred_tracks)
        return torch.stack(processed_frames, dim=-1)
    else:
        return torch.zeros(video_batch.shape[0], video_batch.shape[1], args.grid_size**2,2,video_batch.shape[2]).to(device)

# Load the sequences
directory = Path(__file__).parent.parent

csv_path = directory / "dataset/annotations.csv"
dataset_path = directory / "dataset"
class_path = directory / "classes.json"

with open(class_path, 'r') as file:
    classes = json.load(file)["classes"]

seq_names = ['seq_1', 'seq_2', 'seq_3', 'seq_4', 'seq_5', 'seq_6', 'seq_7',
              'seq_8', 'seq_9', 'seq_10', 'seq_11', 'seq_12', 'seq_13']

# seq_names = ['seq_4', 'seq_4', 'seq_4']


random.seed(33)
random.shuffle(seq_names)

# shuffle the dataset
train_size = int(0.8 * len(seq_names))
train_seq_names = seq_names[:train_size]
validation_seq_names = seq_names[train_size:]

print("dataset is loading ...")
# Load the dataset
print(f"Training sequences: {train_seq_names}"
      f"\nValidation sequences: {validation_seq_names}")

video_dataset_train = MultiVideoDatasetStrideAllVideosSequentiallyLSTM(train_seq_names, dataset_path, csv_path, class_path, frame_chunk_size=args.frames)
video_loader_train = DataLoader(video_dataset_train, batch_size=1, shuffle=False, num_workers=0, drop_last=False)


video_dataset_val = MultiVideoDatasetStrideAllVideosSequentiallyLSTM(validation_seq_names, dataset_path, csv_path, class_path, frame_chunk_size=args.frames)
video_loader_val = DataLoader(video_dataset_val, batch_size=1, shuffle=False, num_workers=0, drop_last=False)

# Define your models and optimizer
temporal_model = ViewAwareLSTM(num_features=args.grid_size**2*2*8, hidden_size=args.hidden_size,num_layers=args.num_layers, num_classes=len(classes), device=device)

# Assuming optimizer has been defined
optimizer = torch.optim.Adam(temporal_model.parameters(), lr=0.001)

# Define the learning rate scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)


# Define loss function
class_counts = get_class_counts(csv_path, class_path)
class_weights = torch.tensor([np.sum(class_counts) / x for x in class_counts], dtype=torch.float32)
# if torch.cuda.is_available():
class_weights = class_weights.cuda().to(device)
loss_function = nn.CrossEntropyLoss(weight=class_weights)
# loss_function = nn.CrossEntropyLoss()

# Import the necessary libraries
# Initialize wandb

if args.use_wandb:
    wandb.init(project="maaf", entity="btrgvn", name=name)

# Define variables to track the lowest validation loss and corresponding epoch
lowest_val_loss = float('inf')
best_epoch = -1

print("training starts ...")

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
    train_accs = []
    hidden = None
    for iter,(video, labels) in enumerate(video_loader_train):
        video = video.to(device)  
        labels = labels.to(device)
        
        
        # Forward pass
        start_time = time.time()

        processed_video = pre_process(video)

        if processed_video.mean() == 0:
            hidden = None
            continue

        pred_labels,hidden = temporal_model(processed_video.to(device),hidden)

        forward_pass_time = time.time() - start_time

        
        # Backward and optimize
        loss = loss_function(pred_labels.view(-1,len(classes)), labels.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        # Collect predictions and labels for F1 score and accuracy calculation
        train_pred = torch.argmax(pred_labels.squeeze(), dim=-1).view(-1).cpu().numpy()
        true_label = labels.squeeze().view(-1).cpu().numpy()

        # Calculate F1 score
        train_f1 = f1_score(train_pred, true_label, average='weighted')
        train_f1s.append(train_f1)

        # Calculate accuracy
        train_acc = (train_pred == true_label).mean() * 100
        train_accs.append(train_acc)

        print(f'Epoch [{epoch+1}/{num_epochs}], Iteration {iter}, Train Loss: {train_loss/(iter+1):.2f}, \
            Train F1 Score: {(train_f1):.2f}, Train Accuracy: {train_acc:.2f}%, Forward pass time: {forward_pass_time:.2f}')

        # Log metrics to wandb
        if args.use_wandb:
            wandb.log({"train_loss": train_loss/(iter+1), "Train f1_score": (train_f1), "Train accuracy": (train_acc)})

    # Validation Phase
    temporal_model.eval()
    val_loss = 0
    val_f1s = []
    val_accs = []
    hidden = None
    with torch.no_grad():
        for iter,(video, labels) in enumerate(video_loader_val):
            video = video.to(device)  
            labels = labels.to(device)
            processed_video = pre_process(video)
            if processed_video.mean() == 0:
               hidden = None
               continue
            pred_labels,hidden = temporal_model(processed_video.to(device),hidden)
            loss = loss_function(pred_labels.view(-1,len(classes)), labels.view(-1))
            val_loss += loss.item()

            # Collect predictions and labels for F1 score and accuracy calculation
            val_pred = torch.argmax(pred_labels.squeeze(), dim=-1).view(-1).cpu().numpy()
            true_label = labels.squeeze().view(-1).cpu().numpy()

            # Calculate F1 score
            val_f1 = f1_score(val_pred, true_label, average='weighted')
            val_f1s.append(val_f1)

            # Calculate accuracy
            val_acc = (val_pred == true_label).mean() * 100
            val_accs.append(val_acc)

            print(f'Epoch [{epoch+1}/{num_epochs}], Val Loss: {val_loss/(iter+1):.2f}, Val F1 Score: {(val_f1):.2f}, Val Accuracy: {(val_acc):.2f}%')
                    # Log metrics to wandb
            if args.use_wandb:
                wandb.log({"val_loss": val_loss/(iter+1), "Val f1_score": (val_f1), "Val accuracy": (val_acc)})

    # Save weights based on lowest validation loss
    if val_loss < lowest_val_loss:


        lowest_val_loss = val_loss
        best_epoch = epoch

        checkpoint_path = os.path.join(checkpoint_dir, 'best_model_weightsLSTM.pth')
        torch.save(temporal_model.state_dict(), checkpoint_path)


    # Log metrics to wandb
    if args.use_wandb:
        wandb.log({"epoch": epoch+1, "avg train_loss": train_loss/len(video_loader_train), "avg val_loss": val_loss/len(video_loader_val), \
                "avg Train f1_score": np.mean(train_f1s), "avg Val f1_score": np.mean(val_f1s), \
                "avg Train accuracy": np.mean(train_accs), "avg Val accuracy": np.mean(val_accs)})
    # Print Epoch Summary
    print('\n')
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss/len(video_loader_train):.2f}, Validation Loss: {val_loss/len(video_loader_val):.2f}, \
        Train F1 Score: {np.mean(train_f1s):.2f}, Val F1 Score: {np.mean(val_f1s):.2f}, Train Accuracy: {np.mean(train_accs):.2f}%, Val Accuracy: {np.mean(val_accs):.2f}% ')


    # Check if the current validation loss is below the early stopping threshold
    if val_loss/len(video_loader_val) < early_stopping_threshold:
        break  # Stop training if the threshold is reached
    scheduler.step()  # Update the learning rate
# Print best epoch and lowest validation loss
print(f'Best Epoch: {best_epoch+1}, Lowest Validation Loss: {lowest_val_loss}')

