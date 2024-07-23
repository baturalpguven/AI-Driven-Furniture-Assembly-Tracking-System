import sys
import os

# Add the parent directory of `maaf` to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from maaf.light_weight_model.dataloader_new import get_processed_dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch
import os
from model_Transformer_new import ViewAwareTransformer
from pathlib import Path
import numpy as np
import argparse
import wandb
from sklearn.metrics import f1_score
import datetime
import json
import time
from tabulate import tabulate
import json
import torch.nn as nn
# from torch.optim.lr_scheduler import ReduceLROnPlateau


# initilizations
print("initilizations started ...")
parser = argparse.ArgumentParser(description='Transfomer model parameters')
parser.add_argument('--num_views', type=int, default=-1, help='Number of multiview')
parser.add_argument('--num_heads', type=int, default=8, help='Number of heads in the transformer')
parser.add_argument('--num_layers', type=int, default=2, help='Number of encoder layers in the transformer')
parser.add_argument('--num_features', type=int, default=9500, help='Number of track points for co-tracker')
parser.add_argument('--sel_GPU', type=int, default=1, help='Select the gpu to run the model on')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train the model')
parser.add_argument('--use_wandb',action='store_true',default=False,help='Save the results to wandb')
parser.add_argument('--name',type=str,default='co_tracker+transformer+new',help='Name of the experiment')
parser.add_argument('--batch_size',type=int,default=1,help='Batch size for the model')
parser.add_argument('--lr',type=float,default=1e-5,help='lr for the model')
parser.add_argument('--dr_rate',type=float,default=0.5,help='Dropout rate for the model')
parser.add_argument("--selected_views", nargs="+", type=int, default=[0, 1, 3, 6], help="Selected views")

args = parser.parse_args()

args.num_views = len(args.selected_views)

os.environ["CUDA_VISIBLE_DEVICES"] = ".".join(str(args.sel_GPU))
os.environ['TORCH_USE_CUDA_DSA'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
print("Number of GPUs: ", torch.cuda.device_count())

import torchtune.modules as tune


# Convert args to a dictionary
args_dict = vars(args)


# Create a table with the args
table = tabulate(args_dict.items(), headers=["Argument", "Value"], tablefmt="grid")

# Print the table
print(table)

name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")+'_transfomer_action_only' + args.name
checkpoint_dir = f'/root/maaf/checkpoint/{name}'
os.makedirs(checkpoint_dir, exist_ok=True)


# Save args as a JSON file with spaces
args_file_path = os.path.join(checkpoint_dir, 'args.json')
with open(args_file_path, 'w') as args_file:
    json.dump(args_dict, args_file, indent=4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the sequences
directory = Path(__file__).parent.parent 

csv_path = directory / "dataset/annotations.csv"
dataset_path = directory / "dataset"
class_path = directory / "classes_only_actions.json"

with open(class_path, 'r') as file:
    classes = json.load(file)["classes"]


# load the dataset
print("dataset is loading ...")

npz_path = '/root/maaf/processed_datas/processed_actions_co_tracker.npz'
dataset = get_processed_dataset(npz_path)

# Determine the lengths of the splits
train_len = int(0.8 * len(dataset))
val_len = len(dataset) - train_len

# Split the dataset
torch.manual_seed(42)
train_dataset, val_dataset = random_split(dataset, [train_len, val_len])

video_loader_train = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
video_loader_val = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)


# Load the dataset
print(f"Training sequences length: {len(train_dataset)}"
      f"\nValidation sequences length: {len(val_dataset)}")


# Init your model
temporal_model = ViewAwareTransformer(num_features=(10000 - args.num_features), num_views=args.num_views, num_classes=len(classes), num_layers=args.num_layers, nhead=args.num_heads,device=device,dropout=args.dr_rate).to(device)

# Define the optimizer
optimizer = torch.optim.Adam(temporal_model.parameters(), lr=args.lr)

# cosine warmup scheduler
num_train_epochs = args.epochs
num_training_steps = num_train_epochs * len(video_loader_train)
num_warmup_steps = num_training_steps * 0.1  # 10% of training steps as warmup

# Setup the scheduler
scheduler = tune.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
# scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=1)

# Define your loss function
# class_counts = dataset.get_class_counts()
# class_weights = torch.tensor([np.sum(class_counts) / x for x in class_counts], dtype=torch.float32)

# class_weights = class_weights.cuda().to(device)
# loss_function = nn.CrossEntropyLoss(weight=class_weights)
loss_function = nn.CrossEntropyLoss()

# Initialize wandb
if args.use_wandb:
    wandb.init(project="maaf", entity="btrgvn", name=name)

# Define variables to track the lowest validation loss and corresponding epoch
best_f1_val = -float('inf')
best_epoch = -1

print("training starts ...")

# Print model shape
print("Model shape:", temporal_model)
print("Input shape to the model:", video_loader_train.dataset[0][0].shape)

# Training and Validation Loop
num_epochs = args.epochs # Define the number of epochs
selected_views = args.selected_views

for epoch in range(num_epochs):
    # Training Phase
    temporal_model.train()
    train_loss = 0
    train_f1s = []
    train_accs = []
    for iter,(video, labels) in enumerate(video_loader_train):
        video = video[:,:,args.num_features:,:2,:]
        video = video[..., selected_views]
        video = video.to(device)  
        labels = labels.to(device)-1
        if labels == 5 or labels==1:
            continue
        # Forward pass
        start_time = time.time()

        pred_labels = temporal_model(video.to(device))

        forward_pass_time = time.time() - start_time

        
        # Backward and optimize
        loss = loss_function(pred_labels.view(-1,len(classes)), labels.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        train_loss += loss.item()

        # Collect predictions and labels for F1 score and accuracy calculation
        train_pred = torch.argmax(pred_labels.squeeze(), dim=-1).view(-1).cpu().numpy()
        true_label = labels.squeeze().view(-1).cpu().numpy()

        # Calculate F1 score
        train_f1 = f1_score(train_pred, true_label, average='weighted')*100
        train_f1s.append(train_f1)

        # Calculate accuracy
        train_acc = (train_pred == true_label).mean() * 100
        train_accs.append(train_acc)

        print(f'Epoch [{epoch+1}/{num_epochs}], Iteration {iter}, Train Loss: {train_loss/(iter+1):.2f}, Forward pass time: {forward_pass_time:.2f}')


    # Validation Phase
    temporal_model.eval()
    val_loss = 0
    val_f1s = []
    val_accs = []
    with torch.no_grad():
        for iter,(video, labels) in enumerate(video_loader_val):
            video = video[:,:,args.num_features:,:2,:]
            video = video[..., selected_views]
            video = video.to(device)  
            labels = labels.to(device)-1
            if labels == 5 or labels==1:
                continue
            pred_labels = temporal_model(video.to(device))
            loss = loss_function(pred_labels.view(-1,len(classes)), labels.view(-1))
            val_loss += loss.item()

            # Collect predictions and labels for F1 score and accuracy calculation
            val_pred = torch.argmax(pred_labels.squeeze(), dim=-1).view(-1).cpu().numpy()
            true_label = labels.squeeze().view(-1).cpu().numpy()

            # Calculate F1 score
            val_f1 = f1_score(val_pred, true_label, average='weighted')*100
            val_f1s.append(val_f1)

            # Calculate accuracy
            val_acc = (val_pred == true_label).mean() * 100
            val_accs.append(val_acc)

            # scheduler.step(val_loss/(iter+1))

            print(f'Epoch [{epoch+1}/{num_epochs}], Iteration {iter}, Val Loss: {val_loss/(iter+1):.2f}')

    # Save weights based on best f1 loss
    if np.mean(val_f1s) > best_f1_val:


        best_f1_val = np.mean(val_f1s)
        best_epoch = epoch

        checkpoint_path = os.path.join(checkpoint_dir, 'best_model_weights.pth')
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


# Print best epoch and lowest validation loss
print(f'Best Epoch: {best_epoch+1}, Lowest Validation Loss: {best_f1_val}')

