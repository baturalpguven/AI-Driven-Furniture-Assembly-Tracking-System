import argparse
import json

def build_train_lstm(args):
    import os
    import numpy as np
    import torch
    import wandb
    import datetime


    os.environ["CUDA_VISIBLE_DEVICES"] = ".".join(args.sel_GPU)
    print('Number of GPUs available:', torch.cuda.device_count())

    from transformers import ViTModel, ViTConfig, ViTImageProcessorFast
    # HOME variable for cd
    HOME = '/root/maaf_umer'

    from sklearn.model_selection import StratifiedShuffleSplit

    # Extract labels from action folders names
    from utilities import get_vids

    if args.dataset_type == 'rgb':
        dataset_folder = 'Data_Action'
    elif args.dataset_type == 'mask':
        dataset_folder = 'Data_Segmented_Action/Annotations'
    else:
        dataset_folder = 'Data_Segmented_Tracked_Action'
    path2ajpgs = os.path.join(HOME, dataset_folder)
    all_vids, all_labels, catgs = get_vids(path2ajpgs)
    print(len(all_vids), len(all_labels), len(catgs))

    labels_dict = {}
    ind = 0
    for uc in catgs:
        labels_dict[uc] = ind
        ind += 1

    num_classes = 6
    unique_ids = [id_ for id_, label in zip(all_vids, all_labels) if labels_dict[label] < num_classes]
    unique_labels = [label for id_, label in zip(all_vids, all_labels) if labels_dict[label] < num_classes]

    # Make datasets from extracted images 
    sss = StratifiedShuffleSplit(n_splits=2, test_size=0.1, random_state=0)
    train_indx, test_indx = next(sss.split(unique_ids, unique_labels))

    train_ids = [unique_ids[ind] for ind in train_indx]
    train_labels = [unique_labels[ind] for ind in train_indx]

    test_ids = [unique_ids[ind] for ind in test_indx]
    test_labels = [unique_labels[ind] for ind in test_indx]

    from maaf.big_model.dataset_ablations import VideoDataset,MaskDataset,Co_trackerDataset
    import torchvision.transforms as transforms

    # Defining train dataset
    train_transformer = transforms.Compose([
                    transforms.ToTensor(),
                ])
    if args.dataset_type == 'rgb':
        
        train_ds = VideoDataset(ids=train_ids, labels=train_labels, labels_dict= labels_dict , transform=train_transformer)
    elif args.dataset_type == 'mask':

        # Defining train dataset
        train_transformer = transforms.Compose([
                    transforms.ToTensor(),
                    ])
        train_ds = MaskDataset(ids=train_ids, labels=train_labels, labels_dict= labels_dict , transform=train_transformer)
    else:
        # Defining train dataset
        train_transformer = transforms.Compose([
                    transforms.ToTensor(),
                    ])
        train_ds = Co_trackerDataset(ids=train_ids, labels=train_labels, labels_dict= labels_dict , transform=train_transformer)


    # Defining test dataset
    test_transformer = transforms.Compose([
                transforms.ToTensor(),
                ])
        
    if args.dataset_type == 'rgb':
    
        test_ds = VideoDataset(ids=test_ids, labels=test_labels, labels_dict= labels_dict, transform=test_transformer)
    elif args.dataset_type == 'mask':
        
        test_ds = MaskDataset(ids=test_ids, labels=test_labels, labels_dict= labels_dict, transform=test_transformer)
    else:
        test_ds = Co_trackerDataset(ids=test_ids, labels=test_labels, labels_dict= labels_dict, transform=test_transformer)

    from torch.utils.data import DataLoader
    batch_size = args.batch_size
    # Defining dataloaders
    train_dl = DataLoader(train_ds, batch_size=batch_size,
                            shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size,
                            shuffle=False)

    # Defining LSTM model

    from torch import nn
    from torchvision import models

    # We replace the ResNet fc layer with Identity layer to use ResNet as a feature extractor
    class Identity(nn.Module):
        def __init__(self):
            super(Identity, self).__init__()

        def forward(self, x):
            return x


    class ViTLSTM(nn.Module):
        def __init__(self, params_model):
            super(ViTLSTM, self).__init__()

            num_classes = params_model["num_classes"]
            dr_rate = params_model["dr_rate"]
            pretrained = params_model["pretrained"]
            lstm_hidden_size = params_model["lstm_hidden_size"]
            lstm_num_layers = params_model["lstm_num_layers"]

            # Initialize Vision Transformer
            if pretrained:
                # self.baseModel = ViTModel.from_pretrained('google/vit-large-patch16-224')
                self.baseModel = ViTModel.from_pretrained('google/vit-base-patch16-224')
                # self.baseModel = ViTModel.from_pretrained('google/vit-tiny-patch16-224')

                # Use the fast image processor
                # self.image_processor = ViTImageProcessorFast.from_pretrained('google/vit-large-patch16-224', use_fast=True)
                self.image_processor = ViTImageProcessorFast.from_pretrained('google/vit-base-patch16-224', use_fast=True)

            else:
                config = ViTConfig()
                self.baseModel = ViTModel(config)
                # If using a custom processor or non-pretrained model, initialize as needed
                self.image_processor = None

            self.baseModel.config.hidden_dropout_prob = 0.1

            # Prepare LSTM layer
            num_features = self.baseModel.config.hidden_size
            self.lstm = nn.LSTM(num_features, lstm_hidden_size, lstm_num_layers, batch_first=True)
            self.fc = nn.Linear(lstm_hidden_size, num_classes)

        def forward(self, x):
            batch_size, seq_length, C, H, W = x.shape
            device = x.device

            # Process each frame through ViT
            hidden_states = []
            for t in range(seq_length):
                frame = x[:, t, :, :, :]
                inputs = self.image_processor(images=frame, return_tensors='pt').to(device)
                outputs = self.baseModel(**inputs)
                cls_embeddings = outputs.last_hidden_state[:, 0, :]  # Extract [CLS] token's embeddings
                hidden_states.append(cls_embeddings)

            hidden_states = torch.stack(hidden_states, dim=1)  # Shape: [batch_size, seq_length, feature_size]
            lstm_out, _ = self.lstm(hidden_states)
            out = self.fc(lstm_out[:, -1, :])  # Take the output of the last LSTM layer
            return out
            
    from torch import optim
    from torch.optim.lr_scheduler import ReduceLROnPlateau

    # Defining model
    params_model = {
        "num_classes": num_classes,
        "dr_rate": 0.1,
        "pretrained": True,
        "lstm_num_layers": args.lstm_num_layers,
        "lstm_hidden_size": args.lstm_hidden_size}
    model = ViTLSTM(params_model)

    # Path to save weights
    weights_dir = os.path.join(HOME, "assets/models")
    # Make the directory if it doesn't exist
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)

    path2weights = os.path.join(weights_dir, 'weights.pt')
    torch.save(model.state_dict(), path2weights)

    # Transfrer model data to GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Defining loss function
    loss_func = nn.CrossEntropyLoss(reduction="sum")

    # Degining optimizer and lr scheduler
    lr = args.lr
    opt = optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5, verbose=1)

    import copy
    from tqdm import tqdm

    def get_lr(opt):
        for param_group in opt.param_groups:
            return param_group['lr']

    def metrics_batch(output, target):
        pred = output.argmax(dim=1, keepdim=True)
        corrects = pred.eq(target.view_as(pred)).sum().item()
        return corrects

    def loss_batch(loss_func, output, target, opt=None):
        loss = loss_func(output, target)
        with torch.no_grad():
            metric_b = metrics_batch(output, target)
        if opt is not None:
            opt.zero_grad()
            loss.backward()
            opt.step()
        return loss.item(), metric_b

    def loss_epoch(model, loss_func, dataset_dl, sanity_check=False, opt=None):
        running_loss = 0.0
        running_metric = 0.0
        len_data = len(dataset_dl.dataset)
        for xb, yb in tqdm(dataset_dl):
            xb = xb.to(device)
            yb = yb.to(device)
            output = model(xb)
            loss_b, metric_b = loss_batch(loss_func, output, yb, opt)
            running_loss += loss_b

            if metric_b is not None:
                running_metric += metric_b
            if sanity_check is True:
                break
        loss = running_loss / float(len_data)
        metric = running_metric / float(len_data)
        return loss, metric

    from sklearn.metrics import precision_recall_fscore_support

    def metrics_batch(output, target):
        pred = output.argmax(dim=1, keepdim=True)  # Predictions
        corrects = pred.eq(target.view_as(pred)).sum().item()  # Correct predictions
        
        # To calculate precision, recall, and F1, we need true positives, false positives, and false negatives
        # Using sklearn to calculate these values (ensure to detach and move tensors to CPU for sklearn compatibility)
        pred = pred.view(-1).cpu()
        target = target.view(-1).cpu()
        precision, recall, f1, _ = precision_recall_fscore_support(target, pred, average='micro')
        
        return corrects, precision, recall, f1

    def loss_batch(loss_func, output, target, opt=None):
        loss = loss_func(output, target)
        with torch.no_grad():
            metric_b = metrics_batch(output, target)  # This now returns more metrics
        if opt is not None:
            opt.zero_grad()
            loss.backward()
            opt.step()
        return loss.item(), metric_b

    def loss_epoch(model, loss_func, dataset_dl, sanity_check=False, opt=None):
        running_loss = 0.0
        running_metric = [0.0, 0.0, 0.0, 0.0]  # This will store sums of corrects, precision, recall, and F1
        len_data = len(dataset_dl.dataset)
        for xb, yb in tqdm(dataset_dl):
            xb = xb.to(device)
            yb = yb.to(device)
            output = model(xb)
            loss_b, metrics = loss_batch(loss_func, output, yb, opt)
            running_loss += loss_b

            # Update metrics
            for i in range(4):
                running_metric[i] += metrics[i]

            if sanity_check:
                break

        # Compute average loss and metric
        loss = running_loss / float(len_data)
        # Calculating the average of metrics across all batches
        average_metrics = [m / float(len_data) for m in running_metric]

        return loss, average_metrics  # Returns loss, and a list containing average corrects, precision, recall, and F1

    def train_val(model, params):

        if params["use_wandb"]:
            # save the wandb configuration
            wandb.init(project="maaf",entity="btrgvn",name=params_train["run_name"])

        
        num_epochs = params["num_epochs"]
        loss_func = params["loss_func"]
        opt = params["optimizer"]
        train_dl = params["train_dl"]
        val_dl = params["val_dl"]
        sanity_check = params["sanity_check"]
        lr_scheduler = params["lr_scheduler"]
        path2weights = params["path2weights"]
        weight_name = params["save_name"]

        params_dict = {
                'num_epochs': params['num_epochs'],
                'run_name': params['run_name'],
                'lstm_hidden_size': params['lstm_hidden_size'],
                'lstm_num_layers': params['lstm_num_layers'],
                'use_wandb': params['use_wandb']
            }
        
        # save the params inside the model weight directory
        os.makedirs(path2weights, exist_ok=True)
        with open(os.path.join(path2weights, 'params.json'), 'w') as f:

            json.dump(params_dict, f, indent=4)


        loss_history = {
            "train": [],
            "val": [],
        }

        metric_history = {
            "train": [],
            "val": [],
        }

        best_model_wts = copy.deepcopy(model.state_dict())
        best_loss = float('inf')

        for epoch in range(num_epochs):
            current_lr = get_lr(opt)
            print('Epoch {}/{}, current lr={}'.format(epoch, num_epochs - 1, current_lr))
            model.train()
            train_loss, train_metric = loss_epoch(model, loss_func, train_dl, sanity_check, opt)
            loss_history["train"].append(train_loss)
            metric_history["train"].append(train_metric)
            model.eval()
            with torch.no_grad():
                val_loss, val_metric = loss_epoch(model, loss_func, val_dl, sanity_check)
            if val_loss < best_loss:
                best_loss = val_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), weight_name)
                print("Copied best model weights!")

            loss_history["val"].append(val_loss)
            metric_history["val"].append(val_metric)

            lr_scheduler.step(val_loss)
            if current_lr != get_lr(opt):
                print("Loading best model weights!")
                model.load_state_dict(best_model_wts)

            #print("train loss: %.6f, dev loss: %.6f, accuracy: %.2f" % (train_loss, val_loss, 100 * val_metric))
            print(f"train loss: {train_loss}, dev loss: {val_loss}, accuracy: {val_metric[0]*100}, precision: {val_metric[1]}, recall: {val_metric[2]}, F1-score: {val_metric[3]}")
            if params["use_wandb"]:
                wandb.log({"Epoch":epoch,"train_loss": train_loss, "val_loss": val_loss, "accuracy": val_metric[0], "precision": val_metric[1], "recall": val_metric[2], "F1-score": val_metric[3]})
            print("-" * 10)
        model.load_state_dict(best_model_wts)

        return model, loss_history, metric_history

    # Training params

    save_dir = weights_dir +'/' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    params_train = {
        "num_epochs": args.num_epochs,
        "optimizer": opt,
        "loss_func": loss_func,
        "train_dl": train_dl,
        "val_dl": test_dl,
        "sanity_check": False,            
        "lr_scheduler": lr_scheduler,
        "path2weights": save_dir,
        "save_name": f'{args.save_name}.pt',
        "run_name": args.run_name,
        "lstm_hidden_size": args.lstm_hidden_size,
        "lstm_num_layers": args.lstm_num_layers,
        "use_wandb": args.use_wandb
        }

    # Start traning
    model, loss_hist, metric_hist = train_val(model, params_train)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--sel_GPU', type=str, default='0')
    parser.add_argument('--run_name', type=str, default='resnet_lstm')
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--save_name', type=str, default='LSTM')
    parser.add_argument('--lstm_hidden_size', type=int, default=100)
    parser.add_argument('--lstm_num_layers', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--use_wandb', default=False, action='store_true')
    parser.add_argument('--dataset_type', type=str, default='rgb',help='rgb or maks or co_tracker')
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()

    build_train_lstm(args)