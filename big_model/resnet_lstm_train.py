import os
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch import optim
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from resnet_lstm_model import ResntLSTM
from sklearn.model_selection import StratifiedShuffleSplit
from utilities import get_vids
from dataset import VideoDataset
from train_utils import get_lr, loss_epoch_v1
import copy
import wandb


def build_train_lstm():
    # HOME variable for cd
    HOME = '/root/maaf_umer'

    # Extract labels from action folders names
    path2ajpgs = os.path.join(HOME, 'Data_Action_V2456')
    all_vids, all_labels, catgs = get_vids(path2ajpgs)
    print(len(all_vids), len(all_labels), len(catgs))

    labels_dict = {}
    ind = 0
    for uc in catgs:
        labels_dict[uc] = ind
        ind += 1

    # Build labels dict
    print('labels_dict: ', labels_dict)

    # Check for duplicate data points
    num_classes = len(labels_dict)
    unique_ids = [id_ for id_, label in zip(all_vids, all_labels) if labels_dict[label] < num_classes]
    unique_labels = [label for id_, label in zip(all_vids, all_labels) if labels_dict[label] < num_classes]
    print('Video numbers:', len(unique_ids), 'Label numbers:', len(unique_labels))

    # Make datasets from extracted images 
    sss = StratifiedShuffleSplit(n_splits=2, test_size=0.1, random_state=0)
    train_indx, test_indx = next(sss.split(unique_ids, unique_labels))

    train_ids = [unique_ids[ind] for ind in train_indx]
    train_labels = [unique_labels[ind] for ind in train_indx]
    print('Training videos:', len(train_ids), 'Training labels:', len(train_labels))

    test_ids = [unique_ids[ind] for ind in test_indx]
    test_labels = [unique_labels[ind] for ind in test_indx]
    print('Test videos:', len(test_ids), 'Test labels:', len(test_labels))

    # Resized images to 224 x 224 for ResNet
    h, w =224, 224
    # ImageNet values
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]

    # Define train dataset
    train_transformer = transforms.Compose([
                transforms.Resize((h, w)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
                ])

    train_ds = VideoDataset(ids=train_ids, labels=train_labels, labels_dict=labels_dict , transform=train_transformer)
    print('Train dataset:', len(train_ds))

    # Define test dataset
    test_transformer = transforms.Compose([
                transforms.Resize((h, w)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
                ])

    test_ds = VideoDataset(ids=test_ids, labels=test_labels, labels_dict=labels_dict, transform=test_transformer)
    print('Test dataset:', len(test_ds))

    batch_size = 1
    # Defining dataloaders
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    # Defining model
    params_model = {
        "num_classes": num_classes,
        "dr_rate": 0.5,
        "pretrained": True,
        "lstm_num_layers": 2,
        "lstm_hidden_size": 128
        }
    model = ResntLSTM(params_model)

    # Path to save weights
    weights_dir = os.path.join(HOME, "assets/models")
    # Make the directory if it doesn't exist
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)

    path2weights = os.path.join(weights_dir, 'weights.pt')
    torch.save(model.state_dict(), path2weights)
    
    def train_val(model, params):
        num_epochs = params["num_epochs"]
        #loss_func = params["loss_func"]
        #opt = params["optimizer"]
        train_dl = params["train_dl"]
        val_dl = params["val_dl"]
        sanity_check = params["sanity_check"]
        #lr_scheduler = params["lr_scheduler"]
        path2weights = params["path2weights"]
        device = params_train["device"]

        # Defining loss function
        class_counts = [435, 198, 197, 68, 455, 312]
        class_weights = torch.tensor([np.sum(class_counts) / x for x in class_counts], dtype=torch.float32)
        class_weights = class_weights.cuda().to(device)
        loss_func = nn.CrossEntropyLoss(reduction="sum", weight=class_weights)

        # Degining optimizer and lr scheduler
        lr = 2.5e-5
        opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        lr_scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=10, verbose=1)

        loss_history = {
            "train": [],
            "val": [],
        }

        metric_history = {
            "train": [],
            "val": [],
        }

        # Transfrer model data to GPU
        model = model.to(device)

        best_model_wts = copy.deepcopy(model.state_dict())
        best_loss = float('inf')

        for epoch in range(num_epochs):
            current_lr = get_lr(opt)
            print('Epoch {}/{}, current lr={}'.format(epoch, num_epochs - 1, current_lr))
            model.train()
            train_loss, train_metric = loss_epoch_v1(model, loss_func, train_dl, device, sanity_check, opt)
            loss_history["train"].append(train_loss)
            metric_history["train"].append(train_metric)
            model.eval()
            with torch.no_grad():
                val_loss, val_metric = loss_epoch_v1(model, loss_func, val_dl, device, sanity_check)
            if val_loss < best_loss:
                best_loss = val_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), path2weights)
                print("Copied best model weights!")

            loss_history["val"].append(val_loss)
            metric_history["val"].append(val_metric)

            lr_scheduler.step(val_loss)
            if current_lr != get_lr(opt):
                print("Loading best model weights!")
                model.load_state_dict(best_model_wts)

            #print("train loss: %.6f, dev loss: %.6f, accuracy: %.2f" % (train_loss, val_loss, 100 * val_metric))
            print(f"train loss: {train_loss}, train accuracy: {train_metric[0]*100}, dev loss: {val_loss}, dev accuracy: {val_metric[0]*100}")
            print("-" * 10)

            # Log metrices to wandb
            wandb.log({"train_loss": train_loss, "train accuracy": train_metric[0]*100, "dev loss": val_loss, "dev accuracy" : val_metric[0]*100})

        model.load_state_dict(best_model_wts)

        return model, loss_history, metric_history

    # Training params
    run_name = 'resnet50_lstm128x2_dr0.5_rgb_w_mistakes_v2456_wl_wd'
    params_train = {
        "num_epochs": 50,
        #"optimizer": opt,
        #"loss_func": loss_func,
        "train_dl": train_dl,
        "val_dl": test_dl,
        "sanity_check": False,            
        #"lr_scheduler": lr_scheduler,
        "path2weights": os.path.join(weights_dir, run_name+'.pt'),
        "device": torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        }

    # Init wandb to log the model run
    wandb.init(project='maaf', entity = 'btrgvn', name = run_name)
    # Start traning
    model, loss_hist, metric_hist = train_val(model, params_train)
 

if __name__ == "__main__":
    build_train_lstm()