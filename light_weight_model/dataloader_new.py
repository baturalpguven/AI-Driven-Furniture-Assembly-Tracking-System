import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split

class get_processed_dataset(Dataset):
    
    def __init__(self, npz_path):
        self.data = np.load(npz_path,allow_pickle=True)
        self.data = self.data['processed_actions']

    def __len__(self):
        return len(self.data)

    def get_class_counts(self):

        # Get the specified column
        column = self.data[:, -1]

        # Get the unique classes and their counts
        classes, counts = np.unique(column, return_counts=True)

        # Print the classes and their counts
        for cls, count in zip(classes, counts):
            print(f"Class {cls}: {count} instances")

        # counts = np.insert(counts, 0, 1) 
        return counts

    def __getitem__(self, idx):

        # buffer = []
        # org_data = self.data[idx][0]
        # indices = self.data[idx][0][:,0,2,0]
        # for index in range(len(org_data)):
        #     buffer.append(org_data[index,int(indices[index]):,...])

        data = torch.tensor(self.data[idx][0], dtype=torch.float32)
        label = torch.tensor(self.data[idx][1], dtype=torch.long)
        return data, label
    
class get_processed_dataset_single_view(Dataset):
    
    def __init__(self, npz_path):
        self.data = np.load(npz_path,allow_pickle=True)
        self.data = self.data['processed_actions']

    def __len__(self):
        return len(self.data)

    def get_class_counts(self):

        # Get the specified column
        column = self.data[:, -1]

        # Get the unique classes and their counts
        classes, counts = np.unique(column, return_counts=True)

        # Print the classes and their counts
        for cls, count in zip(classes, counts):
            print(f"Class {cls}: {count} instances")

        # counts = np.insert(counts, 0, 1) 
        return counts

    def __getitem__(self, idx):

        # buffer = []
        # org_data = self.data[idx][0]
        # indices = self.data[idx][0][:,0,2,0]
        # for index in range(len(org_data)):
        #     buffer.append(org_data[index,int(indices[index]):,...])

        data = torch.tensor(self.data[idx][0], dtype=torch.float32)
        label = torch.tensor(self.data[idx][1], dtype=torch.long)

        data = torch.split(data, 1, dim=-1)
        data = torch.stack(data, dim=0) 

        label = label.repeat(data.shape[0],1)

        return data, label



        
# Usagee example
if __name__ == "__main__":

    npz_path = '/root/maaf/processed_datas/processed_actions_co_tracker.npz'
    dataset = get_processed_dataset(npz_path)

    # Determine the lengths of the splits
    train_len = int(0.8 * len(dataset))
    val_len = len(dataset) - train_len

    # Split the dataset
    train_dataset, val_dataset = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=True)

    for data, labels in train_loader:
        print(data.shape, labels.shape)
        break
        # Add your model training code here

