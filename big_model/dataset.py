# define video VideoDataset
import glob
from PIL import Image
import random
import torch
from torch.utils.data import Dataset
import numpy as np

# set seed for random number
np.random.seed(2024)
random.seed(2024)
torch.manual_seed(2024)

class VideoDataset(Dataset):
    def __init__(self, ids, labels, labels_dict,  transform):
        self.transform = transform
        self.ids = ids
        self.labels = labels
        self.labels_dict = labels_dict

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        path2imgs = glob.glob(self.ids[idx] + "/*.jpg")
        label = self.labels_dict[self.labels[idx]]
        frames = []
        for p2i in path2imgs:
            frame = Image.open(p2i)
            frames.append(frame)

        seed = np.random.randint(1e9)
        frames_tr = []
        for frame in frames:
            random.seed(seed)
            np.random.seed(seed)
            frame = self.transform(frame)
            frames_tr.append(frame)
        if len(frames_tr) > 0:
            frames_tr = torch.stack(frames_tr)
        return frames_tr, label