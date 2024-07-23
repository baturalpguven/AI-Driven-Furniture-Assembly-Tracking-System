# define video VideoDataset
import glob
from PIL import Image
import random
import torch
from torch.utils.data import Dataset
import numpy as np
import cv2

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

class MaskDataset(Dataset):
    def __init__(self, ids, labels, labels_dict,  transform):
        self.transform = transform
        self.ids = ids
        self.labels = labels
        self.labels_dict = labels_dict

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        path2imgs = glob.glob(self.ids[idx] + "/*.npz")
        label = self.labels_dict[self.labels[idx]]
        frames = []
        for p2i in path2imgs:
            frame = np.load(p2i)
            frame = frame['arr_0']
            frame = np.repeat(frame['arr_0'][:, :, np.newaxis], 3, axis=2)
            frame = Image.fromarray(frame.astype(np.uint8))
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

class Co_trackerDataset(Dataset):
    def __init__(self, ids, labels, labels_dict,  transform):
        self.transform = transform
        self.ids = ids
        self.labels = labels
        self.labels_dict = labels_dict

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        path2imgs = glob.glob(self.ids[idx] + "/*.npz")
        label = self.labels_dict[self.labels[idx]]
        frames = []
        for p2i in path2imgs:
            frame = np.load(p2i)
            frame = frame['arr_0'][0,0]

            # Extract the tracking points from the frame
            # Org Image dimensions
            height = 480
            width = 640
            # Create a black background image of specified dimensions
            background = np.zeros((height, width))
            # Overlay coordinates onto the background
            for point in frame:
                y, x = point  # Extract coordinates
                if 0 <= x < width and 0 <= y < height:
                    background[int(x), int(y)] = 1  # Set pixel value to 1
                    
            background = np.repeat(background[:, :, np.newaxis], 3, axis=2)
            frame = Image.fromarray(background.astype(np.uint8))
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
    

class Co_trackerDataset_overlay(Dataset):
    def __init__(self, ids, labels, labels_dict,  transform):
        self.transform = transform
        self.ids = ids
        self.labels = labels
        self.labels_dict = labels_dict

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        path2imgs = glob.glob(self.ids[idx] + "/*.npz")
        label = self.labels_dict[self.labels[idx]]
        frames = []
        for p2i in path2imgs:
            org_image = cv2.imread(p2i.replace('.npz', '.jpg'))
            org_image = cv2.cvtColor(org_image, cv2.COLOR_BGR2RGB)

            coordinates = np.load(p2i)
            coordinates = coordinates['arr_0'][0,0]

            width = org_image.shape[0]
            height = org_image.shape[1]
            # Define the radius of the points (circles)
            radius = 5

            # Define the color (white) and thickness (filled circle)
            color = (0, 255, 0)
            thickness = -1  # -1 means the circle will be filled

            # Overlay coordinates onto the background
            for point in coordinates:
                y, x = point  # Extract coordinates
                if 0 <= x < width and 0 <= y < height:
                    # Draw a filled circle at each point
                    cv2.circle(org_image, (int(y), int(x)), radius, color, thickness)
                    # print('here')
                    
            org_image = Image.fromarray(org_image.astype(np.uint8))
            frames.append(org_image)

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
    
# if __name__ == "__main__":
    