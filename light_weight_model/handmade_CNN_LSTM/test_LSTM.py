import torch
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
import os
import json
from dataloaderMultiview import MultiVideoDataset
from model_Transformer import ViewAwareTransformer
from sklearn.metrics import f1_score
from co_tracker.cotracker.predictor import CoTrackerPredictor

def load_best_model(model_path, num_features, num_views, num_classes, num_layers, nhead, device):
    model = ViewAwareTransformer(num_features=num_features * num_features, num_views=num_views, num_classes=num_classes, num_layers=num_layers, nhead=nhead, device=device)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    return model

# Define a function for processing a batch of videos
def process_batch_co_tracker(video_batch):
    processed_frames = []
    for view in range(video_batch.shape[2]):
        pred_tracks, pred_visibility = feature_model(video_batch[:,:,view,...], grid_size=10)
        processed_frames.append(pred_tracks)
    return torch.stack(processed_frames, dim=-1)

def main():
    # Load the co-tracker model
    feature_model = CoTrackerPredictor(
        checkpoint=os.path.join(
            '/root/maaf/co_tracker/checkpoints/cotracker2.pth'
        )
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_model = feature_model.to(device)
    directory = Path(__file__).parent

    dataset_path = directory / "dataset"
    csv_path = directory / "dataset/annotations.csv"
    class_path = directory / "classes.json"
    model_path = directory / "checkpoint/best_model_weights.pth"

    # Load class mappings
    with open(class_path, 'r') as file:
        classes = json.load(file)["classes"]

    # Prepare test dataset
    test_seq_names = ['seq_4']  # Or however you determine the test sequences
    video_dataset_test = MultiVideoDataset(test_seq_names, dataset_path, csv_path, class_path, frame_chunk_size=10)
    video_loader_test = DataLoader(video_dataset_test, batch_size=1, shuffle=False, num_workers=0, drop_last=True)

    # Load the best model
    model = load_best_model(model_path, num_features=10, num_views=8, num_classes=len(classes), num_layers=4, nhead=10, device=device)

    # Initialize lists to store individual F1 scores
    f1_scores = []

    # Test loop
    with torch.no_grad():
        for video_chunk, label_chunk in video_loader_test:
            video_chunk = video_chunk.to(device)
            label_chunk = label_chunk.to(device)
            processed_video = process_batch_co_tracker(video_chunk)

            # Get predictions
            pred_labels = model(processed_video)
            predicted_classes = torch.argmax(pred_labels.squeeze(), dim=1)

            # Calculate F1 score for the current sample
            f1 = f1_score(label_chunk.squeeze().cpu().numpy(), predicted_classes.squeeze().cpu().numpy(), average='weighted')
            f1_scores.append(f1)

            # print("Predicted Classes:", predicted_classes)

    # Calculate average F1 score for the entire dataset
    print("Average F1 Score:", np.mean(f1_scores))

# Entry point
if __name__ == "__main__":
    main()
