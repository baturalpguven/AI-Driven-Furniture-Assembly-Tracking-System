from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
import torch

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

def loss_epoch(model, loss_func, dataset_dl, device, sanity_check=False, opt=None):
    running_loss = 0.0
    running_metric = 0.0
    len_data = len(dataset_dl.dataset)
    for xb, yb in tqdm(dataset_dl):
        xb = xb.to(device)
        yb = yb.to(device)
        print(xb.device)
        print(yb.device)
        print(model.device)
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

# This v1 set return corrects, precision, recall, f1

def metrics_batch_v1(output, target):
    pred = output.argmax(dim=1, keepdim=True)  # Predictions
    corrects = pred.eq(target.view_as(pred)).sum().item()  # Correct predictions
    
    # To calculate precision, recall, and F1, we need true positives, false positives, and false negatives
    # Using sklearn to calculate these values (ensure to detach and move tensors to CPU for sklearn compatibility)
    pred = pred.view(-1).cpu()
    target = target.view(-1).cpu()
    precision, recall, f1, _ = precision_recall_fscore_support(target, pred, average='micro')
    
    return corrects, precision, recall, f1

def loss_batch_v1(loss_func, output, target, opt=None):
    loss = loss_func(output, target)
    with torch.no_grad():
        metric_b = metrics_batch_v1(output, target)  # This now returns more metrics
    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()
    return loss.item(), metric_b

def loss_epoch_v1(model, loss_func, dataset_dl, device, sanity_check=False, opt=None):
    running_loss = 0.0
    running_metric = [0.0, 0.0, 0.0, 0.0]  # This will store sums of corrects, precision, recall, and F1
    len_data = len(dataset_dl.dataset)
    for xb, yb in tqdm(dataset_dl):
        xb = xb.to(device)
        yb = yb.to(device)
        output = model(xb)
        loss_b, metrics = loss_batch_v1(loss_func, output, yb, opt)
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

# Custom collate func to incorporate varying length sequences
def collate_fn_lstm(batch):
    """
    Collate function to pad input sequences of images to the same length in a batch,
    while also handling the associated labels.

    Parameters:
    - batch (list of tuples): Each tuple contains a tensor of shape [sequence_length, channels, height, width]
      where 'sequence_length' can vary between samples but other dimensions are fixed, and a label.

    Returns:
    - tuple: A tuple containing two tensors:
        1. Tensor of padded image sequences.
        2. Tensor of labels corresponding to the sequences.
    """
    # Unpack the images and labels from the batch
    imgs_batch, label_batch = zip(*batch)

    # Filter out empty sequences and corresponding labels
    imgs_batch = [imgs for imgs in imgs_batch if len(imgs) > 0]
    label_batch = [torch.tensor(l) for l, imgs in zip(label_batch, imgs_batch) if len(imgs) > 0]

    # Determine the maximum sequence length in this filtered batch
    max_length = max(seq.size(0) for seq in imgs_batch)

    # Pad each sequence to the maximum length
    padded_imgs_batch = []
    for seq in imgs_batch:
        pad_size = max_length - seq.size(0)
        # Pad the sequence along the sequence length dimension (dim=0)
        if pad_size > 0:
            padded_seq = torch.cat([seq, torch.zeros(pad_size, *seq.shape[1:], dtype=seq.dtype, device=seq.device)], dim=0)
        else:
            padded_seq = seq
        padded_imgs_batch.append(padded_seq)

    # Stack all padded sequences and all labels into tensors
    imgs_tensor = torch.stack(padded_imgs_batch)
    labels_tensor = torch.stack(label_batch)

    return imgs_tensor, labels_tensor