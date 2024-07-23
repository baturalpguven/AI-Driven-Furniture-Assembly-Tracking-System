# Ligth Weight Model

Action classification based on masked co-tracker points with DL-based methods. Aims to take advantage of a multi-view setup.

This model uses efficient channel attention (ECA) on multi-view input followed by a CNN that aims to reduce the dimensionality, and lastly learns temporal relations with a view-aware transformer.


## Sample Input

Input to the view-aware models are the right-most keypoints for each view.

![deva_co_tracker](/imgs/Placeholder.png)

## Pre-trained
The pre-trained high-performing model results with the ablations are under the `checkpoints` folder.

## Run the training

### Transformer

To start the training for transformer pass the following command 
```bash
python handmade_transformer/train_Transformer.py --num_views 4 --num_heads 8 --num_layers 2 --num_features 9500 --sel_GPU 1 --epochs 200 --lr 1e-4 --dr_rate 0.5 --selected_views 0 1 3 6 --use_wandb True --name transfomer_action_onlyViewAwareTransformerDifferentViews --batch_size 1
```


#### Training Parameters

- `--num_views`: This argument specifies the number of multiviews. It determines how many different views or perspectives are considered in the model.

- `--num_heads`: This argument sets the number of heads in the transformer. In transformer models, attention is divided into multiple heads, allowing the model to focus on different parts of the input simultaneously.

- `--num_layers`: This argument determines the number of encoder layers in the transformer. The encoder layers process the input data in a hierarchical manner, extracting features at different levels of abstraction.

- `--num_features`: This argument specifies the number of track points for co-tracker.  The co-tracker is a component that tracks multiple objects or features across frames in a video or sequence of data.

- `--sel_GPU`: This argument allows you to select the GPU (Graphics Processing Unit) on which the model will run.  If you have multiple GPUs available, you can use this argument to specify which one to use for running the model.

- `--epochs`: This argument determines the number of epochs (complete passes through the training data) for which the model will be trained.  Increasing the number of epochs can potentially improve the model's performance, but it also increases the training time.

- `--use_wandb`: This argument determines whether to save the results to wandb (Weights & Biases), a tool for experiment tracking and visualization. By setting it to True, the results of the model training process can be logged and analyzed using wandb.

- `--name`: This argument specifies the name of the experiment. The experiment name can be customized to provide a meaningful description of the specific configuration or purpose of the model.

- `--selected_views`:select desired views

### LSTM

To start the training for transformer pass the following command 
```bash
python handmade_CNN_LSTM/train_LSTM.py --hidden_size 512 --num_layers 2 --grid_size 20 --sel_GPU 1 --epochs 100 --early_stopping_threshold 0.001 --use_wandb True --name co_tracker+lstm --frames 5
```
Note: Use tmux to prevent interruptions in the training

#### Training Parameters


- `--hidden_size`: This argument determines the number of heads in the transformer model. The number of heads affects the model's capacity to capture complex patterns and relationships in the data.

- `--num_layers`: This argument defines the number of encoder layers in the transformer model. The number of layers affects the depth and complexity of the model, allowing it to learn hierarchical representations of the input data.

- `--num_features`: This argument specifies the number of track points for a co-tracker.The track points represent specific features or landmarks in the input data that the model focuses on during tracking.

- `--sel_GPU`: This argument allows the user to select the GPU (Graphics Processing Unit) on which the model will run. By specifying a different GPU number, the model can utilize the computational power of a specific GPU device.

- `--epochs`: This argument determines the number of epochs (complete passes through the training data) for which the model will be trained.  Increasing the number of epochs can potentially improve the model's performance, but it also increases the training time.

- `--early_stopping_threshold`: This argument sets the threshold for early stopping during training. Early stopping is a technique used to prevent overfitting by stopping the training process if the model's performance on a validation set stops improving significantly.

- `--use_wandb`: This argument determines whether to save the results to wandb (Weights & Biases), a tool for experiment tracking and visualization. By setting it to True, the results of the model training process can be logged and analyzed using wandb.

- `--name`: This argument specifies the name of the experiment. The experiment name can be customized to provide a meaningful description of the specific configuration or purpose of the model.

- `--frames`: This argument determines the number of frames to consider for the model. Frames refer to consecutive snapshots or instances of the input data that the model processes.

- `--stride`: This argument sets the stride for the model. Stride refers to the step size or the number of elements to skip between consecutive frames.


