# Big Model

Various ablations and models are avaliable. Best results achieved with ResNet50+LSTM. For the ablations you can see the commands. For the base model simply run `resnet_lstm_train.py`.

## Pre-trained
The pretrained high performing model results are under `assets/model` folder

## Run the training

To train the model using the `resnet_lstm_train.py` script, use the following command with the available options:

```bash
python resnet_lstm_train_ablations.py --sel_GPU 0 --run_name resnet_lstm --num_epochs 200 --save_name LSTM --lstm_hidden_size 100 --lstm_num_layers 1 --batch_size 1  --dataset_type rgb --lr 1e-4 --use_wandb
```

```bash
python resnet_lstm_train_ablations.py --sel_GPU 0 --run_name resnet_lstm_co_tracker --num_epochs 200 --save_name LSTM --lstm_hidden_size 100 --lstm_num_layers 1 --batch_size 1  --dataset_type co_tracker --lr 1e-4 --use_wandb
```

```bash
python resnet_lstm_train_better.py --sel_GPU 1 --run_name resnet18_lstm128_2stack_rgb --num_epochs 200 --save_name LSTM --lstm_hidden_size 128 --lstm_num_layers 2 --batch_size 1  --dataset_type rgb --lr 1e-4 --weight_decay 1e-4 --use_wandb
```


```bash
python ViT_lstm_train.py --sel_GPU 0 --run_name ViT_lstm --num_epochs 200 --save_name LSTM --lstm_hidden_size 100 --lstm_num_layers 1 --batch_size 1  --dataset_type rgb --lr 1e-4 --use_wandb
```


## Command Line Arguments

The `resnet_lstm_train.py` script accepts the following command line arguments:

- `--sel_GPU`: Specifies the ID of the GPU to use for training. Default is '0'.
- `--run_name`: Sets the name of the run, which can be useful for tracking different experiments. Default is 'resnet_lstm'.
- `--num_epochs`: Defines the number of training epochs. Default is 200.
- `--save_name`: Name under which the model and its checkpoints will be saved. Default is 'LSTM'.
- `--lstm_hidden_size`: Specifies the size of the hidden layers for the LSTM. Default is 100.
- `--lstm_num_layers`: Sets the number of layers in the LSTM. Default is 1.
- `--batch_size`: Determines the batch size used during training. Default is 1.
- `--dataset_type`: Type of dataset to use, can be 'rgb', 'maks', or 'co-tracker'. Default is 'rgb', and it's important to specify if a different dataset format is used.
- `--use_wandb`: if present in the CLI use wandb
- `--lr`: learning rate
-`weight_decay`: L2 reg weight
