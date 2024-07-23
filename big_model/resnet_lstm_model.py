# Defining LSTM model
from torch import nn
from torchvision import models
from torchvision.models import ResNet50_Weights, ResNet152_Weights

# We replace the ResNet fc layer with Identity layer to use ResNet as a feature extractor
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class ResntLSTM(nn.Module):
    def __init__(self, params_model):
        super(ResntLSTM, self).__init__()
        num_classes = params_model["num_classes"]
        dr_rate = params_model["dr_rate"]
        pretrained = params_model["pretrained"]
        lstm_hidden_size = params_model["lstm_hidden_size"]
        lstm_num_layers = params_model["lstm_num_layers"]

        baseModel = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        #baseModel = models.resnet152(weights=ResNet152_Weights.DEFAULT)
        num_features = baseModel.fc.in_features
        baseModel.fc = Identity()

        self.baseModel = baseModel
        self.dropout = nn.Dropout(dr_rate)
        # LSTM cell
        self.lstm = nn.LSTM(num_features, lstm_hidden_size, lstm_num_layers, dropout=dr_rate)
        # FC layer for final output
        self.fc = nn.Linear(lstm_hidden_size, num_classes)

        # Freeze the ResNet layers
        #for param in baseModel.parameters():
        #    param.requires_grad = False

    def forward(self, x):
        b_z, ts, c, h, w = x.shape
        i = 0
        y = self.baseModel((x[:, i]))

        output, (hn, cn) = self.lstm(y.unsqueeze(1))
        for i in range(1, ts):
            y = self.baseModel((x[:, i]))
            out, (hn, cn) = self.lstm(y.unsqueeze(1), (hn, cn))

        out = self.dropout(out[:, -1])
        out = self.fc(out)
        return out