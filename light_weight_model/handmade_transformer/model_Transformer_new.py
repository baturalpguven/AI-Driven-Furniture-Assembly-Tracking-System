import torch
import torch.nn as nn
import torch.nn.functional as Func
import math

class EfficientChannelAttention(nn.Module):
    def __init__(self, gamma=2, b=1, num_views=8, device='cuda'):
        super(EfficientChannelAttention, self).__init__()
        self.gamma = gamma
        self.b = b
        self.device = device
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()
        t = int(abs((math.log(num_views, 2) + self.b) / self.gamma))
        k = t if t % 2 else t + 1
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k//2, bias=False).to(self.device)

    def forward(self, x):
        N, T, F, XY, V = x.size()
        x = x.permute(0, 1, 4, 2, 3)  # (N, T, V, F, XY)

        out_list = []
        for i in range(T):
            y = self.avg_pool(x[:, i]).squeeze(-1).squeeze(-1)  # (N, V)
            y = self.conv(y.unsqueeze(1)).transpose(1, 2).unsqueeze(-1)  # Apply conv and adjust dimensions
            y = self.sigmoid(y)
            out_list.append(x[:, i] * y)  # Apply attention

        return torch.stack(out_list, dim=1) 


class ViewAwarePositionalEncoding(nn.Module):
    def __init__(self, num_features, num_views):
        super(ViewAwarePositionalEncoding, self).__init__()
        self.num_features = num_features
        self.num_views = num_views
        self.pe = torch.zeros(num_views, num_features)
        position = torch.arange(0, num_views, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, num_features, 2).float() * -(math.log(10000.0) / num_features))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe.unsqueeze(0).unsqueeze(0)  # Add batch and time dimensions

    def forward(self, x):
        N, T, V, F = x.size()
        pe = self.pe.to(x.device)  # Ensure positional encoding is on the same device as x

        if F < self.num_features:
            padding = torch.zeros(N, T, V, self.num_features - F).to(x.device)
            x = torch.cat([x, padding], dim=-1)
        elif F > self.num_features:
            raise ValueError("Input feature dimension is greater than the expected feature dimension")

        pe = pe.expand_as(x)
        x = x + pe
        return x


class DimensionalityReductionCNN(nn.Module):
    def __init__(self, input_channels=1, output_channels=50, dropout=0.5):
        super(DimensionalityReductionCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, (3, 2), stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, (3, 1), stride=1, padding=1)
        self.pool1 = nn.MaxPool2d((2, 1), stride=2)
        self.pool2 = nn.MaxPool2d((2, 1), stride=2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 1 * 1, output_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = Func.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.dropout(x)
        x = Func.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.dropout(x)
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        return x


class ReduceCNN(nn.Module):
    def __init__(self, output_channels, dropout=0.5):
        super(ReduceCNN, self).__init__()
        self.dimensionality_reduction_cnn = DimensionalityReductionCNN(input_channels=1, output_channels=output_channels, dropout=dropout)

    def forward(self, x):
        N, T, V, F, XY = x.shape
        x = x.reshape(N * T * V, 1, F, XY)
        x = self.dimensionality_reduction_cnn(x)
        x = x.view(N, T, V, -1)
        return x


class ViewAwareTransformer(nn.Module):
    def __init__(self, num_features, num_views, num_classes, num_layers=1, nhead=10, dropout=0.5, device='cuda'):
        super(ViewAwareTransformer, self).__init__()
        self.channel_attention = EfficientChannelAttention().to(device)
        self.spatial_conv = ReduceCNN(output_channels=num_features, dropout=dropout).to(device)
        self.pos_encoder = ViewAwarePositionalEncoding(num_features, num_views).to(device)
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=num_features*num_views,
            nhead=nhead,
            dropout=dropout,
            batch_first=True
        ).to(device)
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=num_layers).to(device)
        self.fc_out = nn.Linear(num_features*num_views, num_classes).to(device)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        N, T, F, XY, V = x.shape
        x = self.channel_attention(x)
        x = self.spatial_conv(x)
        x = self.pos_encoder(x)
        N, T, F, V = x.shape
        x = x.reshape(N, T, F*V)
        x = self.transformer_encoder(x)
        x = self.dropout(x[:, -1, :])
        out = self.fc_out(x)
        return out


# Example usage
if __name__ == "__main__":
    input_tensor = torch.rand(1, 50, 500, 2, 4).cuda()
    model = ViewAwareTransformer(num_features=100, num_views=4, num_classes=7)
    output = model(input_tensor)
    print(output.shape)
