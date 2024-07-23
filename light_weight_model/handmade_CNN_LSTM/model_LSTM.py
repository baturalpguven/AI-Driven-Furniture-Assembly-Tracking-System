import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class EfficientChannelAttention(nn.Module):
    def __init__(self, gamma=2, b=1,num_views = 8, device='cuda'):
        super(EfficientChannelAttention, self).__init__()
        self.gamma = gamma
        self.b = b
        self.device = device
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()
        # Determine the kernel size and create conv layer only once
        t = int(abs((math.log(num_views, 2) + self.b) / self.gamma))
        k = t if t % 2 else t + 1
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k//2, bias=False).to(self.device)


    def forward(self, x):
        N, T, F, XY, V = x.size()
        x = x.permute(0, 1, 4, 2, 3)  # (N, T, V, F, XY)

        # Process each time step
        out_list = []
        for i in range(T):
            y = self.avg_pool(x[:, i]).squeeze(-1).squeeze(-1)  # Reduce spatial dimensions and squeeze (N, V, 1, 1) -> (N, V)
            y = self.conv(y.unsqueeze(1)).transpose(1, 2).unsqueeze(-1)  # Apply conv and adjust dimensions
            y = self.sigmoid(y)
            out_list.append(x[:, i] * y)  # Apply attention


        return torch.stack(out_list, dim=1) 

class ViewAwareLSTM(nn.Module):
    def __init__(self, num_features, num_classes, hidden_size=128, num_layers=2, device='cuda'):
        super().__init__()
        self.channel_attention = EfficientChannelAttention(device=device).to(device)
        self.lstm = nn.LSTM(input_size=num_features, hidden_size=hidden_size, num_layers=num_layers, batch_first=True).to(device)
        self.fc_out = nn.Linear(hidden_size, num_classes).to(device)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device

    def forward(self, x, hidden=None):
        N, T, F, XY, V = x.shape

        x = self.channel_attention(x)  # Apply ECA

        x = x.reshape((N,T,V*F*XY))
        if hidden is None:
            hidden = self.init_hidden(N)

        x, new_hidden  = self.lstm(x, hidden)  # Stateful LSTM
        new_hidden = (new_hidden[0].detach(), new_hidden[1].detach())  # Detach to avoid backprop through the entire history

        x = self.fc_out(x)  # Classification from last time step
        
        return x, new_hidden 

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        return (weight.new(self.num_layers, batch_size, self.hidden_size).zero_().to(self.device),
                weight.new(self.num_layers, batch_size, self.hidden_size).zero_().to(self.device))
# Example usage
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input_tensor = torch.rand(10, 5, 100, 2, 8).to(device)  # Example input tensor
    model = ViewAwareLSTM(num_features=100*8*2, hidden_size=256, num_classes=7,num_layers=2, device=device)
    output,hidden = model(input_tensor)
    print("Output shape:", output.shape)
