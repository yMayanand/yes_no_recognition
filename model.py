import torch.nn as nn
from einops import rearrange, reduce, repeat

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2) # 100, 128
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2) # 50, 64
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2) # 25, 32
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=2) # 12, 16

        self.lstm = nn.LSTM(352, 256)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(256, 3)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = x.permute(3, 0, 1, 2)
        x = rearrange(x, 's b c h -> s b (c h)')
        x = self.lstm(x)
        x = self.fc(x[0])
        x = x.log_softmax(2)
        return x