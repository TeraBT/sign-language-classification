import torch
import torch.nn as nn


class DummyNetwork(nn.Module):
    def __init__(self, num_classes=36):
        super(DummyNetwork, self).__init__()

        self.fc1 = nn.Linear(in_features=128 * 128, out_features=100)
        self.relu1 = nn.ReLU()
        
        self.fc2 = nn.Linear(in_features=100, out_features=100)
        self.relu2 = nn.ReLU()
        
        self.fc = nn.Linear(in_features=100, out_features=num_classes)

    def forward(self, x):
        x = x.view(-1, 128*128)
        
        x = self.fc1(x)
        x = self.relu1(x)

        x = self.fc2(x)
        x = self.relu2(x)

        x = self.fc(x)

        return x