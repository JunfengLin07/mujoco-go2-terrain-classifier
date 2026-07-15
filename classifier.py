import torch
import torch.nn as nn
import torch.utils.data as data
import pandas as pd
import numpy as np


class TerrainDataset(data.Dataset):
    def __init__(self, csv_paths, window_size=50, mode='train'):
        terrain_map = {'hard': 0, 'soft': 1, 'mud': 2, 'sand': 3}
        self.samples = []
        self.labels = []

        for csv_path in csv_paths:
            df = pd.read_csv(csv_path)
            terrain = df['terrain'].iloc[0]
            label = terrain_map[terrain]
            sensors = df.drop(columns=['time', 'terrain']).values

            # split by time — first 80% train, last 20% val
            split = int(len(sensors) * 0.8)
            if mode == 'train':
                sensors = sensors[:split]
            else:
                sensors = sensors[split:]

            for start in range(0, len(sensors) - window_size, window_size // 4):
                window = sensors[start:start + window_size]
                self.samples.append(window)
                self.labels.append(label)

        self.samples = np.array(self.samples, dtype=np.float32)
        self.labels = np.array(self.labels)

        # normalize
        """
        mean = self.samples.mean(axis=(0, 1), keepdims=True)
        std  = self.samples.std(axis=(0, 1), keepdims=True) + 1e-8
        self.samples = (self.samples - mean) / std
        """

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        feature = torch.FloatTensor(self.samples[idx]).T
        label = torch.LongTensor([self.labels[idx]])[0]
        return feature, label

class ManualNorm(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_std', torch.ones(num_features))
        self.momentum = 0.1
    
    def forward(self, x):
        if self.training:
            mean = x.mean(dim=(0, 2), keepdim=True)
            std  = x.std(dim=(0, 2), keepdim=True) + 1e-8
            
            # detach so updates don't become part of computation graph
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.squeeze()
                self.running_std  = (1 - self.momentum) * self.running_std  + self.momentum * std.squeeze()
        else:
            mean = self.running_mean.view(1, -1, 1)
            std  = self.running_std.view(1, -1, 1)
        
        return (x - mean) / std

class classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(23,64,3,stride=1,padding = 1)
        self.conv2 = nn.Conv1d(64,128,3,stride=1,padding = 1)
        self.conv3 = nn.Conv1d(128,256,3,stride=1,padding = 1)

        self.bn1 = ManualNorm(64)
        self.bn2 = ManualNorm(128)
        self.bn3 = ManualNorm(256)

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc1 = nn.Linear(256,4)
        self.relu = nn.ReLU()



    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        """
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        """
        x = self.pool(x)
        x = x.squeeze(-1)
        x = self.fc1(x)

        return x
    



