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
        mean = self.samples.mean(axis=(0, 1), keepdims=True)
        std  = self.samples.std(axis=(0, 1), keepdims=True) + 1e-8
        self.samples = (self.samples - mean) / std

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        feature = torch.FloatTensor(self.samples[idx]).T
        label = torch.LongTensor([self.labels[idx]])[0]
        return feature, label


class classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(23,64,3,stride=1,padding = 1)
        self.conv2 = nn.Conv1d(64,128,3,stride=1,padding = 1)
        self.conv3 = nn.Conv1d(128,256,3,stride=1,padding = 1)

        #self.bn1 = nn.GroupNorm(8,64)
        #self.bn2 = nn.GroupNorm(8,128)
        #self.bn3 = nn.GroupNorm(8,256)

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc1 = nn.Linear(256,4)
        self.relu = nn.ReLU()

    def forward(self,x):

        """
        x = self.bn1(self.relu(self.conv1(x)))
        x = self.bn2(self.relu(self.conv2(x)))
        x = self.bn3(self.relu(self.conv3(x)))
        """

        x = self.conv1(x)
        #x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        #x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        #x = self.bn3(x)
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





