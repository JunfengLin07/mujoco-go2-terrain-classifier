from collections import Counter
import classifier
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import random_split

train_dataset = classifier.TerrainDataset([
    'sensor_log_hard.csv',
    'sensor_log_sand.csv',
    'sensor_log_mud.csv',
    'sensor_log_soft.csv'
], mode='train')

val_dataset = classifier.TerrainDataset([
    'sensor_log_hard.csv',
    'sensor_log_sand.csv',
    'sensor_log_mud.csv',
    'sensor_log_soft.csv'
], mode='val')

trainLoader = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True)
valLoader = DataLoader(val_dataset, batch_size=16, shuffle=False)


net = classifier.classifier()

if __name__ == "__main__":

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=0.001,
        momentum = 0.9,
        weight_decay=1e-4
    )

    train_correct = 0
    train_total = 0
    for epoch in range(10):
        train_correct = 0
        train_total = 0
        for x, labels in trainLoader:
            optimizer.zero_grad()
            output = net(x)
            loss = criterion(output,labels)
            loss.backward()
            optimizer.step()
            predicted = output.argmax(1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)

        print(loss)

        # validation
        net.eval()
        val_correct = 0
        val_total = 1
        with torch.no_grad():
            for x, labels in valLoader:
                output = net(x)
                predicted = output.argmax(1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total

        print(f"Epoch{epoch+1}|Loss:{loss.item():.4f} | train acc{train_acc:.1f} | val acc {val_acc:.1f}")
        
    net.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, labels in valLoader:
            output = net(x)
            predicted = output.argmax(1)
            all_preds.extend(predicted.tolist())
            all_labels.extend(labels.tolist())

    terrain_names = {0: 'hard', 1: 'soft', 2: 'mud', 3: 'sand'}

    print("\nPrediction distribution:")
    print(Counter(all_preds))

    print("\nConfusion matrix:")
    for true in range(4):
        row = [all_preds[i] for i in range(len(all_labels)) if all_labels[i] == true]
        print(f"True {terrain_names[true]}: {Counter(row)}")


        

