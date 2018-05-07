from __future__ import print_function
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Urbansound8k MLP')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=1, metavar='N',
                    help='number of epochs to train (default: 1)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many epochs to wait before logging training status')
args = parser.parse_args()

# CUDA parameters
use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device('cuda' if use_cuda else 'cpu')
kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}

# Hyper Parameters
num_epochs = args.epochs
batch_size = args.batch_size
learning_rate = args.lr


class SoundDataset(Dataset):
    '''
    Pytorch Dataset for loading extracted features from Urban sound music
    '''

    def __init__(self, features_file, labels_file, transform=None):
        self.X_train = np.load(features_file)
        self.y_train = np.load(labels_file)
        self.transform = transform

    def __len__(self):
        return len(self.X_train)

    def __getitem__(self, index):
        data = torch.from_numpy(self.X_train[index]).float()
        label = torch.Tensor(self.y_train[index]).long()

        if self.transform:
            data = self.transform(data)

        return data, label


# CNN
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(2, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 12 * 7, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = x.view(-1, 2, 60, 41) # image_# x n_channel x width x height
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 12 * 7)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(x, dim=1)


if __name__ == '__main__':
    dset_train = SoundDataset(features_file='../data/urbansound_train_features_CNN.npy',
                              labels_file='../data/urbansound_train_labels_CNN.npy')
    dset_test = SoundDataset(features_file='../data/urbansound_test_features_CNN.npy',
                             labels_file='../data/urbansound_test_labels_CNN.npy')
    dset_loader = DataLoader(dset_train, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = DataLoader(dset_test, batch_size=batch_size, shuffle=True, **kwargs)

    net = Net().to(device)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-5)

    # Train the Model
    for epoch in range(num_epochs):
        net.train()
        for batch_idx, (data, target) in enumerate(dset_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = net(data)
            loss = criterion(output, torch.max(target, 1)[1])
            loss.backward()
            optimizer.step()

            # print statistics
            # if epoch % (num_epochs // 10) == 0 and batch_idx % 10 == 0:
            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch,
                    batch_idx * len(data),
                    len(dset_loader.dataset),
                    100. * batch_idx / len(dset_loader),
                    loss.data.item()
                ))

    print('Training over')

    # Test model
    correct = 0
    total = 0
    net.eval()

    with torch.no_grad():
        for data, labels in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = net(data)
            predicted = torch.max(outputs, 1)[1]
            total += labels.size(0)
            correct += (predicted == torch.max(labels, 1)[1].to(device)).sum()

    print('Accuracy of the network : %d %%' % (100 * correct / total))
