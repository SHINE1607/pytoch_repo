import torch
import torchvision
import torch.nn
import torch.optim as optim     # all optimization algorithms 
import torch.nn.functional as F     # all fucntions that dont have any parameters
from torch.utils.data import DataLoader     # for dataset managamemt s
import torchvision.datasets as datasets     # for importing standard datasets import pytorch store
import torchvision.trasnform as transform       # for transformations to apply on datasets


# set device 
device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper parameters 
input_size = 28
num_classes = 28
num_layers = 2  
hidden_size = 256
num_classes = 10 
learning_rate = 0.001
batch_size = 64
num_epochs = 1

# Create fully connected neural network
class RNN(nn.module()):
    def __init__(self, input_size, num_classes, hidden_size, num_classes):
        super(RNN, self).__init__()
        self
#  load data
train_dataset = dataset.MNIST(root = 'dataset/', train = True,
                              transform=transform.ToTensor(), download=True)

test_dataset = dataset.MNIST(root = 'dataset/', train = False,
                              transform=transform.ToTensor(), download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# intialize the network 
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train Network
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        



