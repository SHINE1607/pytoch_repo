# imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms


# Load data
train_dataset  = datasets.MNIST(root = "dataset/", 
                               train = True, 
                               transform = transforms.ToTensor(),
                               download  = True)


val_dataset  = datasets.MNIST(root = "dataset/",
                             transform = transforms.ToTensor(), 
                             download = True)


# create fully copnnected layer
class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        # first layer output
        x = F.relu(self.fc1(x))
            
        # second layer output
        x = self.fc2(x)

        return x        
            


# initialize the model
class CNN(NN.module):
    def __init__(self, in_channels, num_classes = 10):
        super(NN, self).__init__()
        # adding convilution layer
        self.conv1 = nn.Conv2d(in_channels = in_channels, 
                               out_channels = 8, 
                               kernel_size = (3, 3), 
                               stride = (1, 1), 
                               padding = (1, 1))
        
        # adding pooling layer
        self.pool1 = nn.MaxPool2d(kernel_size = (2, 2), 
                                  stride = (2, 2))
        
        # adding second convolution layer
        self.conv2d(input_channels = 8, 
                    out_channels = 16, 
                    kernel_size = (3, 3), 
                    stride = (1, 1), 
                    padding = (1, 1))
        
        # adding pooling layer
        self.pool2 = nn.MaxPool2d(kernel_size = (2, 2), 
                                  stride = (2, 2))
        
        # adding fully connected layer
        # addng 2 pooling layers make the image half the size
        self.fc1 = nn.Linear(input_size = 16*7*7,
                             num_classes = num_classes)
        
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool1(x) 

        # reshaping the images
        x = x.view(x.shape[0], -1)                
        
        x = self.fc1(x)
        
        
        return x
        


# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Hyperparamters
input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs =  10

# dataloading
train_loader = DataLoader(train_dataset, 
                          batch_size = batch_size, 
                          shuffle = True)
val_loader = DataLoader(val_dataset, 
                          batch_size = batch_size, 
                          shuffle = True)

# initialize  network(Fully connected Neural network)
model = NN(input_size = input_size,
           num_classes = num_classes).to(device)



# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate )



# Train network
for epoch in range(num_epochs):
    num_correct = 0
    accuracy = 0
    num_samples = 0
    print("========== epoh number: {} ========== ".format(epoch))
    for batch_index, (data, targets) in enumerate(train_loader):
        # loading data to the device
        data.to(device = device)
        targets = targets.to(device = device)
        
        # flattening each image
        data = data.view(data.shape[0], -1)
        
        
        # forward 
        scores = model(data)
        loss = criterion(scores, targets)
        
        # backpropogation
        optimzer.zero_grad() # clearning all grads from the previous batch
        loss.backward()
        
        # gradient descent
        optimizer.step()
        
        _, predictions = scores.max(1)
        num_correct += float((predictions == targets).sum())
        num_samples += len(predictions)
    accuracy = float(num_correct)/num_samples
    print("Accuracy of given batch: {:.2f}".format(accuracy))
        
        
        
        
        
#  check accuracy on traininig and test to see how good our model is
def check_accuracy(loader, model):
    
    if loader.dataset.train == True:
        print("Checking accuracy on train data")
    else:
        print("Checking accuracy on test data")
    num_correct = 0
    num_samples = 0
    model.eval()# this turns off the specific features like batchnorm layers and dropout layer during evaluation
    accuracy = 0
    with torch.no_grad():   # we dont have to compute the gradients during evaluation of model
        for batch_index ,(x, y) in enumerate(loader):
            
            scores = model(x)
            _, predictions = scores.max(1)
            
            num_correct = (predcitions == y).sum()
            num_samples = len(y)
            
        accuracy += (accuracy * (batch_index - 1) * x.shape[0] + (num_correct))/(batch_size * x.shape[0])
        print("Accuracy of the batch: {:.2f}".format(num_correct/num_samples))
        print("Total accuracy: {:.2f}".format(accuracy))