

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import torch.optim as optim
import torch.nn.functional as F




# Loading Data
tran = transforms.Compose([transforms.ToTensor()])
train_set = datasets.MNIST(root='./', train=True, transform=tran, download=True)
test_set = datasets.MNIST(root='./', train=False, transform=tran, download=True)
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
test_loader = DataLoader( test_set, batch_size=32, shuffle=True)



#Define the CNN Model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim = 1)
    

# Define the optimizer and loss function
conv_model = Net()
optimizer = optim.SGD(conv_model.parameters(), lr=0.01,
                      momentum=0.5)

# Train the model

def train_model(num_epoch):
    conv_model.train()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad() # Reset gradients
        output = conv_model(data) # Forward pass
        loss = F.nll_loss(output, target) # Compute loss
        loss.backward() # Backward pass
        optimizer.step() # Update weights
        
        if (batch_idx + 1)% 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                num_epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                100. * (batch_idx + 1) / len(train_loader), loss.item()))
            
def test():
    conv_model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = conv_model(data)
            
            test_loss += F.nll_loss(output, target, size_average=False).item()

            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            
    test_loss /= len(test_loader.dataset)

    print('\nAverage Val Loss: {:.4f}, Val Accuracy: {}/{} ({:.3f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

num_epochs = 50



if __name__ == '__main__':
    for n in range(num_epochs):
        train_model(n)
        test()

    torch.save(conv_model.state_dict(), 'model.pth')