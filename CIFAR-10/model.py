import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# import matplotlib.pyplot as plt
# import numpy as np

# hyperparameters
batch_size = 4
lr = 1e-3
momentum = 0.9
epochs = 2

# def imshow(img):
#     img = img / 2 + 0.5     # unnormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # 3 input channel2, 6 output channels, 5x5 square convolution
        # 32x32 -> 6x 28x28
        self.conv1 = nn.Conv2d(3, 6, 5) 
        
        # max pooling over (2,2) window
        # 6x 28x28 -> 6x 14x14
        self.pool = nn.MaxPool2d(2, 2)
        
        # 6 input channel, 10 output channels, 5x5 square convolution
        # 6x 14x14 -> 16x 10x10
        self.conv2 = nn.Conv2d(6, 16, 5)     
        
        # fully connected
        # 16x5x5 -> 120
        self.fc1 = nn.Linear(16*5*5, 120)
        
        # fully connected
        # 120 -> 84
        self.fc2 = nn.Linear(120, 84)
        
        # fully connected
        # 84 -> 10
        self.fc3 = nn.Linear(84, 10)
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        
        x = torch.flatten(x, start_dim=1) # flatten everything except batch dimension 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# The guard ensures that the code is only executed when the script is run directly, and not when it's imported as a module.
if __name__ == '__main__':

    # converts input image to tensor and then scales values from [0, 1] -> [-1, 1]
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(net.parameters(), lr=lr)

    for epoch in range(epochs):
        running_loss = 0 
        
        for i, data in enumerate(trainloader):
            # get inputs and targets
            inputs, labels = data
            
            # forward pass
            outputs = net(inputs)
            
            # calculate loss
            loss = criterion(outputs, labels)
            
            # zero gradients
            optimizer.zero_grad()
            
            # backward pass
            loss.backward()
            
            # update
            optimizer.step()
            
            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
            
    print("finished training!")

    PATH = './cifar_net.pth'
    torch.save(net.state_dict(), PATH)

    dataiter = iter(testloader)
    # images, labels = next(dataiter)

    # # print images
    # imshow(torchvision.utils.make_grid(images))
    # print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

    # outputs = net(images)
    # _, predicted = torch.max(outputs, 1)

    # print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}' for j in range(batch_size)))

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')