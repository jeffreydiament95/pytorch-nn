import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from dataclasses import dataclass


# -----------------------------------------------------------------------------
# constants
PIXEL_RANGE = 255
IMAGE_SIZE = 28*28
DIGIT_CLASSES = 10

# -----------------------------------------------------------------------------
# gloabl variables to make plotting easier
fig, axes = None, None

# -----------------------------------------------------------------------------

@dataclass
class ModelConfig:
    first_hidden_layer: int = None
    second_hidden_layer: int = None

# -----------------------------------------------------------------------------
# MLP model

class MLP(nn.Module):
    """
    takes in an image of size 28x28 and outputs a vector of size 10 predicting the digit in the image along with the loss (if targets are provided)
    """
    def __init__(self):
        super().__init__()
        
        self.fc1 = nn.Linear(IMAGE_SIZE, config.first_hidden_layer)
        self.fc2 = nn.Linear(config.first_hidden_layer, config.second_hidden_layer)
        self.fc3 = nn.Linear(config.second_hidden_layer, DIGIT_CLASSES)

        
    def forward(self, x, targets=None):
        
        x = x.view(-1, IMAGE_SIZE) # batch_size x image_size
        h1 = F.relu(self.fc1(x)) # batch_size x first_hidden_layer
        h2 = F.relu(self.fc2(h1)) # batch_size x second_hidden_layer
        logits = self.fc3(h2) # batch_size x digit_classes
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss


# -----------------------------------------------------------------------------
# helper function for loading and normalizing the MNIST dataset
def load_data():
    # load the train dataset for mean calculation
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())

    # compute mean and std of the train dataset
    mean = train_dataset.data.float().mean() / PIXEL_RANGE
    std = train_dataset.data.float().std() / PIXEL_RANGE

    # define transformation to be applied to the images
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(mean,), std=(std,))
    ])

    # load the train and test datasets with normalization. these normalizations are applied to the images when they are loaded
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)


    return train_dataset, test_dataset

# -----------------------------------------------------------------------------
# helper functions for evaluating and visualizing the model performance
@torch.no_grad()
def visualize_accuracy(model, dataset, num_images=25):
    global fig, axes
    
    n = int(num_images**0.5)
    assert num_images**0.5 == n, "num_images must be a perfect square"
    
    if fig is None and axes is None:
        fig, axes = plt.subplots(n, n, figsize=(10, 10))
        fig.tight_layout()
    
    for ax in axes.flat:
        ax.clear()

    images = []
    labels = []
    predictions = []

    data_loader = DataLoader(dataset, shuffle=True, batch_size=1)

    for i, data in enumerate(data_loader):
        image, label = data
        output, _ = model(image)
        _, predicted = torch.max(output, 1)

        images.append(image)
        labels.append(label)
        predictions.append(predicted)
        if i == num_images:
            break

    images = torch.cat([images[i] for i in range(num_images)])
    labels = torch.cat([labels[i] for i in range(num_images)])
    predictions = torch.cat([predictions[i] for i in range(num_images)])

    images = images.view(-1, 28, 28)

    for ax in axes.flat:
        ax.clear()

    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i], cmap='gray')
        ax.axis('off')
        title = f"Label: {labels[i].item()}\nPrediction: {predictions[i].item()}"
        if labels[i] == predictions[i]:
            ax.set_title(title, color='white', backgroundcolor='green')
        else:
            ax.set_title(title, color='white', backgroundcolor='red')

    plt.show(block=False)
    plt.pause(0.1)

@torch.no_grad()
def evaluate(model, dataset, batch_size=64, max_batches=None):
    data_loader = DataLoader(dataset, shuffle=True, batch_size=batch_size)
    losses = []
    accuracy = []

    for i, data in enumerate(data_loader):
        inputs, targets = data
        logits, loss = model(inputs, targets)
        losses.append(loss.item())
        _, predicted = torch.max(logits, 1)
        accuracy.append((predicted == targets).sum().item() / targets.size(0)) 
        if max_batches is not None and i >= max_batches-1:
            break
    
    mean_loss = torch.tensor(losses).mean().item()
    mean_accuracy = torch.tensor(accuracy).mean().item()
    return mean_accuracy, mean_loss

# -----------------------------------------------------------------------------
if __name__ == '__main__':

    # parse command line args
    parser = argparse.ArgumentParser(description="MNIST")
    # system
    parser.add_argument('--seed', type=int, default=42, help="seed")
    # model
    parser.add_argument('--n-layer1', type=int, default=512, help="number of neurons in MLP layer 1")
    parser.add_argument('--n-layer2', type=int, default=512, help="number of neurons in MLP layer 2")
    # optimization
    parser.add_argument('--batch-size', '-b', type=int, default=64, help="batch size during optimization")
    parser.add_argument('--learning-rate', '-l', type=float, default=1e-1, help="learning rate")
    parser.add_argument('--epochs', '-e', type=float, default=5, help="epochs")
    # other
    parser.add_argument('--print', '-p', action='store_true', default=False, help="enable print loss and accuracy during optimization")
    parser.add_argument('--visualize', '-v', action='store_true', default=False, help="enable visualize model performance during optimization")


    args = parser.parse_args()
    print(vars(args))
    
    # system inits
    torch.manual_seed(args.seed)
    
    # init model
    config = ModelConfig(first_hidden_layer=args.n_layer1, second_hidden_layer=args.n_layer2)

    # load the data and create a train data_loader
    train_dataset, test_dataset = load_data()
    train_data_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)

    # initialize the model and the optimizer
    model = MLP()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    print(f"model #params: {sum(p.numel() for p in model.parameters())}")

    # training loop
    for epoch in range(args.epochs):
        running_loss = []
        for i, data in enumerate(train_data_loader):            
            # get batch data 
            inputs, targets = data
            
            # forward pass
            logits, loss = model(inputs, targets)
            
            # set gradients to zero
            optimizer.zero_grad()
            
            # backward pass
            loss.backward()
            
            # gradient descent
            optimizer.step()
            
            # print loss and accuracy
            running_loss.append(loss.item())
            if i % 100 == 0:
                accuracy, test_loss = evaluate(model, test_dataset, max_batches=1)
                if args.visualize:
                    visualize_accuracy(model, test_dataset)
                if args.print:
                    print(f'epoch {epoch+1}/{args.epochs}, iteration {i+1}: train loss {sum(running_loss) / len(running_loss):0.4f}, test loss {test_loss :0.4f} test accuracy {accuracy:0.2%}')
                # input('Press Enter to continue...')


    accuracy, test_loss = evaluate(model, test_dataset)
    print(f'Performance over all test data: loss {test_loss:0.4f}, accuracy {accuracy:0.2%}')