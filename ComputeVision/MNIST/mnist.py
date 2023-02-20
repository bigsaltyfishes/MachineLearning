#
# Copyright (c) 2022 bigsaltyfishes 
# Email: 32119500038@e.gzhu.edu.cn
# Create date: 2022-11-28 18:18 UTC+8
#
import os
import time
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


# Before we start training, we need to define our network
# Define our neural network
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.net_pone = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),  # Layer 1
            nn.MaxPool2d(2),  # Layer 2
            nn.ReLU(),  # Layer 3
            nn.Conv2d(10, 20, kernel_size=5),  # Layer 4
            nn.Dropout2d(),  # Layer 5
            nn.MaxPool2d(2),  # Layer 6
            nn.ReLU()  # Layer 7
        )
        self.net_ptwo = nn.Sequential(
            nn.Linear(320, 50),  # Layer 8
            nn.ReLU(),  # Layer 9
            nn.Dropout(),  # Layer 10
            nn.Linear(50, 10),  # Layer 11
        )
        # Use CUDA if available
        # Or use DirectML instead
        if torch.cuda.is_available():
            print("[INFO] CUDA detected, use CUDA for training")
            self.net_pone = self.net_pone.cuda()
            self.net_ptwo = self.net_ptwo.cuda()
        else:
            print("[INFO] CUDA not available")
            try:
                import torch_directml
            except ModuleNotFoundError:
                print("[INFO] DirectML not available")
                print("[WARN] Use CPU for training")
            else:
                self.dml = torch_directml.device()
                if not self.dml == None:
                    print("[INFO] DirectML detected, use DirectML for training")
                    self.net_pone = self.net_pone.to(self.dml)
                    self.net_ptwo = self.net_ptwo.to(self.dml)
                else:
                    print("[INFO] DirectML not available")
                    print("[WARN] Use CPU for training")

    def forward(self, x):
        x = self.net_pone(x)
        x = x.view(-1, 320)
        x = self.net_ptwo(x)
        return nn.functional.log_softmax(x, dim=1)


# Before we start, we need to set hyper parameters
hyper_params = {
    'n_epoch': 64,
    'batch_size': 64,
    'batch_size_test': 1000,
    'lr': 0.01,
    'momentum': 0.5,
    'random_seed': None
}

# Okay, let's load our datasets
# Load training datasets
datasets = torchvision.datasets.MNIST(
    './data',
    train=True,
    download=True,
    transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])
)

# Load test datasets
datasets_test = torchvision.datasets.MNIST(
    './data',
    train=False,
    download=True,
    transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])
)

# Preprocess data with DataLoader
train_loader = torch.utils.data.DataLoader(
    datasets,
    batch_size=hyper_params['batch_size'],
    shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    datasets_test,
    batch_size=hyper_params['batch_size_test'],
    shuffle=True
)

# Everything seems to be ok now, let's prepare for training
# First Initialize Network and Optimizer
model = MyModel()
optimizer = torch.optim.SGD(
    model.parameters(),
    hyper_params['lr'],
    hyper_params['momentum']
)

# If there are already a model, we should continue training 
# instead of training from zero
is_incremental = False
try:
    f=open('./output/model.ckpt', 'r')
    f.close()
except FileNotFoundError:
    print("[INFO] Training mode: Initial training")
    if not os.path.exists('./output'):
        os.makedirs('./output')

    # Generate new random seed
    hyper_params['random_seed'] = time.time_ns()
    print(f"[INFO] Generated new random seed: {hyper_params['random_seed']}")
    with open('./output/seed.txt', 'w') as f:
        f.write(str(hyper_params['random_seed']))
except PermissionError:
    print ("[ERROR] Failed to read model: Permission denied")
    exit(-1)
else:
    print("[INFO] Detect exist model, continue training")
    try:
        with open('./output/seed.txt', 'r') as f:
            hyper_params['random_seed'] = int(f.read())
    except (FileNotFoundError, PermissionError):
        print('[ERROR] Failed to read random seed')
        exit(-1)
        
    model.load_state_dict(torch.load('./output/model.ckpt'))
    optimizer.load_state_dict(torch.load('./output/optimizer.ckpt'))

# Then initialize Random Pool
torch.manual_seed(hyper_params['random_seed'])

# We need some variable to record loss and tracking training progress
# Simply initialize a dict to do that
tracker = {
    'loss': {
        'train': [],
        'test': []
    },
    'counter': {
        'train': [],
        'test': [i * len(train_loader.dataset) for i in range(hyper_params['n_epoch'] + 1)]
    }
}


# Function for training the model
def train(epoch):
    # Get into training mode
    model.train()

    # Training Loop
    for batch_idx, (data, label) in enumerate(train_loader):
        # Set Gradient to Zero
        optimizer.zero_grad()

        # Use cuda if available
        if torch.cuda.is_available():
            data, label = data.to('cuda'), label.to('cuda')
        else:
            data, label = data.to(model.dml), label.to(model.dml)

        # Predict data
        pred = model(data)

        # Calculate loss
        loss = nn.functional.nll_loss(pred, label)

        # Okay, let's get backward, optimizer will automatically 
        # calculate the gradients we need
        loss.backward()

        # Update parameters
        optimizer.step()

        # Track with the progress (Track once after training 10 batches)
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{}] ({:.2f}%)\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx * len(data) / len(train_loader.dataset), loss.item()
            ))
            tracker['loss']['train'].append(loss.item())
            tracker['counter']['train'].append((batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset)))

            # Don't forget to save the model
            torch.save(model.state_dict(), './output/model.ckpt')
            torch.save(optimizer.state_dict(), './output/optimizer.ckpt')


# Function for Test
def test():
    # Get into evaluation mode
    model.eval()

    # Variable for storing total loss
    total_loss = 0
    correct = 0

    # Test Loop
    with torch.no_grad():
        for data, label in test_loader:
            # Use cuda if available
            if torch.cuda.is_available():
                data, label = data.to('cuda'), label.to('cuda')
            else:
                data, label = data.to(model.dml), label.to(model.dml)

            # Predict data
            pred = model(data)

            # Calculate loss
            loss = nn.functional.nll_loss(pred, label, reduction='sum').item()

            # Append losses
            total_loss += loss
            pred = pred.data.max(1, keepdim=True)[1]
            correct += pred.eq(label.data.view_as(pred)).cpu().sum()

        avg_loss = total_loss / len(test_loader.dataset)
        tracker['loss']['test'].append(avg_loss)
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            avg_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

if __name__ == '__main__':
    # Run a test before we start
    test()
    
    for epoch in range(1, hyper_params['n_epoch'] + 1):
        train(epoch)
        test()
    
    # After training and test, let's draw the training curve
    fig = plt.figure()
    plt.plot(tracker['counter']['train'], tracker['loss']['train'], color='blue')
    plt.scatter(tracker['counter']['test'], tracker['loss']['test'], color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    plt.savefig('./output/training_curve.png')