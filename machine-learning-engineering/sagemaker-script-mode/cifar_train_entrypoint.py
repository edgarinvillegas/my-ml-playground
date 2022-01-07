import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

class Net(nn.Module):
    def __init__(self):
        #TODO: Complete this function
        super().__init__()
        self.fc1 = nn.Linear(32*32*3, 1000)
        self.fc2 = nn.Linear(1000, 300)
        self.fc3 = nn.Linear(300, 90)
        self.fc4 = nn.Linear(90, 10)
        
    def forward(self, x):
        #print('original x shape: ', x.shape)
        #TODO: Complete the forward function
        x = x.flatten(start_dim=1)
        #print('flattened x shape: ', x.shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))        
        x = F.relu(self.fc3(x))        
        x = F.log_softmax(self.fc4(x), dim=1)
        return x

def train(model, train_loader, optimizer, epoch):
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )


def test(model, test_loader):
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)
        )
    )

def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch CIFAR Example")
    # TODO: Add your arguments here
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        metavar="N",
        help="Number of epochs (default: 2)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        metavar="N",
        help="input batch size for training (default: 32)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for testing (default: 64)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        metavar="N",
        help="learning rate (default: 0.001)",
    )
    return parser.parse_args()
    
def main():
    # Training settings
    
    args = parse_args()    
    batch_size = args.batch_size
    test_batch_size = args.test_batch_size
    lr = args.lr
    epochs = args.epochs
    
    print('args: ', args)
    
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0, 0, 0), (1, 1, 1)),
        transforms.RandomHorizontalFlip()
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0, 0, 0), (1, 1, 1))
    ])

    # TODO: Add the CIFAR10 dataset and create your data loaders        
    trainset = datasets.CIFAR10('./data', download=True, train=True, transform=transform_train)
    testset = datasets.CIFAR10('./data', download=True, train=False, transform=transform_test)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=True)

    model = Net()

    optimizer = optim.Adam(model.parameters(), lr=lr) # TODO: Add your optimizer

    for epoch in range(1, epochs + 1):
        train(model, train_loader, optimizer, epoch)
        test(model, test_loader)


if __name__ == "__main__":
    main()
