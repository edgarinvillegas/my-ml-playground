#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
# import torch.nn as nn
from torch import nn
# import torch.optim as optim
from torch import optim
import torchvision
import torchvision.models as models
# import torchvision.transforms as transforms
from torchvision import datasets, transforms, models
from PIL import ImageFile

import argparse
import os
import json

#TODO: Import dependencies for Debugging andd Profiling

def test(model, test_loader):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    pass

def train(model, trainloader, criterion, optimizer):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''



def get_device():
    # Use GPU if it's available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


def net():
    '''
    TODONE: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    model = models.densenet121(pretrained=True)
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    from collections import OrderedDict
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(1024, 512)),
        ('relu', nn.ReLU()),
        ('dropout', nn.Dropout(0.2)),
        ('fc2', nn.Linear(512, 133)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    model.classifier = classifier
    model.to(get_device())
    return model


def create_data_loaders(train_dir, batch_size, test_dir, test_batch_size):
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    # Pass transforms in here, then run the next cell to see how the transforms look
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=test_batch_size)
    return trainloader, testloader


def train_test(model, trainloader, testloader, criterion, optimizer, epochs):
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    steps = 0
    running_loss = 0
    print_every = 5
    device = get_device()
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            logps = model.forward(inputs)
            loss = criterion(logps, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in testloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        test_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch + 1}/{epochs}.. "
                      f"Train loss: {running_loss / print_every:.3f}.. "
                      f"Test loss: {test_loss / len(testloader):.3f}.. "
                      f"Test accuracy: {accuracy / len(testloader):.3f}.. "
                      f"Progress: {100 * steps / epochs / len(trainloader):.1f}%"
                      )
                running_loss = 0
                model.train()
    return model

def main(args):
    print('ALL ARGS: ', args)
    '''
    TODO: Initialize a model by calling the net function
    '''
    model=net()
    
    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.lr)
    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    trainloader, testloader = create_data_loaders(args.train_dir, args.batch_size, args.test_dir, args.test_batch_size)
    assert len(trainloader) > 0
    assert len(testloader) > 0

    model = train_test(model, trainloader, testloader, loss_criterion, optimizer, args.epochs)

    # model=train(model, trainloader, loss_criterion, optimizer)
    
    '''
    TODO: Test the model to see its accuracy
    '''
    # test(model, test_loader, criterion)
    
    '''
    TODO: Save the trained model
    '''
    torch.save(model, args.model_dir + 'model.pth')

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    TODONE: Specify any training args that you might need
    '''
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=256,
        metavar="N",
        help="input batch size for testing (default: 256)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        metavar="N",
        help="number of epochs to train (default: 1)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.003, metavar="LR", help="learning rate (default: 0.03)"
    )

    # Container environment
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
    parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--train-dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--test-dir", type=str, default=os.environ["SM_CHANNEL_TEST"])
    parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])

    args=parser.parse_args()
    
    main(args)
    print('done')
