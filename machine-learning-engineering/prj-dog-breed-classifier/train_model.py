import numpy as np
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from PIL import ImageFile

import argparse
import os
import json


# DONE: Import dependencies for Debugging andd Profiling
# ====================================#
# 1. Import SMDebug framework class. #
# ====================================#
import smdebug.pytorch as smd


def train(model, trainloader, criterion, optimizer, hook, epoch):
    '''
    DONE: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    model.train()
    # =================================================#
    # 2a. Set the SMDebug hook for the training phase. #
    # =================================================#
    #if hook:
    hook.set_mode(smd.modes.TRAIN)        
        
    device = get_device()
    running_avg_loss = 0
    print_every = 10

    for batch_idx, (inputs, labels) in enumerate(trainloader):
        # steps += 1
        # Move input and label tensors to the default device
        inputs, labels = inputs.to(device), labels.to(device)

        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        # print('recording train loss tensor')
        hook.record_tensor_value(tensor_name="NLLLoss", tensor_value=loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_avg_loss = loss.item() / inputs.shape[0]
        running_avg_loss += loss.item()

        if batch_idx % print_every == 0:
            print(f"  Epoch {epoch + 1}.. "
                  f"Epoch progress: {100 * batch_idx / len(trainloader):.1f}%.. "
                  f"Batch avg train loss: {batch_avg_loss:.3f}.. ")

    running_avg_loss /= len(trainloader.dataset)

    return (running_avg_loss,)


def test(model, testloader, criterion, hook=None, epoch=0):
    '''
    DONE: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    model.eval()
    # ===================================================#
    # 2b. Set the SMDebug hook for the validation phase. #
    # ===================================================#
    #if hook:
    hook.set_mode(smd.modes.EVAL)
    
    device = get_device()
    accuracies = []
    epoch_avg_loss = 0
    
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)
            # test_losses.append(batch_loss.item())
            # print('recording test loss tensor')
            hook.record_tensor_value(tensor_name="NLLLoss", tensor_value=batch_loss)
            
            epoch_avg_loss += batch_loss.item()

            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracies.append(torch.mean(equals.type(torch.FloatTensor)).item())

    epoch_avg_loss /= len(testloader.dataset)
    epoch_avg_accuracy = np.mean(accuracies)

    model.train()
    return epoch_avg_loss, epoch_avg_accuracy


def get_device():
    # Use GPU if it's available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


def net():
    '''
    DONE: Complete this function that initializes your model
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
    print('Device: ', get_device())
    # model.to(get_device())
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



def main(args):
    print('ALL ARGS: ', args)
    '''
    DONE: Initialize a model by calling the net function
    '''
    model = net()
    model.to(get_device())
    
    # ======================================================#
    # 3a. Register the SMDebug hook to save output tensors. #
    # ======================================================#
    #hook = smd.Hook.create_from_json_file()
    #hook.register_hook(model)
    #hook.register_module(model)
    
    hook = smd.get_hook(create_if_not_exists=True)    

    '''
    DONE: Create your loss and optimizer
    '''
    loss_criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.lr)
    
    #if hook:
    # hook.register_loss(loss_criterion)
    # print('loss registered')

    '''
    DONE: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    trainloader, testloader = create_data_loaders(args.train_dir, args.batch_size, args.test_dir, args.test_batch_size)
    assert len(trainloader) > 0
    assert len(testloader) > 0

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    epochs = args.epochs

    print('Start training...')
    for epoch in range(epochs):
        # ===========================================================#
        # 3b. Pass the SMDebug hook to the train and test functions. #
        # ===========================================================#
        (train_loss,) = train(model, trainloader, loss_criterion, optimizer, hook, epoch)
        (test_loss, test_accuracy) = test(model, testloader, loss_criterion, hook, epoch)

        print(f"Epoch {epoch + 1}.. "
              f"Progress: {100 * epoch / epochs:.1f}% "
              f"Train loss: {train_loss:.3f} "
              f"Test loss: {test_loss:.3f} "
              f"Test accuracy: {test_accuracy:.3f}")

    '''
    DONE: Test the model to see its accuracy
    '''
    (test_loss, test_accuracy) = test(model, testloader, loss_criterion, hook)
    print(f"Final test accuracy: {test_accuracy:.3f}")

    '''
    DONE: Save the trained model
    '''
    torch.save(model, args.model_dir + '/model2.pth')

    
def model_fn(model_dir):
    model = net()
    print('Calling model_fn, model_dir=', model_dir)
    with open(os.path.join(model_dir, 'model2.pth'), 'rb') as f:
        model.load_state_dict(torch.load(f))
    return model


def predict_fn(input_data, model):
    device = get_device()
    print('Calling predict_fn with device ', device, '. Data: input_data', input_data)
    model.to(device)
    model.eval()
    with torch.no_grad():
        return model(input_data.to(device))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    '''
    DONE: Specify any training args that you might need
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

    args = parser.parse_args()

    main(args)
    print('done')
