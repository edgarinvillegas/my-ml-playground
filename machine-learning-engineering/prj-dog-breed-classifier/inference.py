import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import os
import json
import io

import sys

def get_device():
    # Use GPU if it's available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device

def model_fn(model_dir):
    with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
        model = torch.load(f)
    model.to(get_device())
    return model


def input_fn(request_body, content_type):    
    if content_type == 'image/jpeg': 
        return Image.open(io.BytesIO(request_body))    
    raise Exception(f'Only image/jpeg is supported. Sent {content_type}')


def predict_fn(input_object, model):
    device = get_device()
    print('Calling predict_fn with device ', device, '. Data: input_data', input_object)
    model.to(device)
    model.eval()
    
    test_transform = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
    input_object=test_transform(input_object)

    with torch.no_grad():        
        prediction = model(input_object.unsqueeze(0).to(device))
    return prediction


