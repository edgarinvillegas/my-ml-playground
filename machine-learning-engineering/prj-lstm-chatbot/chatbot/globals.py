import torch

USE_GPU = torch.cuda.is_available()
device = torch.device("cuda" if USE_GPU else "cpu")

PAD_TOKEN = 0
SOS_TOKEN = 1
EOS_TOKEN = 2
