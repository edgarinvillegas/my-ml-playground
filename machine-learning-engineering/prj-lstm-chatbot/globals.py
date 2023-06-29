import torch

USE_GPU = torch.cuda.is_available()
device = torch.device("cuda" if USE_GPU else "cpu")

PAD_TOKEN = 0
SOS_TOKEN = 1
EOS_TOKEN = 2
MAX_SENTENCE_LENGTH = 10  # Maximum sentence length to consider
MIN_COUNT = 3    # Min word count for trimming
