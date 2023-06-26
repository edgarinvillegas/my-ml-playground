import torch
import re
import unicodedata
from io import open
import itertools

from torchtext.datasets import SQuAD2

train_dataset = SQuAD2(split='dev')
train_iter = iter(train_dataset)

def print_lines(file, n=10):
    with open(file, 'rb') as datafile:
        lines = datafile.readlines()
        for line in lines[:n]:
            print(line)

def extract_questions_answers(dataset):
    data_iter = iter(dataset)
    pairs = []
    row = next(data_iter, None)
    while row is not None:
        question = remove_question_words(row[1])
        answer = row[2][0]
        pairs.append([question, answer])
        row = next(data_iter, None)
    return pairs

MAX_SENTENCE_LENGTH = 10  # Maximum sentence length to consider

# Thanks to https://stackoverflow.com/a/518232/2809427
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s

# Default word tokens
PAD_TOKEN = 0
SOS_TOKEN = 1
EOS_TOKEN = 2

class Vocab:
    def __init__(self):
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_TOKEN: "PAD", SOS_TOKEN: "SOS", EOS_TOKEN: "EOS"}
        self.num_words = 3  # Count SOS, EOS, PAD

    def add_qa(self, qa):
        for word in qa.split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    # Remove too frequent words
    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = [k for k, v in self.word2count.items() if v >= min_count]

        print('keep_words {} / {} = {:.4f}'.format(
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_TOKEN: "PAD", SOS_TOKEN: "SOS", EOS_TOKEN: "EOS"}
        self.num_words = 3 # Default tokens

        for word in keep_words:
            self.add_word(word)

######################################################################
# Load and trim data

MAX_SENTENCE_LENGTH = 10  # Maximum sentence length to consider

# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s


# Read question/answer pairs and return a Vocab object
def read_vocabs(dataset):
    raw_pairs = extract_questions_answers(dataset)
    # Split lines into qa pairs and normalize
    pairs = [[normalize_string(s) for s in l] for l in raw_pairs]
    vocab = Vocab()
    return vocab, pairs

# Remove question words to make phrases shorter
def remove_question_words(str):
    return str.lower().replace('?', '').replace('what', '').replace('who', '').replace('why', '').replace('how', '').replace('when', '').replace('where', '').replace('which', '').replace('whose', '').strip()

# To make question and answer of similar sizes
def are_size_compatible(answer, question):
    l1, l2 = len(answer), len(question)
    return abs(l1 - l2) < max(l1, l2) * 0.5

# Returns True if both sentences in a pair 'p' are under the MAX_SENTENCE_LENGTH threshold
def filter_pair(p):
    # Input sequences need to preserve the last word for EOS token
    return len(p[0].split(' ')) < MAX_SENTENCE_LENGTH and len(p[1].split(' ')) < MAX_SENTENCE_LENGTH and are_size_compatible(p[0], p[1])

# Filter qa pairs using filter_pair predicate
def filter_pairs(pairs):
    return [pair for pair in pairs if filter_pair(pair)]

def load_prepare_data(dataset):
    print("Start preparing training data ...")
    vocab, pairs = read_vocabs(dataset)
    print("Read {!s} sentence pairs".format(len(pairs)))
    pairs = filter_pairs(pairs)
    print("Trimmed to {!s} sentence pairs".format(len(pairs)))
    print("Counting words...")
    for pair in pairs:
        vocab.add_qa(pair[0])
        vocab.add_qa(pair[1])
    print("Counted words:", vocab.num_words)
    return vocab, pairs

def trim_unfrequent_words(vocab, pairs, MIN_COUNT):
    # Trim words used under the MIN_COUNT from the voc
    vocab.trim(MIN_COUNT)
    # Filter out pairs with trimmed words
    keep_pairs = []
    for pair in pairs:
        input_sentence = pair[0]
        output_sentence = pair[1]
        keep_input = True
        keep_output = True
        # Check input sentence
        for word in input_sentence.split(' '):
            if word not in vocab.word2index:
                keep_input = False
                break
        # Check output sentence
        for word in output_sentence.split(' '):
            if word not in vocab.word2index:
                keep_output = False
                break

        # Only keep pairs that do not contain trimmed word(s) in their input or output sentence
        if keep_input and keep_output:
            keep_pairs.append(pair)

    print("Trimmed from {} pairs to {}, {:.4f} of total".format(len(pairs), len(keep_pairs), len(keep_pairs) / len(pairs)))
    return keep_pairs


def get_indexes_from_sentence(vocab, sentence):
    return [vocab.word2index[word] for word in sentence.split(' ')] + [EOS_TOKEN]


def zero_pad(l, fillvalue=PAD_TOKEN):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))

def bin_matrix(l):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == PAD_TOKEN:
                m[i].append(0)
            else:
                m[i].append(1)
    return m

# Returns padded input sequence tensor and lengths
def prepare_input(l, vocab):
    indexes_batch = [get_indexes_from_sentence(vocab, sentence) for sentence in l]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zero_pad(indexes_batch)
    padVar = torch.LongTensor(padList)
    return padVar, lengths

# Returns padded target sequence tensor, padding mask, and max target length
def prepare_output(l, vocab):
    indexes_batch = [get_indexes_from_sentence(vocab, sentence) for sentence in l]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zero_pad(indexes_batch)
    mask = bin_matrix(padList)
    mask = torch.ByteTensor(mask)
    padVar = torch.LongTensor(padList)
    return padVar, mask, max_target_len

# Returns all items for a given batch of pairs
def batch_to_train_data(vocab, pair_batch):
    pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp, lengths = prepare_input(input_batch, vocab)
    output, mask, max_target_len = prepare_output(output_batch, vocab)
    return inp, lengths, output, mask, max_target_len
