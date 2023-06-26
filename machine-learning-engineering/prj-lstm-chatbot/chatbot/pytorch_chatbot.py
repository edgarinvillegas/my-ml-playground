"""
Chatbot Tutorial
================
Code based on Matthew Inkawhich's <https://github.com/MatthewInkawhich>
"""

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import random
import re
import os
import unicodedata
from io import open
import itertools

USE_GPU = torch.cuda.is_available()
device = torch.device("cuda" if USE_GPU else "cpu")

####################
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
# ~~~~~~~~~~~~~~~~~~
#
# Our next order of business is to create a vocabulary and load
# query/response sentence pairs into memory.
#
# Note that we are dealing with sequences of **words**, which do not have
# an implicit mapping to a discrete numerical space. Thus, we must create
# one by mapping each unique word that we encounter in our dataset to an
# index value.
#
# For this we define a ``Voc`` class, which keeps a mapping from words to
# indexes, a reverse mapping of indexes to words, a count of each word and
# a total word count. The class provides methods for adding a word to the
# vocabulary (``addWord``), adding all words in a sentence
# (``addSentence``) and trimming infrequently seen words (``trim``). More
# on trimming later.
#


######################################################################
# Now we can assemble our vocabulary and query/response sentence pairs.
# Before we are ready to use this data, we must perform some
# preprocessing.
#
# First, we must convert the Unicode strings to ASCII using
# ``unicodeToAscii``. Next, we should convert all letters to lowercase and
# trim all non-letter characters except for basic punctuation
# (``normalizeString``). Finally, to aid in training convergence, we will
# filter out sentences with length greater than the ``MAX_SENTENCE_LENGTH``
# threshold (``filterPairs``).
#

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

# Load/Assemble voc and pairs
save_dir = os.path.join("checkpoints")
vocab, pairs = load_prepare_data(train_dataset)
# Print some pairs to validate
print("\npairs:")
for pair in pairs[:10]:
    print(pair)


######################################################################
# Another tactic that is beneficial to achieving faster convergence during
# training is trimming rarely used words out of our vocabulary. Decreasing
# the feature space will also soften the difficulty of the function that
# the model must learn to approximate. We will do this as a two-step
# process:
#
# 1) Trim words used under ``MIN_COUNT`` threshold using the ``voc.trim``
#    function.
#
# 2) Filter out pairs with trimmed words.
#

MIN_COUNT = 3    # Minimum word count threshold for trimming

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


# Trim voc and pairs
pairs = trim_unfrequent_words(vocab, pairs, MIN_COUNT)


######################################################################
# Prepare Data for Models
# -----------------------
#
# Although we have put a great deal of effort into preparing and massaging our
# data into a nice vocabulary object and list of sentence pairs, our models
# will ultimately expect numerical torch tensors as inputs. One way to
# prepare the processed data for the models can be found in the `seq2seq
# translation
# tutorial <https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html>`__.
# In that tutorial, we use a batch size of 1, meaning that all we have to
# do is convert the words in our sentence pairs to their corresponding
# indexes from the vocabulary and feed this to the models.
#
# However, if you’re interested in speeding up training and/or would like
# to leverage GPU parallelization capabilities, you will need to train
# with mini-batches.
#
# Using mini-batches also means that we must be mindful of the variation
# of sentence length in our batches. To accomodate sentences of different
# sizes in the same batch, we will make our batched input tensor of shape
# *(max_length, batch_size)*, where sentences shorter than the
# *max_length* are zero padded after an *EOS_token*.
#
# If we simply convert our English sentences to tensors by converting
# words to their indexes(\ ``indexesFromSentence``) and zero-pad, our
# tensor would have shape *(batch_size, max_length)* and indexing the
# first dimension would return a full sequence across all time-steps.
# However, we need to be able to index our batch along time, and across
# all sequences in the batch. Therefore, we transpose our input batch
# shape to *(max_length, batch_size)*, so that indexing across the first
# dimension returns a time step across all sentences in the batch. We
# handle this transpose implicitly in the ``zeroPadding`` function.
#
# .. figure:: /_static/img/chatbot/seq2seq_batches.png
#    :align: center
#    :alt: batches
#
# The ``inputVar`` function handles the process of converting sentences to
# tensor, ultimately creating a correctly shaped zero-padded tensor. It
# also returns a tensor of ``lengths`` for each of the sequences in the
# batch which will be passed to our decoder later.
#
# The ``outputVar`` function performs a similar function to ``inputVar``,
# but instead of returning a ``lengths`` tensor, it returns a binary mask
# tensor and a maximum target sentence length. The binary mask tensor has
# the same shape as the output target tensor, but every element that is a
# *PAD_token* is 0 and all others are 1.
#
# ``batch2TrainData`` simply takes a bunch of pairs and returns the input
# and target tensors using the aforementioned functions.
#

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


# Example for validation
small_batch_size = 5
batches = batch_to_train_data(vocab, [random.choice(pairs) for _ in range(small_batch_size)])
input_variable, lengths, target_variable, mask, max_target_len = batches

print("input_variable:", input_variable)
print("lengths:", lengths)
print("target_variable:", target_variable)
print("mask:", mask)
print("max_target_len:", max_target_len)


######################################################################
# Define Models
# -------------
#
# Seq2Seq Model
# ~~~~~~~~~~~~~
#
# The brains of our chatbot is a sequence-to-sequence (seq2seq) model. The
# goal of a seq2seq model is to take a variable-length sequence as an
# input, and return a variable-length sequence as an output using a
# fixed-sized model.
#
# `Sutskever et al. <https://arxiv.org/abs/1409.3215>`__ discovered that
# by using two separate recurrent neural nets together, we can accomplish
# this task. One RNN acts as an **encoder**, which encodes a variable
# length input sequence to a fixed-length context vector. In theory, this
# context vector (the final hidden layer of the RNN) will contain semantic
# information about the query sentence that is input to the bot. The
# second RNN is a **decoder**, which takes an input word and the context
# vector, and returns a guess for the next word in the sequence and a
# hidden state to use in the next iteration.
#
# .. figure:: /_static/img/chatbot/seq2seq_ts.png
#    :align: center
#    :alt: model
#
# Image source:
# https://jeddy92.github.io/JEddy92.github.io/ts_seq2seq_intro/
#


######################################################################
# Encoder
# ~~~~~~~
#
# The encoder RNN iterates through the input sentence one token
# (e.g. word) at a time, at each time step outputting an “output” vector
# and a “hidden state” vector. The hidden state vector is then passed to
# the next time step, while the output vector is recorded. The encoder
# transforms the context it saw at each point in the sequence into a set
# of points in a high-dimensional space, which the decoder will use to
# generate a meaningful output for the given task.
#
# At the heart of our encoder is a multi-layered Gated Recurrent Unit,
# invented by `Cho et al. <https://arxiv.org/pdf/1406.1078v3.pdf>`__ in
# 2014. We will use a bidirectional variant of the GRU, meaning that there
# are essentially two independent RNNs: one that is fed the input sequence
# in normal sequential order, and one that is fed the input sequence in
# reverse order. The outputs of each network are summed at each time step.
# Using a bidirectional GRU will give us the advantage of encoding both
# past and future context.
#
# Bidirectional RNN:
#
# .. figure:: /_static/img/chatbot/RNN-bidirectional.png
#    :width: 70%
#    :align: center
#    :alt: rnn_bidir
#
# Image source: https://colah.github.io/posts/2015-09-NN-Types-FP/
#
# Note that an ``embedding`` layer is used to encode our word indices in
# an arbitrarily sized feature space. For our models, this layer will map
# each word to a feature space of size *hidden_size*. When trained, these
# values should encode semantic similarity between similar meaning words.
#
# Finally, if passing a padded batch of sequences to an RNN module, we
# must pack and unpack padding around the RNN pass using
# ``nn.utils.rnn.pack_padded_sequence`` and
# ``nn.utils.rnn.pad_packed_sequence`` respectively.
#
# **Computation Graph:**
#
#    1) Convert word indexes to embeddings.
#    2) Pack padded batch of sequences for RNN module.
#    3) Forward pass through GRU.
#    4) Unpack padding.
#    5) Sum bidirectional GRU outputs.
#    6) Return output and final hidden state.
#
# **Inputs:**
#
# -  ``input_seq``: batch of input sentences; shape=\ *(max_length,
#    batch_size)*
# -  ``input_lengths``: list of sentence lengths corresponding to each
#    sentence in the batch; shape=\ *(batch_size)*
# -  ``hidden``: hidden state; shape=\ *(num_layers x num_directions,
#    batch_size, hidden_size)*
#
# **Outputs:**
#
# -  ``outputs``: output features from the last hidden layer of the GRU
#    (sum of bidirectional outputs); shape=\ *(max_length, batch_size,
#    hidden_size)*
# -  ``hidden``: updated hidden state from GRU; shape=\ *(num_layers x
#    num_directions, batch_size, hidden_size)*
#
#

class Encoder(nn.Module):
    def __init__(self, hidden_size, embedding):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = 2
        self.embedding = embedding

        self.gru = nn.GRU(hidden_size, hidden_size, self.num_layers, dropout=0.1, bidirectional=True)

    def forward(self, input_sequence, input_lengths, hidden=None):
        embedded = self.embedding(input_sequence)
        packed_sequence = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths.to('cpu'))
        output, hidden = self.gru(packed_sequence, hidden)
        output, _ = nn.utils.rnn.pad_packed_sequence(output)
        output = output[:, :, :self.hidden_size] + output[:, : ,self.hidden_size:]
        return output, hidden

class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()

    def forward(self, hidden, encoder_outputs):
        energies = torch.sum(hidden * encoder_outputs, dim=2)
        energies = energies.t()
        return F.softmax(energies, dim=1).unsqueeze(1)

class Decoder(nn.Module):
    def __init__(self, embedding, hidden_size, output_size):
        super(Decoder, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = 2
        dropout = 0.1

        self.embedding = embedding
        self.emb_dropout = nn.Dropout(dropout)
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, self.num_layers, dropout=dropout)
        self.out = nn.Linear(hidden_size, output_size)
        self.attn = Attention()

    def forward(self, input_step, prev_hidden, enc_outputs):
        embedded = self.emb_dropout(self.embedding(input_step))
        rnn_output, hidden = self.gru(embedded, prev_hidden)
        attention_weights = self.attn(rnn_output, enc_outputs)
        context_vector = attention_weights.bmm(enc_outputs.transpose(0, 1))
        rnn_output = rnn_output.squeeze(0)
        context_vector = context_vector.squeeze(1)
        concat_input = torch.cat((rnn_output, context_vector), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        output = self.out(concat_output) # Predict next word
        output = F.softmax(output, dim=1)
        return output, hidden


class Seq2Seq(nn.Module):

    def __init__(self, encoder_hidden_size, decoder_hidden_size, decoder_output_size, embedding):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(encoder_hidden_size, embedding)
        self.decoder = Decoder(embedding, decoder_hidden_size, decoder_output_size)

    def forward_batch(self, src, trg, max_target_len, lengths, mask, teacher_forcing_ratio=0.5, batch_size=1):
        # Set device options
        src = src.to(device)
        lengths = lengths.to(device)
        trg = trg.to(device)
        mask = mask.to(device)
        # Initialize variables
        loss = 0
        print_losses = []
        n_totals = 0
        # Forward pass through encoder
        encoder_outputs, encoder_hidden = self.encoder(src, lengths)
        # Create initial decoder input (start with SOS tokens for each sentence)
        decoder_input = torch.LongTensor([[SOS_TOKEN for _ in range(batch_size)]])
        decoder_input = decoder_input.to(device)
        # Set initial decoder hidden state to the encoder's final hidden state
        decoder_hidden = encoder_hidden[:self.decoder.num_layers]
        # Determine if we are using teacher forcing this iteration
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        # Forward batch of sequences through decoder one time step at a time
        # print('max_target_len: ', max_target_len)
        for t in range(max_target_len):
            decoder_output, decoder_hidden = self.decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            if use_teacher_forcing:
                # Teacher forcing: next input is current target
                decoder_input = trg[t].view(1, -1)
            else:
                # No teacher forcing: next input is decoder's own current output
                _, topi = decoder_output.topk(1)
                decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
                decoder_input = decoder_input.to(device)
            # Calculate and accumulate loss
            mask_loss, nTotal = mask_NLLLoss(decoder_output, trg[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal
        return loss, n_totals, print_losses


######################################################################
# Define Training Procedure
# -------------------------
#
# Masked loss
# ~~~~~~~~~~~
#
# Since we are dealing with batches of padded sequences, we cannot simply
# consider all elements of the tensor when calculating loss. We define
# ``maskNLLLoss`` to calculate our loss based on our decoder’s output
# tensor, the target tensor, and a binary mask tensor describing the
# padding of the target tensor. This loss function calculates the average
# negative log likelihood of the elements that correspond to a *1* in the
# mask tensor.
#

def mask_NLLLoss(inp, target, mask):
    nTotal = mask.sum()
    crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.to(device)
    return loss, nTotal.item()


######################################################################
# Single training iteration
# ~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The ``train`` function contains the algorithm for a single training
# iteration (a single batch of inputs).
#
# We will use a couple of clever tricks to aid in convergence:
#
# -  The first trick is using **teacher forcing**. This means that at some
#    probability, set by ``teacher_forcing_ratio``, we use the current
#    target word as the decoder’s next input rather than using the
#    decoder’s current guess. This technique acts as training wheels for
#    the decoder, aiding in more efficient training. However, teacher
#    forcing can lead to model instability during inference, as the
#    decoder may not have a sufficient chance to truly craft its own
#    output sequences during training. Thus, we must be mindful of how we
#    are setting the ``teacher_forcing_ratio``, and not be fooled by fast
#    convergence.
#
# -  The second trick that we implement is **gradient clipping**. This is
#    a commonly used technique for countering the “exploding gradient”
#    problem. In essence, by clipping or thresholding gradients to a
#    maximum value, we prevent the gradients from growing exponentially
#    and either overflow (NaN), or overshoot steep cliffs in the cost
#    function.
#
# .. figure:: /_static/img/chatbot/grad_clip.png
#    :align: center
#    :width: 60%
#    :alt: grad_clip
#
# Image source: Goodfellow et al. *Deep Learning*. 2016. https://www.deeplearningbook.org/
#
# **Sequence of Operations:**
#
#    1) Forward pass entire input batch through encoder.
#    2) Initialize decoder inputs as SOS_token, and hidden state as the encoder's final hidden state.
#    3) Forward input batch sequence through decoder one time step at a time.
#    4) If teacher forcing: set next decoder input as the current target; else: set next decoder input as current decoder output.
#    5) Calculate and accumulate loss.
#    6) Perform backpropagation.
#    7) Clip gradients.
#    8) Update encoder and decoder model parameters.
#
#
# .. Note ::
#
#   PyTorch’s RNN modules (``RNN``, ``LSTM``, ``GRU``) can be used like any
#   other non-recurrent layers by simply passing them the entire input
#   sequence (or batch of sequences). We use the ``GRU`` layer like this in
#   the ``encoder``. The reality is that under the hood, there is an
#   iterative process looping over each time step calculating hidden states.
#   Alternatively, you ran run these modules one time-step at a time. In
#   this case, we manually loop over the sequences during the training
#   process like we must do for the ``decoder`` model. As long as you
#   maintain the correct conceptual model of these modules, implementing
#   sequential models can be very straightforward.
#
#


def train_step(input_variable, lengths, target_variable, mask, max_target_len, seq2seq, encoder_optimizer,
               decoder_optimizer, batch_size, clip):

    # Zero gradients
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    loss, n_totals, print_losses = seq2seq.forward_batch(input_variable, target_variable, max_target_len, lengths, mask,
                                                   teacher_forcing_ratio, batch_size)

    # Perform backpropatation
    loss.backward()

    # Clip gradients: gradients are modified in place
    _ = nn.utils.clip_grad_norm_(seq2seq.encoder.parameters(), clip)
    _ = nn.utils.clip_grad_norm_(seq2seq.decoder.parameters(), clip)

    # Adjust model weights
    encoder_optimizer.step()
    decoder_optimizer.step()

    avg_loss = sum(print_losses) / n_totals
    return avg_loss


######################################################################
# Training iterations
# ~~~~~~~~~~~~~~~~~~~
#
# It is finally time to tie the full training procedure together with the
# data. The ``trainIters`` function is responsible for running
# ``n_iterations`` of training given the passed models, optimizers, data,
# etc. This function is quite self explanatory, as we have done the heavy
# lifting with the ``train`` function.
#
# One thing to note is that when we save our model, we save a tarball
# containing the encoder and decoder state_dicts (parameters), the
# optimizers’ state_dicts, the loss, the iteration, etc. Saving the model
# in this way will give us the ultimate flexibility with the checkpoint.
# After loading a checkpoint, we will be able to use the model parameters
# to run inference, or we can continue training right where we left off.
#

def train_loop(vocab, pairs, seq2seq, encoder_optimizer, decoder_optimizer, embedding, save_dir, epochs,
               batch_size, print_every, save_every, clip, load_filename):

    # Load batches for each iteration
    training_batches = [batch_to_train_data(vocab, [random.choice(pairs) for _ in range(batch_size)])
                        for _ in range(epochs)]

    # Initializations
    print('Initializing ...')
    start_iteration = 1
    print_loss = 0
    if load_filename:
        start_iteration = checkpoint['iteration'] + 1

    # Training loop
    print("Training...")
    for epoch in range(start_iteration, epochs + 1):
        training_batch = training_batches[epoch - 1]
        # Extract fields from batch
        input_variable, lengths, target_variable, mask, max_target_len = training_batch

        # Run a training iteration with batch
        loss = train_step(input_variable, lengths, target_variable, mask, max_target_len, seq2seq, encoder_optimizer,
                          decoder_optimizer, batch_size, clip)
        print_loss += loss

        # Print progress
        if epoch % print_every == 0:
            print_loss_avg = print_loss / print_every
            print("Epoch: {}; Completion: {:.1f}%; Avg loss: {:.4f}".format(epoch, epoch / epochs * 100, print_loss_avg))
            print_loss = 0

        # Save checkpoint
        if (epoch % save_every == 0):
            directory = os.path.join(save_dir, '{}'.format(hidden_size))
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save({
                'iteration': epoch,
                'vocab_dict': vocab.__dict__,
                'encoder': encoder.state_dict(),
                'encoder_optimizer': encoder_optimizer.state_dict(),
                'decoder': decoder.state_dict(),
                'decoder_optimizer': decoder_optimizer.state_dict(),
                'loss': loss,
                'embedding': embedding.state_dict()
            }, os.path.join(directory, '{}_{}.tar'.format(epoch, 'checkpoint')))


######################################################################
# Define Evaluation
# -----------------
#
# After training a model, we want to be able to talk to the bot ourselves.
# First, we must define how we want the model to decode the encoded input.
#
# Greedy decoding
# ~~~~~~~~~~~~~~~
#
# Greedy decoding is the decoding method that we use during training when
# we are **NOT** using teacher forcing. In other words, for each time
# step, we simply choose the word from ``decoder_output`` with the highest
# softmax value. This decoding method is optimal on a single time-step
# level.
#
# To facilite the greedy decoding operation, we define a
# ``GreedySearchDecoder`` class. When run, an object of this class takes
# an input sequence (``input_seq``) of shape *(input_seq length, 1)*, a
# scalar input length (``input_length``) tensor, and a ``max_length`` to
# bound the response sentence length. The input sentence is evaluated
# using the following computational graph:
#
# **Computation Graph:**
#
#    1) Forward input through encoder model.
#    2) Prepare encoder's final hidden layer to be first hidden input to the decoder.
#    3) Initialize decoder's first input as SOS_token.
#    4) Initialize tensors to append decoded words to.
#    5) Iteratively decode one word token at a time:
#        a) Forward pass through decoder.
#        b) Obtain most likely word token and its softmax score.
#        c) Record token and score.
#        d) Prepare current token to be next decoder input.
#    6) Return collections of word tokens and scores.
#

class SearchDecoder(nn.Module):
    def __init__(self, seq2seq):
        super(SearchDecoder, self).__init__()
        self.seq2seq = seq2seq

    def forward(self, input_seq, input_length, max_length):
        # Forward input through encoder model
        encoder_outputs, encoder_hidden = self.seq2seq.encoder(input_seq, input_length)
        # Prepare encoder's final hidden layer to be first hidden input to the decoder
        decoder_hidden = encoder_hidden[:decoder.num_layers]
        # Initialize decoder input with SOS_token
        decoder_input = torch.ones(1, 1, device=device, dtype=torch.long) * SOS_TOKEN
        # Initialize tensors to append decoded words to
        tokens = torch.zeros([0], device=device, dtype=torch.long)
        scores = torch.zeros([0], device=device)
        # Iteratively decode one word token at a time
        for _ in range(max_length):
            # Forward pass through decoder
            decoder_output, decoder_hidden = self.seq2seq.decoder(decoder_input, decoder_hidden, encoder_outputs)
            # Obtain most likely word token and its softmax score
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            # Record token and score
            tokens = torch.cat((tokens, decoder_input), dim=0)
            scores = torch.cat((scores, decoder_scores), dim=0)
            # Prepare current token to be next decoder input (add a dimension)
            decoder_input = torch.unsqueeze(decoder_input, 0)
        # Return collections of word tokens and scores
        return tokens, scores


######################################################################
# Evaluate my text
# ~~~~~~~~~~~~~~~~
#
# Now that we have our decoding method defined, we can write functions for
# evaluating a string input sentence. The ``evaluate`` function manages
# the low-level process of handling the input sentence. We first format
# the sentence as an input batch of word indexes with *batch_size==1*. We
# do this by converting the words of the sentence to their corresponding
# indexes, and transposing the dimensions to prepare the tensor for our
# models. We also create a ``lengths`` tensor which contains the length of
# our input sentence. In this case, ``lengths`` is scalar because we are
# only evaluating one sentence at a time (batch_size==1). Next, we obtain
# the decoded response sentence tensor using our ``GreedySearchDecoder``
# object (``searcher``). Finally, we convert the response’s indexes to
# words and return the list of decoded words.
#
# ``evaluateInput`` acts as the user interface for our chatbot. When
# called, an input text field will spawn in which we can enter our query
# sentence. After typing our input sentence and pressing *Enter*, our text
# is normalized in the same way as our training data, and is ultimately
# fed to the ``evaluate`` function to obtain a decoded output sentence. We
# loop this process, so we can keep chatting with our bot until we enter
# either “q” or “quit”.
#
# Finally, if a sentence is entered that contains a word that is not in
# the vocabulary, we handle this gracefully by printing an error message
# and prompting the user to enter another sentence.
#

def evaluate(searcher, vocab, sentence, max_sentence_length=MAX_SENTENCE_LENGTH):
    ### Format input sentence as a batch
    # words -> indexes
    indexes_batch = [get_indexes_from_sentence(vocab, sentence)]
    # Create lengths tensor
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    # Transpose dimensions of batch to match models' expectations
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    # Use appropriate device
    input_batch = input_batch.to(device)
    lengths = lengths.to(device)
    # Decode sentence with searcher
    tokens, scores = searcher(input_batch, lengths, max_sentence_length)
    # indexes -> words
    decoded_words = [vocab.index2word[token.item()] for token in tokens]
    return decoded_words


def eval_input(searcher, vocab):
    while(True):
        try:
            # Get input sentence
            input_sentence = input('You> ')
            # Check if it is quit case
            if input_sentence == 'q' or input_sentence == 'quit': break
            # Normalize sentence
            input_sentence = normalize_string(input_sentence)
            # Evaluate sentence
            output_words = evaluate(searcher, vocab, input_sentence)
            # Format and print response sentence
            output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
            print('Chatbot>', ' '.join(output_words))

        except KeyError:
            print("ERROR: Unknown word.")


######################################################################
# Run Model
# ---------
#
# Finally, it is time to run our model!
#
# Regardless of whether we want to train or test the chatbot model, we
# must initialize the individual encoder and decoder models. In the
# following block, we set our desired configurations, choose to start from
# scratch or set a checkpoint to load from, and build and initialize the
# models. Feel free to play with different model configurations to
# optimize performance.
#

# Configure models
hidden_size = 500
batch_size = 64

# Set checkpoint to load from; set to None if starting from scratch

checkpoint_iter = 4000
load_filename = os.path.join(save_dir, '{}'.format(hidden_size), '{}_checkpoint.tar'.format(checkpoint_iter))
# load_filename = None

# Load model if a loadFilename is provided
if load_filename:
    print('Loading from ', load_filename)
    # If loading on same machine the model was trained on
    checkpoint = torch.load(load_filename)
    # If loading a model trained on GPU to CPU
    #checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
    encoder_sd = checkpoint['encoder']
    decoder_sd = checkpoint['decoder']
    encoder_optimizer_sd = checkpoint['encoder_optimizer']
    decoder_optimizer_sd = checkpoint['decoder_optimizer']
    embedding_sd = checkpoint['embedding']
    vocab.__dict__ = checkpoint['vocab_dict']


print('Building encoder and decoder ...')
# Initialize word embeddings
embedding = nn.Embedding(vocab.num_words, hidden_size)
if load_filename:
    embedding.load_state_dict(embedding_sd)
# Initialize encoder & decoder models
seq2seq = Seq2Seq(hidden_size, hidden_size, vocab.num_words, embedding)
encoder = seq2seq.encoder
decoder = seq2seq.decoder
if load_filename:
    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)
# Use appropriate device
encoder = encoder.to(device)
decoder = decoder.to(device)
print('Ready!')


######################################################################
# Run Training
# ~~~~~~~~~~~~
#
# Run the following block if you want to train the model.
#
# First we set training parameters, then we initialize our optimizers, and
# finally we call the ``trainIters`` function to run our training
# iterations.
#

# Configure training/optimization
clip = 50.0
teacher_forcing_ratio = 0.5
learning_rate = 0.0001
decoder_learning_ratio = 5.0
epochs = 4000

print_every = 1
save_every = 500

# Ensure dropout layers are in train mode
encoder.train()
decoder.train()

# Initialize optimizers
print('Building optimizers ...')
encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
if load_filename:
    encoder_optimizer.load_state_dict(encoder_optimizer_sd)
    decoder_optimizer.load_state_dict(decoder_optimizer_sd)

# Run training iterations
print("Starting Training!")
train_loop(vocab, pairs, seq2seq, encoder_optimizer, decoder_optimizer, embedding, save_dir, epochs, batch_size,
           print_every, save_every, clip, load_filename)


######################################################################
# Run Evaluation
# ~~~~~~~~~~~~~~
#
# To chat with your model, run the following block.
#

# Set dropout layers to eval mode
encoder.eval()
decoder.eval()

# Initialize search module
searcher = SearchDecoder(seq2seq)

# Begin chatting (uncomment and run the following line to begin)
eval_input(searcher, vocab)


######################################################################
# Conclusion
# ----------
#
# That’s all for this one, folks. Congratulations, you now know the
# fundamentals to building a generative chatbot model! If you’re
# interested, you can try tailoring the chatbot’s behavior by tweaking the
# model and training parameters and customizing the data that you train
# the model on.
#
# Check out the other tutorials for more cool deep learning applications
# in PyTorch!
#