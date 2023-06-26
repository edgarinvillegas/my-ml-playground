import torch
import torch.nn.functional as F
import torch.nn as nn
import random
from chatbot.globals import SOS_TOKEN, PAD_TOKEN, EOS_TOKEN, device

######################################################################
# Define Models

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


def mask_NLLLoss(inp, target, mask):
    nTotal = mask.sum()
    crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.to(device)
    return loss, nTotal.item()
