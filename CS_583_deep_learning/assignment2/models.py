import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F

# Local imports
import utils

# question 1,2,3 and 5 
class MyGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MyGRUCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size  # hidden state size
        self.Wiz = nn.Linear(input_size, hidden_size)
        self.Wir = nn.Linear(input_size, hidden_size)
        self.Wih = nn.Linear(input_size, hidden_size)
        # hidden layers
        self.Whz = nn.Linear(hidden_size, hidden_size)
        self.Whr = nn.Linear(hidden_size, hidden_size)
        self.Whh = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, h_prev):
        """Forward pass of the GRU computation for one time step.

        Arguments
            x: batch_size x input_size
            h_prev: batch_size x hidden_size

        Returns:
            h_new: batch_size x hidden_size

        \begin{array}{ll}
        z = \sigma(W_{iz} x + b_{iz} + W_{hz} h + b_{hz}) \\
        r = \sigma(W_{ir} x + b_{ir} + W_{hr} h + b_{hr}) \\
       
        n = \ tanh(W_{in} x + b_{in} + r * (W_{hn} h + b_{hn})) \\
        h' = (1 - z) * n + z * h
        \end{array}
        """

        z = torch.sigmoid(self.Wiz(x) + self.Whz(h_prev))
        r = torch.sigmoid(self.Wir(x) + self.Whr(h_prev))
        cc = r * self.Whh(h_prev)
        g = torch.tanh(self.Wih(x) + cc )
        h_new = (1 - z) * g + z * h_prev
        # (1-z) * g + z * h_prev
        return h_new


class GRUEncoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, opts):
        super(GRUEncoder, self).__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.opts = opts

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.gru = MyGRUCell(hidden_size, hidden_size) #This

    def forward(self, inputs):
        """Forward pass of the encoder RNN.

        Arguments:
            inputs: Input token indexes across a batch for all time steps in the sequence. (batch_size x seq_len)

        Returns:
            annotations: The hidden states computed at each step of the input sequence. (batch_size x seq_len x hidden_size)
            hidden: The final hidden state of the encoder, for each sequence in a batch. (batch_size x hidden_size)
        """

        batch_size, seq_len = inputs.size()
        hidden = self.init_hidden(batch_size)

        encoded = self.embedding(inputs)  # batch_size x seq_len x hidden_size
        annotations = []

        for i in range(seq_len):
            x = encoded[:,i,:]  # Get the current time step, across the whole batch
            hidden = self.gru(x, hidden)
            annotations.append(hidden)

        annotations = torch.stack(annotations, dim=1)
        return annotations, hidden

    def init_hidden(self, bs):
        """Creates a tensor of zeros to represent the initial hidden states
        of a batch of sequences.

        Arguments:
            bs: The batch size for the initial hidden state.

        Returns:
            hidden: An initial hidden state of all zeros. (batch_size x hidden_size)
        """
        return utils.to_var(torch.zeros(bs, self.hidden_size), self.opts.cuda)

# question 5 
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()

        self.hidden_size = hidden_size

        # Create a two layer fully-connected network. Hint: Use nn.Sequential
        # hidden_size*2 --> hidden_size, ReLU, hidden_size --> 1

        # ------------
        # FILL THIS IN - START
        # ------------

        self.attention_network = nn.Sequential(
            nn.Linear(hidden_size*2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1))
       	
        # ------------
        # FILL THIS IN - END
        # ------------

        self.softmax = nn.Softmax(dim=1)

    def forward(self, queries, keys, values):
        """The forward pass of the additive attention mechanism.
        Arguments:
            queries: The current decoder hidden state. (batch_size x hidden_size)
            keys: The encoder hidden states for each step of the input sequence. (batch_size x seq_len x hidden_size)
            values: The encoder hidden states for each step of the input sequence. (batch_size x seq_len x hidden_size)
        Returns:
            context: weighted average of the values (batch_size x 1 x hidden_size)
            attention_weights: Normalized attention weights for each encoder hidden state. (batch_size x seq_len x 1)
            The attention_weights must be a softmax weighting over the seq_len annotations.
        """

        batch_size = keys.size(0)

        # You are free to follow the code template below, or do it a different way,
        # as long as the output is correct.
        # ------------
        # FILL THIS IN - START
        # ------------
        
        queries = queries.unsqueeze(1).expand_as(keys)
        inputs = torch.cat((queries, keys), 2)
        unnomarlized_attention = self.attention_network(inputs)
        attention_weights = self.softmax(unnomarlized_attention)
        context = torch.bmm(attention_weights.squeeze(2).unsqueeze(1), values)

        # ------------
        # FILL THIS IN - END
        # ------------

        return context, attention_weights


class AttentionDecoder(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(AttentionDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, hidden_size)

        self.rnn = MyGRUCell(input_size=hidden_size*2, hidden_size=hidden_size)
        self.attention = Attention(hidden_size=hidden_size)
        
        self.out = nn.Linear(hidden_size, vocab_size)

        
    def forward(self, inputs, annotations, hidden_init):
        """Forward pass of the attention-based decoder RNN.
        Arguments:
            inputs: Input token indexes across a batch for all the time step. (batch_size x decoder_seq_len)
            annotations: The encoder hidden states for each step of the input.
                         sequence. (batch_size x seq_len x hidden_size)
            hidden_init: The final hidden states from the encoder, across a batch. (batch_size x hidden_size)
        Returns:
            output: Un-normalized scores for each token in the vocabulary, across a batch for all the decoding time steps. (batch_size x decoder_seq_len x vocab_size)
            attentions: The stacked attention weights applied to the encoder annotations (batch_size x encoder_seq_len x decoder_seq_len)
        """
        
        batch_size, seq_len = inputs.size()
        embed = self.embedding(inputs)  # batch_size x seq_len x hidden_size

        hiddens = []
        attentions = []
        h_prev = hidden_init
        for i in range(seq_len):
            # You are free to follow the code template below, or do it a different way,
            # as long as the output is correct.

            # ------------
            # FILL THIS IN - START
            # ------------

            embed_current = embed[:, i, :]
            context, attention_weights = self.attention(h_prev, annotations, annotations)
            embed_and_context = torch.cat((context.squeeze(1), embed_current), 1)
            h_prev = self.rnn(embed_and_context, h_prev)
            
            # ------------
            # FILL THIS IN - END
            # ------------
            
            hiddens.append(h_prev)
            attentions.append(attention_weights)

        hiddens = torch.stack(hiddens, dim=1) # batch_size x seq_len x hidden_size
        attentions = torch.cat(attentions, dim=2) # batch_size x seq_len x seq_len
        
        output = self.out(hiddens)  # batch_size x seq_len x vocab_size
        return output, attentions



class NoAttentionDecoder(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(NoAttentionDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.GRUCell(input_size=hidden_size, hidden_size=hidden_size)
        self.out = nn.Linear(hidden_size, vocab_size)

#(decoder_inputs, encoder_annotations, encoder_hidden)
    def forward(self, inputs, annotations, hidden_init):
        """Forward pass of the non-attentional decoder RNN.

        Arguments:
            x: Input token indexes across a batch for a single time step. (batch_size x 1)
            h_prev: The hidden states from the previous step, across a batch. (batch_size x hidden_size)
            inputs: This is not used here. It just maintains consistency with the
                    interface used by the AttentionDecoder class.

        Returns:
            output: Un-normalized scores for each token in the vocabulary, across a batch. (batch_size x vocab_size)
            h_new: The new hidden states, across a batch. (batch_size x hidden_size)
            None: Used to maintain consistency with the interface of AttentionDecoder.
        """
        _, seq_len = inputs.size()
        embeded  = self.embedding(inputs) # batch_size x hidden_size
        hiddens = []
        h_prev = hidden_init
        for i in range(seq_len):
            encoded = embeded[:, i, :]
            h_prev = self.rnn(encoded, h_prev)
            hiddens.append(h_prev)

               # batch_size x 1 x hidden_size
        # encoded = encoded.squeeze(1)       # batch_size x hidden_size
        # h_new = self.rnn(encoded, h_prev)  # batch_size x hidden_size
        hiddens = torch.stack(hiddens, dim=1) # batch_size x seq_len x hidden_size
        output = self.out(hiddens)           # batch_size x vocab_size
        return output, None
