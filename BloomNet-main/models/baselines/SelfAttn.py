# _*_ coding: utf-8 _*_

# Source: https://github.com/prakashpandey9/Text-Classification-Pytorch/blob/master/models/selfAttention.py
# Citation: https://github.com/prakashpandey9/Text-Classification-Pytorch

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

class SelfAttnClassifier(nn.Module):
    def __init__(self, vocab_size, hidden_size=768, num_classes=6):
        super(SelfAttnClassifier, self).__init__()

        """
        Arguments
        ---------
        num_classes : 2 = (pos, neg)
        hidden_sie : Size of the hidden_state of the LSTM
        vocab_size : Size of the vocabulary containing unique words
        hidden_size : Embeddding dimension of GloVe word embeddings
        weights : Pre-trained GloVe word_embeddings which we will use to create our word_embedding look-up table 
        
        --------
        
        """

        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        # self.word_embeddings.weights = nn.Parameter(weights, requires_grad=False)
        self.dropout = 0.8
        self.bilstm = nn.LSTM(hidden_size, hidden_size, dropout=self.dropout, bidirectional=True)
        # We will use da = 350, r = 30 & penalization_coeff = 1 as per given in the self-attention original ICLR paper
        self.W_s1 = nn.Linear(2*hidden_size, 350)
        self.W_s2 = nn.Linear(350, 30)
        self.fc_layer = nn.Linear(30*2*hidden_size, 2000)
        self.label = nn.Linear(2000, num_classes)

    def attention_net(self, lstm_output):

        """
        Now we will use self attention mechanism to produce a matrix embedding of the input sentence in which every row represents an
        encoding of the inout sentence but giving an attention to a specific part of the sentence. We will use 30 such embedding of 
        the input sentence and then finally we will concatenate all the 30 sentence embedding vectors and connect it to a fully 
        connected layer of size 2000 which will be connected to the output layer of size 2 returning logits for our two classes i.e., 
        pos & neg.
        Arguments
        ---------
        lstm_output = A tensor containing hidden states corresponding to each time step of the LSTM network.
        ---------
        Returns : Final Attention weight matrix for all the 30 different sentence embedding in which each of 30 embeddings give
                    attention to different parts of the input sentence.
        Tensor size : lstm_output.size() = (batch_size, num_seq, 2*hidden_size)
                        attn_weight_matrix.size() = (batch_size, 30, num_seq)
        """
        attn_weight_matrix = self.W_s2(F.tanh(self.W_s1(lstm_output)))
        attn_weight_matrix = attn_weight_matrix.permute(0, 2, 1)
        attn_weight_matrix = F.softmax(attn_weight_matrix, dim=2)

        return attn_weight_matrix

    def forward(self, input_ids, attention_mask=None, _len=None):
        
        input_sentences = input_ids

        
        """ 
        Parameters
        ----------
        input_sentence: input_sentence of shape = (batch_size, num_sequences)
        
        Returns
        -------
        Output of the linear layer containing logits for pos & neg class.
        
        """

        input = self.word_embeddings(input_sentences)
        input = input.permute(1, 0, 2)

        output, (h_n, c_n) = self.bilstm(input)
        output = output.permute(1, 0, 2)
        # output.size() = (batch_size, num_seq, 2*hidden_size)
        # h_n.size() = (1, batch_size, hidden_size)
        # c_n.size() = (1, batch_size, hidden_size)
        attn_weight_matrix = self.attention_net(output)
        # attn_weight_matrix.size() = (batch_size, r, num_seq)
        # output.size() = (batch_size, num_seq, 2*hidden_size)
        hidden_matrix = torch.bmm(attn_weight_matrix, output)
        # hidden_matrix.size() = (batch_size, r, 2*hidden_size)
        # Let's now concatenate the hidden_matrix and connect it to the fully connected layer.
        fc_out = self.fc_layer(hidden_matrix.view(-1, hidden_matrix.size()[1]*hidden_matrix.size()[2]))
        logits = self.label(fc_out)
        # logits.size() = (batch_size, num_classes)

        return logits