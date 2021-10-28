# model.py

# Source: https://github.com/AnubhavGupta3377/Text-Classification-Models-Pytorch/edit/master/Model_Seq2Seq_Attention/model.py

import torch
from torch import nn
import numpy as np
from torch.nn import functional as F
from utils import *

class Seq2SeqAttnClassifier(nn.Module):
    def __init__(self, vocab_size, hidden_size=768, bidirectional=True, num_layers=2, dropout=0.10, num_classes=6):
        super(Seq2SeqAttnClassifier, self).__init__()
        
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        
        # Embedding Layer
        self.embeddings = nn.Embedding(vocab_size, hidden_size)
        # self.embeddings.weight = nn.Parameter(word_embeddings, requires_grad=False)
        
        # Encoder RNN
        self.lstm = nn.LSTM(input_size = hidden_size,
                            hidden_size = hidden_size,
                            num_layers = num_layers,
                            bidirectional = bidirectional,
                            )
        
        # Dropout Layer
        self.dropout = nn.Dropout(dropout)
        
        # Fully-Connected Layer
        self.fc = nn.Linear(
            hidden_size * (1+bidirectional) * 2,
            num_classes
        )
        
                
    def apply_attention(self, rnn_output, final_hidden_state):
        '''
        Apply Attention on RNN output
        
        Input:
            rnn_output (batch_size, seq_len, num_directions * hidden_size): tensor representing hidden state for every word in the sentence
            final_hidden_state (batch_size, num_directions * hidden_size): final hidden state of the RNN
            
        Returns:
            attention_output(batch_size, num_directions * hidden_size): attention output vector for the batch
        '''
        hidden_state = final_hidden_state.unsqueeze(2)
        attention_scores = torch.bmm(rnn_output, hidden_state).squeeze(2)
        soft_attention_weights = F.softmax(attention_scores, 1).unsqueeze(2) #shape = (batch_size, seq_len, 1)
        attention_output = torch.bmm(rnn_output.permute(0,2,1), soft_attention_weights).squeeze(2)
        return attention_output
        
    def forward(self, input_ids, attention_mask=None, _len=None):
        
        x = input_ids.permute(1, 0)
        # x.shape = (max_sen_len, batch_size)
        embedded_sent = self.embeddings(x)
        # embedded_sent.shape = (max_sen_len=20, batch_size=64,embed_size=300)

        ##################################### Encoder #######################################
        lstm_output, (h_n,c_n) = self.lstm(embedded_sent)
        # lstm_output.shape = (seq_len, batch_size, num_directions * hidden_size)
        
        # print(f'lstm_output.shape = {lstm_output.shape} | h_n.shape = {h_n.shape} | input_ids.shape = {input_ids.shape}')
        # Final hidden state of last layer (num_directions, batch_size, hidden_size)
        batch_size = h_n.shape[1]
        h_n_final_layer = h_n.view(self.num_layers,
                                   self.bidirectional + 1,
                                   batch_size,
                                   self.hidden_size)[-1,:,:,:]
        
        ##################################### Attention #####################################
        # Convert input to (batch_size, num_directions * hidden_size) for attention
        final_hidden_state = torch.cat([h_n_final_layer[i,:,:] for i in range(h_n_final_layer.shape[0])], dim=1)
        
        attention_out = self.apply_attention(lstm_output.permute(1,0,2), final_hidden_state)
        # Attention_out.shape = (batch_size, num_directions * hidden_size)
        
        #################################### Linear #########################################
        concatenated_vector = torch.cat([final_hidden_state, attention_out], dim=1)
        final_feature_map = self.dropout(concatenated_vector) # shape=(batch_size, num_directions * hidden_size)
        logits = self.fc(final_feature_map)
        return logits
    