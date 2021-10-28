import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
class LSTMClassifier(nn.Module):
    
    def __init__(self, vocab_size, hidden_size=768, dropout=0.10, num_layers=1, bidirectional=True, num_classes=6):
        """LSTM baseline for bloom_taxonomy classification

        Args:
            vocab_size (int): size of vocabulary
            hidden_size (int): hidden size, same as embedding size
            dropout (float, optional): dropout rat. Defaults to 0.10.
            n_layers (int, optional): how many layers in LSTM. Defaults to 1.
            bidirectional (bool, optional): whether bidirectional or not. Defaults to True.
            num_classes (int, optional): no of classes in training set. Defaults to 6.
        """
        
        super(LSTMClassifier, self).__init__()
        
        
        # embedding layer to get embeddings of tokenized input
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=hidden_size,
            padding_idx=0,
        )
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional
        )
        
        # classification module
        self.classifier = nn.Sequential(*[
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.LeakyReLU(),
            nn.Linear(in_features=hidden_size, out_features=num_classes)
        ])
        
    def forward(self, input_ids, _len, attention_mask=None):
        
        """forward function of LSTM model.

        Args:
            input_ids (Tensor): shape = [batch_size, max_len]
            _len (Tensor, optional): shape = [batch]. Defaults to None.
            attention_mask ([type], optional): shape = [batch_size, max_len]. Defaults to None.
        """
        embedding =  self.embedding(input_ids)
        
        packed_input = pack_padded_sequence(input=embedding, lengths=_len.to(torch.int64).cpu(), batch_first=True, enforce_sorted=False)
        
        packed_output, (ht, ct) = self.lstm(packed_input)
        
        # packed_output, _ = pad_packed_sequence(packed_output, batch_first=True)
        
        # take mean of last hidden states
        # features = packed_output.mean(dim=1).squeeze()

        
        features = ht.permute(1, 0, 2).mean(1)  # [2, batch_size, hidden_size] -> [batch, hidden_size]
        
        logits = self.classifier(features)
        
        return logits
        # return packed_output, (ht, ct)

        
        
        