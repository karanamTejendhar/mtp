import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn import functional as F



class LSTMAttnClassifier(nn.Module):
    
    def __init__(self,vocab_size, hidden_size=768, num_classes=6):
        super(LSTMAttnClassifier, self).__init__()
        
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=hidden_size,
            padding_idx=0,
        )
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True,
        )
        
        # classification module
        self.classifier = nn.Sequential(*[
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.LeakyReLU(),
            nn.Linear(in_features=hidden_size, out_features=num_classes)
        ])
        
    def attention_net(self, lstm_output, final_state):
        hidden =  final_state.squeeze()
        attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return new_hidden_state
    
    
    
    def forward(self, input_ids, attention_mask=None, _len=None):
        
        embedding = self.embedding(input_ids)
        
        packed_input = pack_padded_sequence(input=embedding, lengths=_len.to(torch.int64).cpu(), batch_first=True, enforce_sorted=False)
         
        output, (ht, ct) = self.lstm(packed_input)
        
        packed_output, _ = pad_packed_sequence(output, batch_first=True)
        
        attn_outputs = self.attention_net(packed_output, ht)
        
        logits = self.classifier(attn_outputs)
        
        return logits
    
        