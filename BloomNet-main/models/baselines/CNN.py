import torch
import torch.nn as nn
import torch.nn.functional as F

#Source: https://github.com/RaffaeleGalliera/pytorch-cnn-text-classification/edit/master/model.py

class CNNClassifier(nn.Module):
    def __init__(self, vocab_size, hidden_size=768, n_filters=400, filter_sizes=[2, 3, 4], num_classes=6,
                 dropout=0.10, pad_idx=0):
        
        super(CNNClassifier, self).__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=hidden_size,
            padding_idx=0,
        )
        

        # The in_channels argument is the number of "channels" in your image going into the convolutional layer.
        # In actual images this is usually 3 (one channel for each of the red, blue and green channels),
        # however when using text we only have a single channel, t
        # he text itself. The out_channels is the number of filters and the kernel_size is the size of the filters.
        # Each of our kernel_sizes is going to be [n x emb_dim] where $n$ is the size of the n-grams.

        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1,
                      out_channels=n_filters,
                      kernel_size=(fs, hidden_size))
            for fs in filter_sizes
        ])

        self.fc = nn.Linear(len(filter_sizes) * n_filters, num_classes)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask=None, _len=None):
        
        embedded = self.embedding(input_ids)

        # embedded = [batch size, sent len, emb dim]

        # In PyTorch, RNNs want the input with the batch dimension second, whereas CNNs want the batch dimension first
        # - we do not have to permute the data here as we have already set batch_first = True in our TEXT field.
        # We then pass the sentence through an embedding layer to get our embeddings.
        # The second dimension of the input into a nn. Conv2d layer must be the channel dimension.
        # As text technically does not have a channel dimension,
        # we unsqueeze our tensor to create one.
        # This matches with our in_channels=1 in the initialization of our convolutional layers.

        embedded = embedded.unsqueeze(1)

        # embedded = [batch size, 1, sent len, emb dim]

        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]

        # conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]

        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]

        # pooled_n = [batch size, n_filters]

        cat = self.dropout(torch.cat(pooled, dim=1))

        # cat = [batch size, n_filters * len(filter_sizes)]

        return self.fc(cat)