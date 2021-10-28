import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class POSModel(nn.Module):


    def __init__(self, src_model="roberta-base", trg_model="vblagoje/bert-english-uncased-finetuned-pos", freeze=False, max_len=64):
        super(POSModel, self).__init__()


        self.decoder = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=src_model)
        self.encoder = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=trg_model)

        self.max_len = max_len

        self.model = AutoModel.from_pretrained(pretrained_model_name_or_path=trg_model)

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

    

    def decode_input_ids(self, input_ids):

        text = []

        for each in input_ids:
            text.append(self.decoder.decode(
                    token_ids = each.tolist(),
                    skip_special_tokens = True,
                )
            )
        return text

    

    def encode_text(self, text):

        input_ids, attention_mask = torch.empty((0, self.max_len), dtype=torch.long, device=device), torch.empty((0, self.max_len), dtype=torch.long, device=device)
        for each in text:

            encoding = self.encoder.encode_plus(
                text=each,
                truncation=True,
                max_length=self.max_len,
                return_tensors="pt",
                return_attention_mask=True,
                padding="max_length",

            )

            # print(f'each_input.shape = {encoding["input_ids"].shape}')

            input_ids = torch.cat((input_ids.to(device), encoding['input_ids'].to(device)), dim=0).to(device)
            attention_mask = torch.cat((attention_mask.to(device), encoding['attention_mask'].to(device)), dim=0)

        return input_ids.to(device), attention_mask.to(device)
    

    def forward(self, input_ids, attention_mask=None):

        text = self.decode_input_ids(input_ids=input_ids)

        input_ids, attention_mask = self.encode_text(text=text)

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        cls = outputs[0][:, 0]

        return cls


class NERModel(nn.Module):
    

    def __init__(self, src_model="roberta-base", trg_model="dslim/bert-base-NER", freeze=False, max_len=64):
        super(NERModel, self).__init__()


        self.decoder = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=src_model)
        self.encoder = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=trg_model)

        self.max_len = max_len

        self.model = AutoModel.from_pretrained(pretrained_model_name_or_path=trg_model)

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

    

    def decode_input_ids(self, input_ids):

        text = []

        for each in input_ids:
            text.append(self.decoder.decode(
                    token_ids = each.tolist(),
                    skip_special_tokens = True,
                )
            )
        return text

    

    def encode_text(self, text):
    

        input_ids, attention_mask = torch.empty((0, self.max_len), dtype=torch.long, device=device), torch.empty((0, self.max_len), dtype=torch.long, device=device)
        for each in text:

            encoding = self.encoder.encode_plus(
                text=each,
                truncation=True,
                max_length=self.max_len,
                return_tensors="pt",
                return_attention_mask=True,
                padding="max_length",

            )
            # print(f'each_input.shape = {encoding["input_ids"].shape}')
            input_ids = torch.cat((input_ids.to(device), encoding['input_ids'].to(device)), dim=0).to(device)
            attention_mask = torch.cat((attention_mask.to(device), encoding['attention_mask'].to(device)), dim=0)

        return input_ids.to(device), attention_mask.to(device)
    

    def forward(self, input_ids, attention_mask=None):

        input_ids, attention_mask = self.encode_text(self.decode_input_ids(input_ids=input_ids))

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        cls = outputs[0][:, 0]

        return cls


class WordAttention(nn.Module):
    """
    Word-level attention module.
    """

    def __init__(self, vocab_size, embed_dim, gru_hidden_dim, gru_num_layers, att_dim, use_layer_norm, dropout):
        super(WordAttention, self).__init__()

        self.embeddings = nn.Embedding(vocab_size, embed_dim)

        # output (batch, hidden_size)
        self.gru = nn.GRU(embed_dim, gru_hidden_dim, num_layers=gru_num_layers, batch_first=True, bidirectional=True,
                          dropout=dropout)

        # NOTE MODIFICATION (FEATURES)
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(2 * gru_hidden_dim, elementwise_affine=True)
        self.dropout = nn.Dropout(dropout)

        # Maps gru output to `att_dim` sized tensor
        self.attention = nn.Linear(2 * gru_hidden_dim, att_dim)

        # Word context vector (u_w) to take dot-product with
        self.context_vector = nn.Linear(att_dim, 1, bias=False)

    def init_embeddings(self, embeddings):
        """
        Initialized embedding layer with pretrained embeddings.
        :param embeddings: embeddings to init with
        """
        # NOTE MODIFICATION (EMBEDDING)
        self.embeddings.weight = nn.Parameter(embeddings)

    def freeze_embeddings(self, freeze=False):
        """
        Set whether to freeze pretrained embeddings.
        :param freeze: True to freeze weights
        """
        # NOTE MODIFICATION (EMBEDDING)
        self.embeddings.weight.requires_grad = not freeze

    def forward(self, sents, sent_lengths):
        """
        :param sents: encoded sentence-level data; LongTensor (num_sents, pad_len, embed_dim)
        :param sent_lengths: sentence lengths; LongTensor (num_sents)
        :return: sentence embeddings, attention weights of words
        """
        # Sort sents by decreasing order in sentence lengths
        sent_lengths, sent_perm_idx = sent_lengths.sort(dim=0, descending=True)
        sents = sents[sent_perm_idx]

        sents = self.embeddings(sents)
        sents = self.dropout(sents)

        packed_words = pack_padded_sequence(sents, lengths=sent_lengths.tolist(), batch_first=True)

        # effective batch size at each timestep
        valid_bsz = packed_words.batch_sizes

        # Apply word-level GRU over word embeddings
        packed_words, _ = self.gru(packed_words)

        # NOTE MODIFICATION (FEATURES)
        if self.use_layer_norm:
            normed_words = self.layer_norm(packed_words.data)
        else:
            normed_words = packed_words

        # Word Attenton
        att = torch.tanh(self.attention(normed_words.data))
        att = self.context_vector(att).squeeze(1)

        val = att.max()
        att = torch.exp(att - val) # att.size: (n_words)

        # Restore as sentences by repadding
        att, _ = pad_packed_sequence(PackedSequence(att, valid_bsz), batch_first=True)

        # NOTE MODIFICATION (BUG) : attention score sum should be in dimension
        att_weights = att / torch.sum(att, dim=1, keepdim=True)

        # Restore as sentences by repadding
        sents, _ = pad_packed_sequence(packed_words, batch_first=True)

        # Compute sentence vectors
        sents = sents * att_weights.unsqueeze(2)
        sents = sents.sum(dim=1)

        # Restore the original order of sentences (undo the first sorting)
        _, sent_unperm_idx = sent_perm_idx.sort(dim=0, descending=False)
        sents = sents[sent_unperm_idx]

        # NOTE MODIFICATION BUG
        att_weights = att_weights[sent_unperm_idx]

        return sents, att_weights





class BloomNet2Classifier(nn.Module):
    
    def __init__(self, vocab_size, hidden_size=768, model_name="roberta-base", att_dim=768, num_layers=4,dropout=0.10, num_classes=6, max_len=64, fusion="concat"):

        super(BloomNet2Classifier, self).__init__()


        self.fusion = fusion

        if fusion=="concat":
            n = 3
        else:
            n = 1
        
        # pretrained transformer model as base
        self.base = AutoModel.from_pretrained(pretrained_model_name_or_path=model_name)

        self.pos = POSModel(src_model=model_name, trg_model="vblagoje/bert-english-uncased-finetuned-pos", max_len=max_len)

        self.ner = NERModel(src_model=model_name, trg_model="dslim/bert-base-NER", max_len=max_len)

        self.word_attention = WordAttention(
            vocab_size=vocab_size, 
            embed_dim=hidden_size, 
            gru_hidden_dim=hidden_size, 
            gru_num_layers=num_layers, 
            att_dim=att_dim, 
            use_layer_norm=True, 
            dropout=dropout,
        )
        
        # nn classifier on top of base model
        self.classifier = nn.Sequential(*[
            nn.Linear(in_features=5*768, out_features=768),
            nn.LeakyReLU(),
            nn.Linear(in_features=768, out_features=256),
            nn.LeakyReLU(),
            nn.Linear(in_features=256, out_features=num_classes),
        ])

    def forward(self, input_ids, attention_mask=None, _len=None):

        # last hidden states
        cls_generic = self.base(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device))[0][:, 0]

        cls_pos = self.pos(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device))

        cls_ner = self.ner(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device))

        # if self.fusion == "concat":
        #     x = torch.cat((cls_generic, cls_pos, cls_ner), dim=1).to(device)
        # else:
        #     x = cls_generic*cls_pos*cls_ner #torch.matmul(torch.matmul(cls_generic, cls_pos), cls_ner).to(device)



        sents, _ = self.word_attention(sents=input_ids, sent_lengths=_len)

        x = torch.cat((cls_generic, cls_pos, cls_ner, sents), dim=1)


        # pass it to nn classifier
        logits = self.classifier(x.to(device))


        return logits