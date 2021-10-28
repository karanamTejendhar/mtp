import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


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


class BloomNetClassifier(nn.Module):
    
    def __init__(self, model_name="roberta-base", num_classes=6, max_len=64, fusion="concat"):

        super(BloomNetClassifier, self).__init__()


        self.fusion = fusion

        if fusion=="concat":
            n = 3
        else:
            n = 1
        
        # pretrained transformer model as base
        self.base = AutoModel.from_pretrained(pretrained_model_name_or_path=model_name)

        self.pos = POSModel(src_model=model_name, trg_model="vblagoje/bert-english-uncased-finetuned-pos", max_len=max_len)

        self.ner = NERModel(src_model=model_name, trg_model="dslim/bert-base-NER", max_len=max_len)
        
        # nn classifier on top of base model
        self.classifier = nn.Sequential(*[
            nn.Linear(in_features=n*768, out_features=768),
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

        if self.fusion == "concat":
            x = torch.cat((cls_generic, cls_pos, cls_ner), dim=1).to(device)
        else:
            x = cls_generic*cls_pos*cls_ner #torch.matmul(torch.matmul(cls_generic, cls_pos), cls_ner).to(device)

        # pass it to nn classifier
        logits = self.classifier(x.to(device))


        return logits
