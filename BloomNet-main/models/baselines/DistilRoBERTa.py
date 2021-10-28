import torch.nn as nn
from transformers import AutoModel, AutoConfig

class DistilRoBERTaClassifier(nn.Module):

    def __init__(self, model_name="distilroberta-base", num_classes=6):
        super(DistilRoBERTaClassifier, self).__init__()
        
        # pretrained transformer model as base
        self.base = AutoModel.from_pretrained(pretrained_model_name_or_path=model_name)
        
        

        # nn classifier on top of base model
        self.classifier = nn.Sequential(*[
            nn.Linear(in_features=768, out_features=256),
            nn.LeakyReLU(),
            nn.Linear(in_features=256, out_features=num_classes),
        ])

    def forward(self, input_ids, attention_mask=None, _len=None):

        # last hidden states
        outputs = self.base(input_ids=input_ids, attention_mask=attention_mask)
        
        # cls token from last hidden states
        pooler = outputs[0][:, 0]

        # pass it to nn classifier
        logits = self.classifier(pooler)


        return logits