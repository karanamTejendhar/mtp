# utils
import torch

# models
import torch.nn as nn

# training and evaluation
import wandb
import pytorch_lightning as pl
import torch.nn.functional as F
import torch.optim as optim
import geoopt
from sklearn.metrics import accuracy_score, f1_score
from models.baselines.LSTM import LSTMClassifier
from models.baselines.RoBERTa import RoBERTaClassifier
# from models.baselines.DistilRoBERTa import DistilRoBERTaClassifier
from models.baselines.LSTMAttn import LSTMAttnClassifier
from models.baselines.CNN import CNNClassifier
from models.baselines.VDCNN import VDCNNClassifier
from models.baselines.RCNN import RCNNClassifier
from models.baselines.SelfAttn import SelfAttnClassifier
from models.baselines.Seq2SeqAttn import Seq2SeqAttnClassifier
from models.baselines.HAN import HANClassifier
from models.BloomNet import BloomNetClassifier
from models.Bloomnet_han_ling import BloomNet2Classifier
# from models.model import Model

class LightningModel(pl.LightningModule):

    def __init__(self, model_name, vocab_size, config):

        super(LightningModel, self).__init__()

        self.config = config
        self.model_name = model_name
        
        if model_name == "model" or model_name=="bloomnet" or model_name == None:
            self.model = BloomNetClassifier(
                model_name="roberta-base",
                num_classes=config['data']['num_classes'],
                max_len=config['data']['max_len'],
                fusion=config['model']['fusion'],
            )
        elif model_name == "bloomnet2" or model_name == "model2":
            self.model = BloomNet2Classifier(
                vocab_size=vocab_size,
            )

        # joint model

        elif model_name == "han":
            self.model = HANClassifier(
                num_classes=config['data']['num_classes'],
                vocab_size=vocab_size,
                embed_dim=config['model']['hidden_size'],
                word_gru_hidden_dim=config['model']['hidden_size'],
                sent_gru_hidden_dim=config['model']['hidden_size'],
                word_gru_num_layers=config['model']['num_layers'],
                sent_gru_num_layers=config['model']['num_layers'],
                word_att_dim=config['model']['hidden_size'],
                sent_att_dim=config['model']['hidden_size'],
                use_layer_norm=True,
                dropout=config['model']['dropout']

            )

        elif model_name == 'lstm':
            
            self.model = LSTMClassifier(
                vocab_size=vocab_size,
                hidden_size=config['model']['hidden_size'],
                num_layers=config['model']['num_layers'],
            )
            
        elif model_name == 'lstm_attn':
            self.model = LSTMAttnClassifier(
                vocab_size=vocab_size
            )
        elif model_name == 'rcnn':
            self.model = RCNNClassifier(
                vocab_size=vocab_size,
                batch_size=config['data']['batch_size'], 
                output_size=config['data']['num_classes'], 
                hidden_size=config['model']['hidden_size'], 
                embedding_length=config['model']['hidden_size'],  
            )
        elif model_name == "cnn":
            self.model = CNNClassifier(
                vocab_size=vocab_size,
            )
        elif model_name == "vdcnn":
            self.model = VDCNNClassifier(
                vocab_size=vocab_size,
                hidden_size=config['model']['hidden_size'],
                num_classes=config['data']['num_classes']
            )
            
        elif model_name == "self_attn":
            self.model = SelfAttnClassifier(
                vocab_size=vocab_size,
                hidden_size=config['model']['hidden_size'],
                num_classes=config['data']['num_classes']
            )
            
        elif model_name == "seq2seq_attn":
            self.model = Seq2SeqAttnClassifier(
                vocab_size=vocab_size,
                hidden_size=config['model']['hidden_size'],
                bidirectional=config['model']['bidirectional'],
                num_layers=config['model']['num_layers'],
                dropout=config['model']['dropout'],
                num_classes=config['data']['num_classes'],
            )
        elif model_name=="roberta-base":
            self.model = RoBERTaClassifier()
            
        elif model_name=="distilroberta-base":
            
            self.model = DistilRoBERTaClassifier()
            
        else:
            self.model = RoBERTaClassifier()
            


    def forward(self, input_ids, attention_mask=None, _len=None):
        logits  = self.model(input_ids=input_ids, attention_mask=attention_mask, _len=_len)
        return logits

    def configure_optimizers(self):
        # euclidean optimizer vs HyRNN
        if self.model_name == "model" or self.model_name=="hyp" or self.model_name == "hyrnn":
            return geoopt.optim.RiemannianAdam(
                self.parameters(),
                lr=self.config['training']['lr'],
                stabilize=10,
            )
        else: return optim.AdamW(params=self.parameters(), lr=self.config['training']['lr'])

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, targets = batch['input_ids'], batch['attention_mask'], batch['target'].squeeze()
        logits = self(input_ids=input_ids, attention_mask=attention_mask, _len=batch['_len'])
        loss = F.cross_entropy(logits, targets)
        acc = accuracy_score(targets.cpu(), logits.argmax(dim=-1).cpu())
        f1 = f1_score(targets.cpu(), logits.argmax(dim=-1).cpu(), average=self.config['training']['average'])
        wandb.log({"loss":loss, "accuracy":acc, "f1":f1})
        return {"loss":loss, "accuracy":acc, "f1":f1}

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, targets = batch['input_ids'], batch['attention_mask'], batch['target'].squeeze()
        logits = self(input_ids=input_ids, attention_mask=attention_mask, _len=batch['_len'])
        loss = F.cross_entropy(logits, targets)
        acc = accuracy_score(targets.cpu(), logits.argmax(dim=-1).cpu())
        f1 = f1_score(targets.cpu(), logits.argmax(dim=-1).cpu(), average=self.config['training']['average'])
        wandb.log({"val_loss":loss, "val_accuracy":torch.tensor([acc]), "val_f1":torch.tensor([f1])})
        return {"val_loss":loss, "val_accuracy":torch.tensor([acc]), "val_f1":torch.tensor([f1])}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_accuracy'] for x in outputs]).mean()
        avg_f1 = torch.stack([x['val_f1'] for x in outputs]).mean()
        return {"val_loss":avg_loss, "val_accuracy":avg_acc, "val_f1":avg_f1,}

    def test_step(self, batch, batch_idx):
        input_ids, attention_mask, targets = batch['input_ids'], batch['attention_mask'], batch['target'].squeeze()
        logits = self(input_ids=input_ids, attention_mask=attention_mask, _len=batch['_len'])
        loss = F.cross_entropy(logits, targets)
        acc = accuracy_score(targets.cpu(), logits.argmax(dim=-1).cpu())
        f1 = f1_score(targets.cpu(), logits.argmax(dim=-1).cpu(), average=self.config['training']['average'])
        return {"test_loss":loss, "test_accuracy":torch.tensor([acc]), "test_f1":torch.tensor([f1])}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['test_accuracy'] for x in outputs]).mean()
        avg_f1 = torch.stack([x['test_f1'] for x in outputs]).mean()
        return {"test_loss":avg_loss, "test_accuracy":avg_acc, "test_f1":avg_f1}