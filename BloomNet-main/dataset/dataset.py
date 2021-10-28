import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from .utils import get_label2idx


class BloomDataset(Dataset):
    
    def __init__(self, tokenizer, file_name, text_field="Text", label_field="Label", max_len=64):
        
        self.tokenizer = tokenizer
        
        self.data = load_dataset("csv", data_files=file_name)['train']
        
        
        self.text = self.data[text_field]
        self.label = self.data[label_field]
        
        self.label2idx = get_label2idx()
        
        self.max_len = max_len
        
        
    def __len__(self):
        return len(self.text)
    
    def __getitem__(self, index):
        
        text = self.text[index]
        label = self.label[index]
        
        target = self.label2idx[label]
        
        
        
        input_encoding = self.tokenizer.encode_plus(
            text=text,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
            return_attention_mask=True,
            padding="max_length",
        )
        
        _len = len(self.tokenizer.tokenize(text))
        if _len>64:
            _len = 64
        
        return {
            "text":text,
            "input_ids":input_encoding['input_ids'].squeeze(),
            "attention_mask":input_encoding['attention_mask'].squeeze(),
            "_len":_len,
            "label":label,
            "target":torch.tensor([target], dtype=torch.long)
        }