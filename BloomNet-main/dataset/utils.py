import json
import os

def get_label2idx():
    
    with open( os.path.join(os.path.dirname(os.path.abspath(__file__)), "label-dictionaries", "label2idx.json"), "r" ) as file:
        label2idx = json.load( file )
    return label2idx


def get_idx2label():
    
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "label-dictionaries","idx2label.json"), "r") as file:
        idx2label = json.load( file )
        return idx2label



if __name__ == "__main__":
    print(f'\nLabel2Idx = {get_label2idx()}\n')
    print(f'\Idx2Label = {get_idx2label()}\n')
    
    
    