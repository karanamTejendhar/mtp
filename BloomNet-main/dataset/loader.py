import os
from torch.utils.data import DataLoader, DataLoader, SubsetRandomSampler
from .dataset import BloomDataset
from sklearn.model_selection import KFold, StratifiedKFold


def get_loaders(tokenizer, config):
    
    path_1 = os.path.join(os.getcwd(), "data", "BloomTaxonomy.csv")
    path_2 = os.path.join(os.getcwd(), "data", "BloomTaxonomy2.csv")
    
    dataset1 = BloomDataset(
        tokenizer=tokenizer,
        file_name=path_1,
        text_field=config['Text'], 
        label_field=config['Label'], 
        max_len=config['max_len']
    )
    
    dataset2 = BloomDataset(
        tokenizer=tokenizer,
        file_name=path_2,
        text_field=config['Text'], 
        label_field=config['Label'], 
        max_len=config['max_len']
    )
    
    # create dataset2 loader
    
    test2_loader = DataLoader(dataset=dataset2, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'], drop_last=True)
    
    loaders = {}
    
    kfold  = KFold(n_splits=config['folds'], shuffle=True, random_state=42)

    
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset1)):
        
        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = SubsetRandomSampler(train_ids)
        test_subsampler = SubsetRandomSampler(test_ids)
        
        # Define data loaders for training and testing data in this fold
        trainloader =DataLoader(dataset=dataset1, batch_size=config['batch_size'], sampler=train_subsampler, num_workers=config['num_workers'])#, drop_last=True
        testloader = DataLoader(dataset=dataset1, batch_size=config['batch_size'], sampler=test_subsampler, num_workers=config['num_workers'])#, drop_last=True
        

        loaders.update({
            "fold"+str(fold):{
                "train":trainloader,
                "valid":testloader,
                "test1":testloader,
                "test2":test2_loader,
            }
        }
    )
    
    
    return loaders
    
    
    


if __name__ == '__main__':
    
    
    print(os.getcwd())
    print(__file__)