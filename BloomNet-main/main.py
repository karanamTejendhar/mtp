import os
import gc
import json
import shutil
import torch
import argparse
from tqdm import tqdm
import pandas as pd
import numpy as np
from config import config
from evaluation import evaluate
from utils import *
import pytorch_lightning as pl
from transformers import AutoTokenizer
from Trainer import LightningModel
from dataset.loader import get_loaders


import warnings
warnings.filterwarnings('ignore')

    
if __name__=="__main__":

    seed(42)
    

    parser = argparse.ArgumentParser("main.py", formatter_class=argparse.ArgumentDefaultsHelpFormatter)


    parser.add_argument("--model", type=str, default="bloomnet",
                        help="model to train")
    
    # parser.add_argument("--model-name", type=str, default="roberta-base",
    #                     help="model to train")
    
    parser.add_argument("--batch", type=int, default=32,
                        help="batch size for training")

    parser.add_argument("--max_len", type=int, default=64,
                        help="batch size for training")
    
    parser.add_argument("--folds", type=int, default=5,
                        help="number of folds in cross validation")
    
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="learning rate")
    
    parser.add_argument("--epochs", type=int, default=20,
                        help="number of epoch to train for")

    parser.add_argument("--fusion", type=str, default="concat",
                        help="only applicable for bloomnet")

    args = parser.parse_args()
    
    
    # update the config from CLI arguments 
    
    config['data']['batch'] = args.batch
    config['data']['folds'] = args.folds
    config['training']['lr'] = args.lr
    config['training']['epochs'] = args.epochs
    config['data']['max_len'] = args.max_len
    config['data']['model'] = args.fusion
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # device = torch.device("cpu")
    
    
    

    # results = {}


    #model_name = args.__dict__['model']#'bert-base-uncased'# 'bert-base-uncased'
    
    model = args.model
    # model_name = args.model_name

    tokenizer = AutoTokenizer.from_pretrained('roberta-base', usefast=True, use_lower_case=True)
    loaders = get_loaders(tokenizer=tokenizer, config=config['data'])
        
    
    CWD = os.getcwd()
    
    outputs_path = os.path.join(CWD, "Outputs")
    os.makedirs(outputs_path, exist_ok=True)

    accuracy_scores = {"dataset1":[], "dataset2":[]}
    f1_scores  = {"dataset1":[], "dataset2":[]}
        
    
    for fold in tqdm(loaders):
        
        
        lm = LightningModel(model_name=model, vocab_size=tokenizer.vocab_size, config=config)
        
        # checkpoints and results path
        ckpt_path = os.path.join(CWD, "Outputs", "Checkpoints", model, fold)
        os.makedirs(ckpt_path, exist_ok=True)
        results_path = os.path.join(CWD, "Outputs", "Results", model, fold)
        os.makedirs(results_path, exist_ok=True)
        
        
        # create trainer object
        trainer, checkpoint_callback = create_trainer(
            config=config,
            run_name=model+"-"+str(fold),
            ckpt_path=ckpt_path,
        )
        
        # training model
        trainer.fit(
            model=lm,
            train_dataloader=loaders[fold]['train'],
            val_dataloaders=loaders[fold]['valid'],
        )
        
        # load best checkpoints

        trainer.test(
            model=lm,
            test_dataloaders=loaders[fold]['test1'],
            verbose=True,
            # ckpt_path="best",
        )
        
        # # lm = LightningModel.load_from_checkpoint(checkpoint_callback.best_model_path)

        # lm.load_state_dict(torch.load(checkpoint_callback.best_model_path)['state_dict'])
        
        # get the results and save them into directory
        # for dataset1 and dataset2
        ground_truth_1, predicted_class_1, f1_1, accuracy_1, cr_1 = evaluate(model=lm, loader=loaders[fold]['test1'], device=device)
        ground_truth_2, predicted_class_2, f1_2, accuracy_2, cr_2 = evaluate(model=lm, loader=loaders[fold]['test2'], device=device)
        
        
        
        # store f1 and acc for each fold in a list
        accuracy_scores['dataset1'].append(accuracy_1)
        accuracy_scores['dataset2'].append(accuracy_2)
        
        f1_scores['dataset1'].append(f1_1)
        f1_scores['dataset2'].append(f1_2)
        
        
        # save classification report and predictions 
        with open(os.path.join(results_path, "dataset1"+str(fold)+".txt"), "w") as file:
            file.write(cr_1)
        with open(os.path.join(results_path, "dataset2"+str(fold)+".txt"), "w") as file:
            file.write(cr_2)


        # save predictions
        df_1 =  df = pd.DataFrame(
            data={
                "Ground_Truth":ground_truth_1,
                "Predicted_Class":predicted_class_1
            }
        )
        df_1.to_csv(os.path.join(results_path, "prediction_dataset1"+str(fold)+".csv"), index=False)
        
        df_2 =  df = pd.DataFrame(
            data={
                "Ground_Truth":ground_truth_2,
                "Predicted_Class":predicted_class_2
            }
        )
        df_2.to_csv(os.path.join(results_path, "prediction_dataset2"+str(fold)+".csv"), index=False)
        
            
        del lm
        del df_1
        del df_2
        del trainer
    
        torch.cuda.empty_cache()
        gc.collect()


        #del checkpoints
        shutil.rmtree(ckpt_path)
        
        
        # break # only for debugging
        
        
      
      
    with open(os.path.join(CWD, "Outputs", "Results", model, "results.json"), "w") as file:
        json.dump({
            "dataset1":{
                "accuracy":{
                    "mean":round(np.mean(accuracy_scores['dataset1']), 4)*100,
                    "std":round(np.std(accuracy_scores['dataset1']), 4)*100,
                },
                "macro-f1":{
                    "mean":round(np.mean(f1_scores['dataset1']), 4)*100,
                    "std":round(np.std(f1_scores['dataset1']), 4)*100,
                },
            },
            "dataset2":{
                "accuracy":{
                    "mean":round(np.mean(accuracy_scores['dataset2']), 4)*100,
                    "std":round(np.std(accuracy_scores['dataset2']), 4)*100,
                },
                "macro-f1":{
                    "mean":round(np.mean(f1_scores['dataset2']), 4)*100,
                    "std":round(np.std(f1_scores['dataset2']), 4)*100,
                },
            }
            
        }, file) 
        
        
        
    del f1_scores
    del accuracy_scores
    gc.collect()
    
    print("Run Successfull!")