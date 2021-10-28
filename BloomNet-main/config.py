config = {
    "data":{
        "Text":"Text",
        "Label":"Label",
        "num_classes":6,
        "max_len":64,
        "batch_size":4,
        "num_workers":4,
        "folds":5,
    },
    
    "model":{
        "bidirectional":True,
        "fusion":"concat",
        "hidden_size":768,
        "num_layers":4, # for lstm like models but will be same for all models 
        "dropout":0.10, # for lstm like models but will be same for all models
    },

    "training":{
        "epochs":20,
        "lr":2e-5, 
        "average":"macro",
    },
    
    "callback":{
        "monitor":"val_accuracy",
        "min_delta":0.001,
        "patience":5,
        "precision":32,
        "project":"bloom-taxonomy-classification",
    }


}