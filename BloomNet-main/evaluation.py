
# import torch
from sklearn.metrics import f1_score, accuracy_score, classification_report
from dataset.utils import get_label2idx
from tqdm import tqdm


def evaluate(model, loader, device):

    true_label = []
    pred_label = []

    ground_truth = []
    predicted_class = []
    
    label2idx = get_label2idx()
    idx2label = dict(zip(label2idx.values(), label2idx.keys()))
    


    model.eval()
    model = model.to(device)

    for batch in tqdm(loader):

        outputs = model(input_ids=batch['input_ids'].to(device), attention_mask=batch['attention_mask'].to(device), _len=batch['_len']).argmax(dim=-1)

        pred_label += outputs.cpu().detach().tolist()
        true_label += batch['target'].cpu().tolist()
    
        ground_truth += batch['label']
        predicted_class += [idx2label[each] for each in outputs.cpu().detach().tolist()]

        # limit to evaluate only on one batch comment it before final run
        # break

    f1 = f1_score(y_true=true_label, y_pred=pred_label, average='macro')
    accuracy = accuracy_score(y_true=true_label, y_pred=pred_label)
    cr = classification_report(y_true=ground_truth, y_pred=predicted_class, digits=4)
    

    return ground_truth, predicted_class, f1, accuracy, cr