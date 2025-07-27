import torch

import torch.nn.functional as F
from sklearn.preprocessing import label_binarize
from sklearn.metrics import average_precision_score
from tqdm import tqdm
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score,balanced_accuracy_score, precision_score, recall_score
from sklearn.metrics import roc_curve, roc_auc_score, RocCurveDisplay
import os
def do_test(test_dataloader, model, device="cuda", criterion=torch.nn.CrossEntropyLoss()):
    test_loss = 0.0
    all_preds = []
    all_labels = []
    test_losses = []
    output = []
    f1_scores=[]
    with torch.no_grad():
        for test_batch in tqdm(test_dataloader, total=len(test_dataloader)):
            inputs = test_batch[0] #.to(device)
                    
            inputs = inputs.to(device)
            labels = torch.squeeze(test_batch[1].to(device))
           # print(labels)
    
            outputs,_ = model(inputs) 
            output.extend(F.softmax(outputs).cpu().numpy()[:, 1])
          #  print(outputs)
    
            
            test_loss += criterion(outputs, labels)
            
            preds = torch.argmax(outputs, dim=1) 
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    test_loss /= len(test_dataloader)
    test_losses.append(test_loss)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    acc = accuracy_score(all_labels, all_preds)
    ba = balanced_accuracy_score(all_labels, all_preds)
    f1_scores.append(f1)
    confusion_matrix_sc = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = confusion_matrix_sc.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    all_labels_bin = label_binarize(all_labels, classes=[0, 1]).ravel()
    
    auc_roc = roc_auc_score(all_labels_bin, output)
    auc_pr = average_precision_score(all_labels_bin, output)
    metrics = {
        'Loss': test_loss / len(test_dataloader.dataset),
        'Accuracy': acc,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'Precision': precision,
        'AUC-ROC': auc_roc,
        'AUC-PR': auc_pr,
        'Balanced Accuracy': ba,
        'F1 Score': f1,
    }
    print("Test loss{b:.3f}, F1 {c:.3f}, Acc {d:.3f}, BA {e:.3f}, precision {f:.3f}, recall {g:.3f}".format( b=test_loss, c=f1, d = acc, e= ba, f=precision, g=recall))
    print(f"cm{confusion_matrix_sc}")
    return metrics, output