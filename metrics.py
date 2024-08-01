import torch
import torchmetrics

def calculate_metrics(preds, targets):
    preds = torch.argmax(preds, dim=1)
    targets = torch.argmax(targets, dim=1)

    accuracy = torchmetrics.Accuracy(num_classes=4, task="multiclass").cuda()
    precision = torchmetrics.Precision(num_classes=4, average="macro", task="multiclass").cuda()
    recall = torchmetrics.Recall(num_classes=4, average="macro", task='multiclass').cuda()
    f1_score = torchmetrics.F1Score(num_classes=4, average="macro", task='multiclass').cuda()
    acc = accuracy(preds, targets)
    prec = precision(preds, targets)
    rec = recall(preds, targets)
    f1 = f1_score(preds, targets)

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1
    }