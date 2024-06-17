import numpy as np
from sklearn.metrics import accuracy_score
import torch


def calculate_accuracy(dataloader, model, max_batches=30):
    model.eval()
    accuracies = list()

    with torch.no_grad():  # Disable gradient calculation TODO ? Compare difference with torch.set_grad_enabled(False)
        for batch_idx, data in enumerate(dataloader):
            inputs, labels = data
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            y_true = labels.numpy()
            y_pred = predicted.numpy()

            accuracies.append(accuracy_score(y_true, y_pred))

            if batch_idx == (max_batches - 1):
                break

    mean_accuracy = np.mean(accuracies)
    return mean_accuracy
