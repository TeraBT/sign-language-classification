import numpy as np
from sklearn.metrics import accuracy_score
import torch

def calculate_accuracy(dataloader, model, max_batches=30):
    model.eval()  # Set the model to evaluation mode
    accuracies = list()

    with torch.no_grad():  # Disable gradient calculation
        for batch_idx, data in enumerate(dataloader):
            inputs, labels = data
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            # Convert to numpy arrays
            y_true = labels.numpy()
            y_pred = predicted.numpy()

            # Calculate accuracy
            accuracies.append(accuracy_score(y_true, y_pred))

            # Consider only the first 30 batches
            if batch_idx == (max_batches - 1):
                break

    mean_accuracy = np.mean(accuracies)
    return mean_accuracy
