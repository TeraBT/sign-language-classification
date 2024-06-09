import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import os

# Step 1: Load the dataset and preprocess the images
class SignLanguageDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.labels_frame = pd.read_csv(csv_file, header=None)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.labels_frame.iloc[idx, 1])
        image = Image.open(img_name).convert('L')  # Convert image to grayscale
        label = self.labels_frame.iloc[idx, 0]
        label = ord(label) - ord('0') if label.isdigit() else ord(label) - ord('a') + 10  # Encode label

        if self.transform:
            image = self.transform(image)

        return image, label

# Step 2: Define transformations
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Step 3: Create dataset and dataloader
dataset = SignLanguageDataset(csv_file='../sign_lang_train/labels.csv', root_dir='../sign_lang_train', transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Step 4: Define the neural network model
class SignLanguageModel(nn.Module):
    def __init__(self):
        super(SignLanguageModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 36)  # 36 classes

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 16 * 16)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# model = SignLanguageModel()
#
# # Step 5: Define loss function and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
#
# # Step 6: Train the model
# num_epochs = 10
#
# for epoch in range(num_epochs):
#     running_loss = 0.0
#     for i, data in enumerate(dataloader, 0):
#         inputs, labels = data
#
#         optimizer.zero_grad()
#
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#
#         running_loss += loss.item()
#         if i % 100 == 99:
#             print(f'Epoch [{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}')
#             running_loss = 0.0
#
# print('Finished Training')

# Load the model with saved weights
model = SignLanguageModel()
model.load_state_dict(torch.load('sign_language_model.pth'))
model.eval()  # Set the model to evaluation mode

# Step 7: Save the model
torch.save(model.state_dict(), 'sign_language_model.pth')

import numpy as np
from sklearn.metrics import accuracy_score

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

# Example usage
mean_accuracy = calculate_accuracy(dataloader, model)
print(f'Mean Accuracy: {mean_accuracy:.4f}')
