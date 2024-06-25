import torch
import os
import cv2
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from string import ascii_lowercase
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from conv_neural_net import ConvolutionalNeuralNetwork

running_local = True if os.getenv('JUPYTERHUB_USER') is None else False
DATASET_PATH = "."

if running_local:
    local_path = "../sign_lang_train"
    if os.path.exists(local_path):
        DATASET_PATH = local_path
else:
    DATASET_PATH = "/data/mlproject22/sign_lang_train"


# def read_csv(csv_file):
#     with open(csv_file, newline='') as f:
#         reader = csv.reader(f)
#         data = list(reader)
#     return data

def read_csv(file_path):
    # Adjust this function to read your CSV properly
    data = pd.read_csv(file_path, header=None).values
    return data


class SignLangDataset(Dataset):
    """Sign language dataset"""

    def __init__(self, csv_file, root_dir, class_index_map=None, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data = read_csv(os.path.join(root_dir, csv_file))
        self.root_dir = root_dir
        self.class_index_map = class_index_map
        self.transform = transform
        # List of class names in order
        self.class_names = list(map(str, list(range(10)))) + list(ascii_lowercase)

    def __len__(self):
        """Calculates the length of the dataset"""
        return len(self.data)

    def __getitem__(self, idx):
        """Returns one sample (dict consisting of an image and its label)"""
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Read the image and labels
        image_path = os.path.join(self.root_dir, self.data[idx][1])
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # Shape of the image should be H,W,C where C=1
        image = np.expand_dims(image, 0)
        image = image.astype(np.float32)
        # Normalize the image
        image = (image - 127.5) / 127.5
        # Convert image to torch tensor
        image = torch.from_numpy(image)

        # The label is the index of the class name in the list ['0','1',...,'9','a','b',...'z']
        # because we should have integer labels in the range 0-35 (for 36 classes)
        label = self.class_names.index(self.data[idx][0])

        # if self.transform:
        #     image = self.transform(image)
        #
        # return image, label

        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample


def train_model(model, criterion, optimizer, train_loader, num_epochs=25, device='cpu'):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}")
        for i, sample in progress_bar:
            image = sample['image'].to(device)
            label = sample['label'].to(device)

            optimizer.zero_grad()

            output = model(image)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * image.size(0)

            progress_bar.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(train_loader.dataset)
        # print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}')

    return model


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = SignLangDataset('labels.csv', DATASET_PATH, transform=None)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = ConvolutionalNeuralNetwork(num_classes=36).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

trained_model = train_model(model, criterion, optimizer, train_loader, num_epochs=10, device=device)

torch.save(trained_model.state_dict(), "cnn-weights.pt")
print("Training completed. Weights saved to 'cnn-weights.pt'.")
