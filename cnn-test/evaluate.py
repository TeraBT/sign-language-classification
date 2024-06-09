import torch
from torch.utils.data import DataLoader
from sign_language_dataset import SignLanguageDataset, transform
from sign_language_model import SignLanguageModel
from utils import calculate_accuracy

# Load the dataset and create dataloader
dataset = SignLanguageDataset(csv_file='../sign_lang_train/labels.csv', root_dir='../sign_lang_train', transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Load the model with saved weights
model = SignLanguageModel()
model.load_state_dict(torch.load('sign_language_model.pth'))
model.eval()  # Set the model to evaluation mode

# Calculate accuracy
mean_accuracy = calculate_accuracy(dataloader, model)
print(f'Mean Accuracy: {mean_accuracy:.4f}')
