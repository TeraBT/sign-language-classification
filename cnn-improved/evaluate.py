import torch
from torch.utils.data import DataLoader
from utils.sign_language_models import ImprovedModel
from utils.accuracy import calculate_accuracy
from utils.sign_language_dataset import SignLanguageDataset
from utils.transformers import improved_transformer

dataset = SignLanguageDataset(csv_file='../sign_lang_train/labels.csv', root_dir='../sign_lang_train',
                              transform=improved_transformer)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

model = ImprovedModel()
model.load_state_dict(torch.load('cnn-improved.pth'))
model.eval()

mean_accuracy = calculate_accuracy(dataloader, model)
print(f'Mean Accuracy: {mean_accuracy:.4f}')
