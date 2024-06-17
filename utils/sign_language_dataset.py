import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset



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
        label = ord(label) - ord('0') if label.isdigit() else ord(label) - ord('a') + 10

        if self.transform:
            image = self.transform(image)

        return image, label



