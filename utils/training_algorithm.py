import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.sign_language_dataset import SignLanguageDataset


def train(save_path, sign_language_model, num_epochs, transform=None):
    dataset = SignLanguageDataset(csv_file='../sign_lang_train/labels.csv', root_dir='../sign_lang_train',
                                  transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = sign_language_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            inputs, labels = data

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                print(f'Epoch [{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0

    print('Finished Training')

    torch.save(model.state_dict(), save_path)
