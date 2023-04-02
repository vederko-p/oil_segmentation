
import os

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from PIL import Image
from tqdm.notebook import tqdm

import feature_extractor as f_extr


class UNetDataSet(Dataset):
    in_channels = 3
    out_channels = 1

    def __init__(self, img_dir, mask_dir, image_size=256):
        self.images = [os.path.join(img_dir, x) for x in sorted(os.listdir(img_dir))]
        self.masks = [os.path.join(mask_dir, x) for x in sorted(os.listdir(mask_dir))]
        self.image_size = image_size

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # paths:
        img_p = self.images[idx]
        mask_p = self.masks[idx]
        # open:
        image, _, binary_mask = f_extr.parse_img_label(img_p, mask_p)
        image = Image.fromarray(image)
        mask = Image.fromarray(binary_mask.astype(float))
        # tensors:
        image_tensor = self.transform(image)
        mask_tensor = torch.where(self.transform(mask) > 0, 1, 0).float()
        return image_tensor, mask_tensor


class ModelTrainer:
    def __init__(self, model, optimizer, loss_function, callback=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.callback = callback

        _device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(_device_name)

    def train_model(self, dataloader, epochs) -> None:
        iterations = tqdm(range(epochs), desc='epoch')
        iterations.set_postfix({'train epoch loss': np.nan})
        for epoch in iterations:
            epoch_loss = self._train_epoch(dataloader)
            iterations.set_postfix({'train epoch loss': epoch_loss})
            if self.callback is not None:
                self.callback(self.model, epoch_loss)

    def _train_epoch(self, batch_generator) -> float:
        epoch_loss = 0
        total = 0
        for batch_x, batch_y in batch_generator:
            batch_loss = self._train_on_batch(
                batch_x.to(self.device), batch_y.to(self.device))
            epoch_loss += batch_loss * batch_x.shape[0]
            total += batch_x.shape[0]
        return epoch_loss / total

    def _train_on_batch(self, x_batch, y_batch) -> float:
        self.model.train()
        self.model.zero_grad()
        output = self.model(x_batch)
        loss = self.loss_function(output, y_batch)
        loss.backward()
        self.optimizer.step()
        return loss.cpu().item()


class CallBack:
    def __init__(self, test_loader, loss_function):
        self.test_loader = test_loader
        self.loss_function = loss_function

        self.train_loss = []
        self.test_loss = []

        _device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(_device_name)

    def __call__(self, model, train_epoch_loss):
        # calc test loss:
        test_epoch_loss = 0
        total = 0
        for batch_x, batch_y in self.test_loader:
            with torch.no_grad():
                output = model(batch_x.to(self.device))
            batch_loss = self.loss_function(output, batch_y.to(self.device))
            test_epoch_loss += batch_loss.cpu().item() * batch_x.shape[0]
            total += batch_x.shape[0]
        test_epoch_loss /= total
        # add losses:
        self.train_loss.append(train_epoch_loss)
        self.test_loss.append(test_epoch_loss)
        # print losses:
        print('-' * 15)
        print(f'epoch: {len(self.test_loss)}')
        print('train:', round(train_epoch_loss, 3))
        print('test:', round(test_epoch_loss, 3))

    def plot_losses(self):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(self.train_loss, label='Train loss')
        ax.plot(self.test_loss, label='Test loss')
        plt.legend()
        plt.show()
