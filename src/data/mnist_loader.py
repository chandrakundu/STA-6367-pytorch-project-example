# src/data/mnist_loader.py

import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os


class MNISTLoader:
    def __init__(self, batch_size=64, shuffle=True, device="cpu"):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device

        # Define data transformation
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )

        data_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "../../data/external"
        )

        # Download MNIST dataset
        train_dataset = datasets.MNIST(
            data_path, train=True, download=True, transform=transform
        )
        test_dataset = datasets.MNIST(
            data_path, train=False, download=True, transform=transform
        )

        # Move datasets to the specified device
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
        )

        self.test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
        )

        # Move data loaders to the specified device
        self.train_loader.dataset.data = self.train_loader.dataset.data.to(self.device)
        self.train_loader.dataset.targets = self.train_loader.dataset.targets.to(
            self.device
        )

        self.test_loader.dataset.data = self.test_loader.dataset.data.to(self.device)
        self.test_loader.dataset.targets = self.test_loader.dataset.targets.to(
            self.device
        )

    def get_train_loader(self):
        return self.train_loader

    def get_test_loader(self):
        return self.test_loader

    def show_samples(self, num_samples=5):
        images, labels = next(iter(self.train_loader))

        # Move images and labels to the CPU before converting to NumPy
        images = images.cpu()
        labels = labels.cpu()

        fig, axes = plt.subplots(1, num_samples, figsize=(10, 2))
        for i in range(num_samples):
            axes[i].imshow(np.squeeze(images[i].numpy()), cmap="gray")
            axes[i].set_title(f"Label: {labels[i].item()}")
            axes[i].axis("off")

        plt.show()


if __name__ == "__main__":
    # Specify the device you want to use (e.g., "cuda" or "cpu")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    mnist_loader = MNISTLoader(batch_size=5, shuffle=True, device=device)

    # get the shape of the training data
    print(mnist_loader.get_train_loader().dataset.data.shape)

    # Display some sample images
    mnist_loader.show_samples()
