# src/models/train_model.py

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm  # Import tqdm for the progress bar


def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=5):
    """
    Train the given model.

    Args:
        model (torch.nn.Module): The neural network model to train.
        train_loader (torch.utils.data.DataLoader): DataLoader for training data.
        test_loader (torch.utils.data.DataLoader): DataLoader for testing data.
        criterion: Loss function.
        optimizer: Optimization algorithm.
        num_epochs (int): Number of training epochs.

    Returns:
        None
    """
    for epoch in range(num_epochs):
        # Training
        model.train()
        running_loss = 0.0
        train_bar = tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", dynamic_ncols=True
        )

        for inputs, labels in train_bar:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            train_bar.set_postfix(
                {"Train Loss": running_loss / len(train_bar)}, refresh=True
            )

        # Calculate average training loss
        average_train_loss = running_loss / len(train_loader)

        # Testing
        model.eval()
        correct = 0
        total = 0
        test_bar = tqdm(test_loader, desc="Testing", dynamic_ncols=True)

        with torch.no_grad():
            for inputs, labels in test_bar:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                test_bar.set_postfix({"Test Accuracy": correct / total}, refresh=True)

        # Calculate accuracy
        accuracy = correct / total

        # Print training progress
        print(
            f"Epoch {epoch + 1}/{num_epochs} | Training Loss: {average_train_loss:.4f} | Accuracy: {accuracy:.2%}"
        )


if __name__ == "__main__":
    # This section is for testing the train_model function
    # You can modify it based on your actual data and model
    import sys

    sys.path.append("/home/ckundu/projects/sta6367/python_temp/my_first_ml/src")

    from networks.sample_net import SampleNet  # Import your model
    from data.mnist_loader import MNISTLoader  # Import your data loader

    # Initialize model and data loader
    model = SampleNet()
    mnist_loader = MNISTLoader(batch_size=64, shuffle=True)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Train the model for a few epochs (for testing purposes)
    train_model(
        model,
        mnist_loader.get_train_loader(),
        mnist_loader.get_test_loader(),
        criterion,
        optimizer,
        num_epochs=3,
    )
