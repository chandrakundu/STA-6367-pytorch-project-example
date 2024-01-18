# src/models/predict_model.py

import torch


def predict_model(model, data_loader):
    """
    Predict labels using the given trained model.

    Args:
        model (torch.nn.Module): The trained neural network model.
        data_loader (torch.utils.data.DataLoader): DataLoader for inference data.

    Returns:
        torch.Tensor: Predicted labels.
    """
    model.eval()
    all_predictions = []

    with torch.no_grad():
        for inputs, _ in data_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_predictions.extend(predicted.tolist())

    return torch.tensor(all_predictions)


if __name__ == "__main__":
    # This section is for testing the predict_model function
    # You can modify it based on your actual data and model
    import sys

    sys.path.append("/home/ckundu/projects/sta6367/python_temp/my_first_ml/src")

    from networks.sample_net import SampleNet  # Import your trained model
    from data.mnist_loader import MNISTLoader  # Import your data loader

    # Initialize model and data loader
    model = SampleNet()
    mnist_loader = MNISTLoader(batch_size=64, shuffle=True)

    # Load the trained model state (if applicable)
    # model.load_state_dict(torch.load('path/to/your/trained_model.pth'))

    # Predict using the model
    predictions = predict_model(model, mnist_loader.get_test_loader())

    # Print the predictions (for testing purposes)
    print(predictions)
