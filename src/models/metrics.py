# src/models/metrics.py


def compute_accuracy(predictions, labels):
    """
    Compute accuracy given predicted labels and ground truth labels.

    Args:
        predictions (torch.Tensor): Predicted labels.
        labels (torch.Tensor): Ground truth labels.

    Returns:
        float: Accuracy.
    """
    correct = (predictions == labels).sum().item()
    total = len(labels)
    accuracy = correct / total
    return accuracy


if __name__ == "__main__":
    # This section is for testing the compute_accuracy function
    # You can modify it based on your actual predictions and ground truth

    import torch

    # Example predictions and ground truth labels (for testing purposes)
    predictions = torch.tensor([0, 1, 2, 3, 4])
    ground_truth = torch.tensor([0, 1, 2, 3, 9])

    # Compute accuracy
    accuracy = compute_accuracy(predictions, ground_truth)

    # Print the accuracy (for testing purposes)
    print(f"Accuracy: {accuracy:.2%}")
