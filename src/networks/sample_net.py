# src/networks/sample_net.py

import torch.nn as nn
import torch.nn.functional as F


class SampleNet(nn.Module):
    """
    SampleNet: A simple neural network model for image classification.

    Args:
        None

    Attributes:
        conv1 (nn.Conv2d): Convolutional layer with 1 input channel, 32 output channels, and a kernel size of 3.
        d1 (nn.Linear): Fully connected layer mapping input features to 128 output features.
        d2 (nn.Linear): Fully connected layer mapping 128 input features to 10 output features.

    Methods:
        forward(x): Defines the forward pass of the model.

    Example:
        model = SampleNet()
        input_tensor = torch.randn(32, 1, 28, 28)  # Example input tensor
        output = model(input_tensor)
    """

    def __init__(self):
        super(SampleNet, self).__init__()

        # 28x28x1 => 26x26x32
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.d1 = nn.Linear(26 * 26 * 32, 128)
        self.d2 = nn.Linear(128, 10)

    def forward(self, x):
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 28, 28).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 10).
        """
        # 32x1x28x28 => 32x32x26x26
        x = self.conv1(x)
        x = F.relu(x)

        # flatten => 32 x (32*26*26)
        x = x.flatten(start_dim=1)

        # 32 x (32*26*26) => 32x128
        x = self.d1(x)
        x = F.relu(x)

        # logits => 32x10
        logits = self.d2(x)
        out = F.softmax(logits, dim=1)
        return out


# Test the model when run as the main script
if __name__ == "__main__":
    # Instantiate the model
    model = SampleNet()

    # Print the model summary
    print(model)
