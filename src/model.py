import torch
import torch.nn as nn
import torchvision.models as models


class SimpleNN(nn.Module):
    def __init__(self, n_classes: int) -> None:
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(
            32 * 32 * 3, 128
        )  # Example for an input size of 784*3 (e.g., flattened 28x28 image) and hidden layer with 128 units
        self.fc2 = nn.Linear(
            128, n_classes
        )  # Output layer with 10 units (e.g., 10 classes for classification)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(
            x, 1
        )  # Flatten the input tensor except for the batch dimension
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class EfficientNetV2(nn.Module):
    def __init__(self, n_classes: int):
        super(EfficientNetV2, self).__init__()
        self.efficientnet = models.efficientnet_v2_s()

        in_features = self.efficientnet.classifier[1].in_features
        if isinstance(in_features, int):
            self.efficientnet.classifier[1] = nn.Linear(in_features, n_classes)
        else:
            raise TypeError(
                f"Expected in_features to be an int, but got {type(in_features)}"
            )

    def forward(self, x):
        return self.efficientnet(x)
