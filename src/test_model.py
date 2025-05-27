import logging
import yaml
from typing import Tuple

import pandas as pd

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from metrics import evaluate_metrics
from loader import create_data_loader
from model import EfficientNetV2

logging.basicConfig(level=logging.INFO)


def test_model(
    model: nn.Module,
    test_loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
) -> Tuple:
    """
    Test a trained model on a test dataset and compute test metrics.

    Args:
    - model (nn.Module): Trained PyTorch model.
    - test_loader (DataLoader): DataLoader for the test dataset.
    - loss_function (nn.Module): Loss function for computing the loss.
    - device (torch.device): Device (CPU or GPU) on which to run the evaluation.

    Returns:
    - tuple: Test loss, accuracy, precision, recall, and F1-score.
    """

    test_loss, acc, precision, recall, f1 = evaluate_metrics(
        model, test_loader, loss_function, device
    )

    logging.info(
        f"Test Loss: {test_loss:.4f}, Accuracy: {acc:.4f}, "
        f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}"
    )

    return test_loss, acc, precision, recall, f1


def main() -> None:
    device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open("params.yaml", "r") as file:
        config = yaml.safe_load(file)

    test_df = pd.read_pickle(config["prepare"]["val_split_path"], compression="zip")
    test_loader = create_data_loader(test_df, config)

    model = EfficientNetV2(n_classes=10).to(device_)
    model.load_state_dict(torch.load(config["train"]["model_save_path"]))

    loss_function = nn.CrossEntropyLoss()

    logging.info("Data loaded. Starting testing...")

    test_model(model, test_loader, loss_function, device_)


if __name__ == "__main__":
    main()
