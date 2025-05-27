import os
import logging
import yaml
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

from metrics import evaluate_metrics
from loader import create_data_loader
from model import EfficientNetV2

logging.basicConfig(level=logging.INFO)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_function: nn.Module,
    optimizer: optim.Optimizer,
    num_epochs: int,
    device: torch.device,
    save_path: Path = Path("best_model.pth"),
) -> Path:

    model.to(device)
    best_val_loss: float = float("inf")
    best_model_path: Path = Path()

    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        for batch_inputs, batch_targets in train_loader:
            batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(
                device
            )

            # Forward pass
            outputs: torch.Tensor = model(batch_inputs)
            loss: torch.Tensor = loss_function(outputs, batch_targets)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        logging.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

        val_loss, acc, precision, recall, f1 = evaluate_metrics(
            model, val_loader, loss_function, device
        )

        logging.info(
            f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}, "
            f"Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = save_path
            torch.save(model.state_dict(), best_model_path)
            logging.info(f"Best model saved with validation loss: {best_val_loss:.4f}")

    logging.info("Training complete.")

    return best_model_path


def main() -> None:
    device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open("params.yaml", "r") as file:
        config = yaml.safe_load(file)

    os.makedirs(os.path.dirname(config["train"]["model_save_path"]), exist_ok=True)

    train_df = pd.read_pickle(config["prepare"]["train_split_path"], compression="zip")
    val_df = pd.read_pickle(config["prepare"]["val_split_path"], compression="zip")

    train_loader = create_data_loader(train_df, config)
    val_loader = create_data_loader(val_df, config)

    model = EfficientNetV2(n_classes=10).to(device_)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config["train"]["lr"])

    logging.info("Data loaded. Starting training...")

    train_model(
        model,
        train_loader,
        val_loader,
        loss_function,
        optimizer,
        num_epochs=config["train"]["num_epochs"],
        device=device_,
        save_path=Path(config["train"]["model_save_path"]),
    )


if __name__ == "__main__":
    main()
