# Lab ML 3

This project implements a convolutional neural network using **EfficientNetV2** to classify images from the **CIFAR-10** dataset. It includes dataset downloading, preprocessing, model training, and evaluation.

## Usage Example

There are two primary ways to use the code:

### Launch with DVC

```bash
dvc repro
```

### Run Standalone Scripts

```bash
python src/download.py
python src/ingest.py
python src/train_model.py
python src/test_model.py
```

### Custom Python Script Example

```python
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import logging
import yaml

import src.download as download
import src.ingestion as ingestion
import src.loader as loader
import src.model as mdl
import src.train_model as train_model
import src.test_model as test_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_dataset(cifar10_url: str, save_dir: str, force_download: bool = False):
    from pathlib import Path

    def _is_extracted(save_dir: str):
        save_path = Path(save_dir)
        return (save_path / "cifar-10-batches-py").exists()

    if not force_download and _is_extracted(save_dir):
        logging.info("Dataset is already extracted, skipping download and extract process")
        return save_dir
    else:
        return download.download_and_extract(cifar10_url, save_dir)

def train(config, save_path):
    logging.info(f"Starting training with {config['prepare']['batch_names_select']} batches...")

    cifar10_url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    cifar10_dir = load_dataset(cifar10_url, "./data")

    # Step 2: Process data
    train_df, val_df, test_df = ingestion.process_data(cifar10_dir, config)

    # Step 3: Create data loaders
    train_loader = loader.create_data_loader(train_df, config)
    val_loader = loader.create_data_loader(val_df, config)
    test_loader = loader.create_data_loader(test_df, config)

    # Step 4: Define model, loss function, optimizer
    model = mdl.EfficientNetV2(n_classes=10).to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config["train"]["lr"])

    def add_batches_to_filename(filepath, number):
        import os
        from pathlib import Path
        directory, filename = os.path.split(filepath)
        name, ext = os.path.splitext(filename)
        new_filename = f"{name}_({number}bt){ext}"
        return Path(os.path.join(directory, new_filename))

    # Step 5: Train model
    best_model_path = train_model.train_model(
        model,
        train_loader,
        val_loader,
        loss_function,
        optimizer,
        num_epochs=config['train']['num_epochs'],
        device=device,
        save_path=add_batches_to_filename(save_path, len(config['prepare']['batch_names_select']))
    )

    # Step 6: Test model
    test_model.test_model(model, test_loader, loss_function, device)

    return best_model_path

with open("params.yaml", "r") as file:
    config = yaml.safe_load(file)

train(config, "./data/models/best_model.pth")
```