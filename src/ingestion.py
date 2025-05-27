import os
import logging
import pickle
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO)


def train_test_split(
    data: pd.DataFrame,
    test_size: Union[float, int] = 0.25,
    random_state: Union[int, None] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split arrays or matrices into random train and test subsets.

    Parameters:
    X :  pd.DataFrame
         The input data to split.
    test_size : float, int, or None, optional (default=0.25)
        If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split.
        If int, represents the absolute number of test samples.
        If None, the value is set to the complement of the train size.
    random_state : int or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If None, the random number generator is the RandomState instance used by np.random.

    Returns:
    Tuple containing:
        - data_train: pd.DataFrame
        - data_test: pd.DataFrame
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_samples = len(data)
    if isinstance(test_size, float):
        test_size = int(n_samples * test_size)

    indices = np.random.permutation(n_samples)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]

    data_train = data.iloc[train_indices]
    data_test = data.iloc[test_indices]

    return data_train, data_test


def unpickle(file):
    """Load a pickled CIFAR-10 batch file."""
    with open(file, "rb") as fo:
        dict_data = pickle.load(fo, encoding="bytes")
    return dict_data


def assign_batches(labels_df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Assigns batch labels to a dataframe.

    Args:
    - labels_df (pd.Dataframe): Dataframe containing dataset information in 'label' and 'image' columns.
    - config (Dict[str, Any]): Configuration dictionary containing parameters for data splitting and paths.

    Returns:
    - pd.DataFrame: dataframe with assigned batch labels in 'batch_name' column.
    """
    labels_df_ = labels_df.copy(deep=True)
    labels_df_["batch_name"] = "not_set"

    n_batches = config["prepare"]["n_batches"]
    batch_size = len(labels_df_) // n_batches

    batch_size_current = 0
    for batch_number in range(n_batches):
        if batch_number == (n_batches - 1):
            # select all the remaining data for the last batch
            loc = labels_df_.columns.get_loc("batch_name")
            if isinstance(loc, int):
                labels_df_.iloc[batch_size_current:, loc] = str(batch_number)
            else:
                raise TypeError(
                    'Expected to find one "batch_name" column in dataframe, but several were found'
                )
        else:
            loc = labels_df_.columns.get_loc("batch_name")
            if isinstance(loc, int):
                labels_df_.iloc[
                    batch_size_current : batch_size_current + batch_size,
                    loc,
                ] = str(batch_number)
            else:
                raise TypeError(
                    'Expected to find one "batch_name" column in dataframe, but several were found'
                )

        batch_size_current += batch_size

    return labels_df_


def select_batches(labels_df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Selects data from labels_df based on 'batch_names' from config.

    Args:
    - labels_df (pd.Dataframe): .
    - config (Dict[str, Any]): Configuration dictionary containing parameters for data splitting and paths.

    Returns:
    - pd.DataFrame:
    """
    batch_names: List[str] = config["prepare"]["batch_names_select"]
    labels_df_ = labels_df.copy(deep=True)

    labels_df_ = labels_df_[labels_df_["batch_name"].isin(batch_names)]

    return labels_df_


def process_data(
    data_dir: str, config: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load CIFAR-10 dataset from extracted directory.

    Args:
    - data_dir (str): Path to extracted CIFAR-10 dataset directory.
    - config (Dict[str, Any]): Configuration dictionary containing parameters for data splitting and paths.

    Returns:
    - Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Training, validation, and testing DataFrames.
    """
    batches = [f"{data_dir}/cifar-10-batches-py/data_batch_{i}" for i in range(1, 6)]
    test_batch = f"{data_dir}/cifar-10-batches-py/test_batch"

    all_data, all_labels = [], []

    # Load training data
    for batch in batches:
        batch_data = unpickle(batch)
        all_data.append(batch_data[b"data"])
        all_labels.extend(batch_data[b"labels"])

    train_data = np.vstack(all_data).reshape(-1, 3, 32, 32).astype("float32") / 255.0
    train_labels = np.array(all_labels)

    # Load test data
    test_data_dict = unpickle(test_batch)
    test_data = test_data_dict[b"data"].reshape(-1, 3, 32, 32).astype("float32") / 255.0
    test_labels = np.array(test_data_dict[b"labels"])

    # Create DataFrames
    train_df = pd.DataFrame({"image": list(train_data), "label": train_labels})
    test_df = pd.DataFrame({"image": list(test_data), "label": test_labels})

    train_df = assign_batches(train_df, config)
    logging.info(f"Split train dataset in {config['prepare']['n_batches']} batches")

    train_df = select_batches(train_df, config)
    logging.info(
        f"Batches {config['prepare']['batch_names_select']} selected from train dataset"
    )

    train_df, val_df = train_test_split(
        train_df,
        test_size=config["prepare"].get("val_size", 0.2),
        random_state=config["prepare"].get("random_state", 42),
    )
    logging.info(
        f"Prepared 3 data splits: train, size: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}"
    )
    return train_df, val_df, test_df


def main():
    with open("params.yaml", "r") as file:
        config = yaml.safe_load(file)

    os.makedirs(os.path.dirname(config["prepare"]["train_split_path"]), exist_ok=True)
    os.makedirs(os.path.dirname(config["prepare"]["val_split_path"]), exist_ok=True)
    os.makedirs(os.path.dirname(config["prepare"]["test_split_path"]), exist_ok=True)

    train_df, val_df, test_df = process_data(config["download"]["cifar_dir"], config)
    for save_path, dataframe in (
        (config["prepare"]["train_split_path"], train_df),
        (config["prepare"]["val_split_path"], val_df),
        (config["prepare"]["test_split_path"], test_df),
    ):
        logging.info(f"Dump split to: {save_path}")
        dataframe.to_pickle(save_path, compression="zip")


if __name__ == "__main__":
    main()
