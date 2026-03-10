import os
import pandas as pd


def get_dataset(name: str) -> pd.DataFrame:
    """
    Load a bundled R Gifi reference dataset.

    Parameters
    ----------
    name : str
        The name of the dataset (e.g., 'ABC', 'galo', 'hartigan', 'gubell',
        'house', 'mammals', 'neumann', 'roskam', 'senate07', 'sleeping',
        'small', 'WilPat2').

    Returns
    -------
    pd.DataFrame
        The requested dataset with appropriate index.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, 'data')
    file_path = os.path.join(data_dir, f"{name}.csv")

    if not os.path.exists(file_path):
        valid_datasets = [f.split('.')[0] for f in os.listdir(
            data_dir) if f.endswith('.csv')]
        raise FileNotFoundError(
            f"Dataset '{name}' not found. Available datasets: {
                ', '.join(valid_datasets)}")

    return pd.read_csv(file_path, index_col=0)
