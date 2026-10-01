
"""
Reusable Utility Functions
--------------------------
Provides common functions for:
- Reading CSV files
- Saving DataFrames
- Saving Python objects using joblib
- Loading Python objects using joblib

Designed for reuse across data ingestion, validation, transformation,
model training, and prediction pipelines.
"""

from pathlib import Path
from typing import Any

import joblib
import pandas as pd


def read_csv(
    file_path: str | Path,
    *args: Any,
    **kwargs: Any,
) -> pd.DataFrame:
    """Read a CSV file into a pandas DataFrame."""
    return pd.read_csv(file_path, *args, **kwargs)


def save_dataframe(
    dataframe: pd.DataFrame,
    file_path: str | Path,
    *args: Any,
    **kwargs: Any,
) -> None:
    """Save a DataFrame to a CSV file."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    dataframe.to_csv(file_path, *args, **kwargs)


def save_object(
    obj: Any,
    file_path: str | Path,
    *args: Any,
    **kwargs: Any,
) -> None:
    """Serialize and save a Python object using joblib."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(obj, file_path, *args, **kwargs)


def load_object(
    file_path: str | Path,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Load a Python object serialized using joblib."""
    return joblib.load(file_path, *args, **kwargs)