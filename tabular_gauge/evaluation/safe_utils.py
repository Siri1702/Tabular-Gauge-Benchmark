import numpy as np
import pandas as pd

def safe_get(series, index=0, default=np.nan):
    """Safely retrieve a value from a pandas Series."""
    try:
        return series.iloc[index] if not series.empty else default
    except (IndexError, KeyError):
        return default

def get_model_order(df: pd.DataFrame, preferred_order: list[str] = None) -> list[str]:
    """
    Dynamically discover models in the results and order them.

    1. Discover all unique models in the DataFrame.
    2. Intersect with preferred_order to maintain a logical sequence.
    3. Append any discovered models not in the preferred list to the end.
    """
    if df.empty or "model" not in df.columns:
        return []

    found_models = sorted(df["model"].unique().tolist())
    if preferred_order is None:
        return found_models

    # Maintain preferred order for found models
    ordered = [m for m in preferred_order if m in found_models]
    # Append others
    others = [m for m in found_models if m not in ordered]

    return ordered + others
