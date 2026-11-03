"""
Unified data loader for OpenML datasets and local CSVs.
Handles all preprocessing required before model training.
"""

from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd
import openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


@dataclass
class Dataset:
    """Holds a fully preprocessed dataset ready for model fitting."""
    name: str
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    n_rows: int
    n_features: int
    class_balance: float       # fraction of positive class
    openml_id: Optional[int] = None


class DataLoader:
    """
    Load and preprocess datasets from OpenML or local CSV files.
    
    All models receive identical splits via the same random seed.
    Preprocessing is intentionally minimal — we want to test models
    on realistic messy data, not sanitized inputs.
    """

    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state

    def load_openml(self, dataset_id: int, dataset_name: str) -> Dataset:
        """Load a dataset from OpenML by its integer ID."""
        print(f"  Loading OpenML dataset {dataset_id} ({dataset_name})...")
        
        dataset = openml.datasets.get_dataset(
            dataset_id,
            download_data=True,
            download_qualities=True,
            download_features_meta_data=False,
        )
        X, y, categorical_indicator, _ = dataset.get_data(
            dataset_format="dataframe",
            target=dataset.default_target_attribute,
        )
        return self._process(X, y, dataset_name, openml_id=dataset_id,
                             categorical_mask=categorical_indicator)

    def load_csv(self, path: str, target_col: str, name: str) -> Dataset:
        """Load a local CSV file."""
        # Resolve absolute path relative to project root if relative
        path = Path(path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found at: {path}")

        try:
            df = pd.read_csv(path)
        except Exception as e:
            raise IOError(f"Failed to read CSV file at {path}: {e}")

        if df.empty:
            raise ValueError(f"CSV file at {path} is empty")

        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in CSV columns: {list(df.columns)}")

        X = df.drop(columns=[target_col])
        y = df[target_col]

        if X.empty:
            raise ValueError(f"Dataset at {path} has no feature columns after removing target '{target_col}'")

        return self._process(X, y, name)

    def _process(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        name: str,
        openml_id: Optional[int] = None,
        categorical_mask: Optional[list] = None,
    ) -> Dataset:
        """
        Minimal but consistent preprocessing:
        1. Encode categorical features as integers (ordinal-style)
        2. Fill missing values with column median (numeric) or mode (categorical)
        3. Encode the target as 0/1 binary
        4. Stratified train/test split
        
        We intentionally do NOT scale features here — tree models don't
        need it, and TabPFN handles its own normalisation internally.
        Add a StandardScaler step inside the MLP wrapper only.
        """
        X = X.copy()
        y = y.copy()
        # 1. Encode target
        if y.dtype == "object" or str(y.dtype) == "category":
            le = LabelEncoder()
            y = le.fit_transform(y.astype(str))
        else:
            y = y.values.astype(np.int32)
        # 2. Stratified split
        X_arr = X.values
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )
        
        # 3. Handle categoricals: encode as integers
        cat_cols = X_train.select_dtypes(include=["object", "category"]).columns
        for col in cat_cols:
            X_train[col] = X_train[col].astype('object').fillna("__MISSING__")
            X_test[col] = X_test[col].astype('object').fillna("__MISSING__")
            le = LabelEncoder()
            le.fit(X_train[col].astype(str))
            train_encoded = pd.Series(le.transform(X_train[col].astype(str)),index=X_train.index)
            X_train[col] = train_encoded.astype(np.float32)
            test_values = X_test[col].astype(str)
            test_encoded = test_values.map(lambda v:v if v in le.classes_ else "__UNSEEN__")
            if "__UNSEEN__" not in le.classes_:
                le.classes_ = np.append(le.classes_,"__UNSEEN__")
            X_test[col] = le.transform(test_encoded).astype(np.float32)
            

        # 4. Fill missing values in numerical cols
        num_cols = X_train.columns.difference(cat_cols)
        for col in num_cols:
            if X_train[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                median = X_train[col].median()
                X_train[col] = X_train[col].fillna(median)
                X_test[col] = X_test[col].fillna(median)

        X_train = X_train.astype(np.float32)
        X_test = X_test.astype(np.float32)
        y_train = np.asarray(y_train,dtype=np.int32)
        y_test = np.asarray(y_test,dtype=np.int32)
        

        return Dataset(
            name=name,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train.astype(np.int32),
            y_test=y_test.astype(np.int32),
            n_rows=len(X_arr),
            n_features=X_arr.shape[1],
            class_balance=float(y.mean()),
            openml_id=openml_id,
        )