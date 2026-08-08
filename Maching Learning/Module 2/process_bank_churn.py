"""Preprocessing utilities for the Bank Customer Churn Prediction dataset.

The functions in this module are used by the Decision Tree homework.
"""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


def split_data(
    raw_df: pd.DataFrame,
    target_col: str = "Exited",
    test_size: float = 0.25,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split the raw dataset into stratified training and validation data.

    Args:
        raw_df: Raw input DataFrame containing the target column.
        target_col: Name of the target column.
        test_size: Fraction of rows assigned to the validation set.
        random_state: Random seed used for reproducibility.

    Returns:
        A tuple containing the training DataFrame and validation DataFrame.
    """
    train_df, val_df = train_test_split(
        raw_df,
        test_size=test_size,
        random_state=random_state,
        stratify=raw_df[target_col],
    )
    return train_df, val_df


def select_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    target_col: str = "Exited",
    columns_to_drop: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, List[str]]:
    """Separate features and targets and remove unwanted input columns.

    Args:
        train_df: Training DataFrame.
        val_df: Validation DataFrame.
        target_col: Name of the target column.
        columns_to_drop: Columns that should not be used as model features.

    Returns:
        Training features, training targets, validation features,
        validation targets, and the list of feature column names.
    """
    if columns_to_drop is None:
        columns_to_drop = ["CustomerId", "Surname"]

    input_cols = [col for col in train_df.columns if col != target_col]

    train_inputs = train_df[input_cols].copy()
    val_inputs = val_df[input_cols].copy()

    train_targets = train_df[target_col].copy()
    val_targets = val_df[target_col].copy()

    train_inputs = train_inputs.drop(columns=columns_to_drop, errors="ignore")
    val_inputs = val_inputs.drop(columns=columns_to_drop, errors="ignore")

    input_cols = train_inputs.columns.tolist()

    return (
        train_inputs,
        train_targets,
        val_inputs,
        val_targets,
        input_cols,
    )


def get_feature_types(
    inputs: pd.DataFrame,
) -> Tuple[List[str], List[str]]:
    """Identify numeric and categorical feature columns.

    Args:
        inputs: DataFrame containing model input features.

    Returns:
        A tuple containing lists of numeric and categorical column names.
    """
    numeric_cols = inputs.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = inputs.select_dtypes(include="object").columns.tolist()

    return numeric_cols, categorical_cols


def scale_features(
    train_inputs: pd.DataFrame,
    val_inputs: pd.DataFrame,
    numeric_cols: List[str],
    scaler: Optional[MinMaxScaler] = None,
    fit_scaler: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[MinMaxScaler]]:
    """Scale numeric features using MinMaxScaler.

    Args:
        train_inputs: Training feature DataFrame.
        val_inputs: Validation feature DataFrame.
        numeric_cols: Numeric columns to scale.
        scaler: Existing fitted scaler. A new scaler is created if omitted.
        fit_scaler: Whether to fit the scaler on the training data.

    Returns:
        Scaled training features, scaled validation features, and the scaler.
        If numeric_cols is empty, the scaler is returned unchanged.
    """
    train_inputs = train_inputs.copy()
    val_inputs = val_inputs.copy()

    if not numeric_cols:
        return train_inputs, val_inputs, scaler

    if scaler is None:
        scaler = MinMaxScaler()

    if fit_scaler:
        scaler.fit(train_inputs[numeric_cols])

    train_inputs[numeric_cols] = scaler.transform(train_inputs[numeric_cols])
    val_inputs[numeric_cols] = scaler.transform(val_inputs[numeric_cols])

    return train_inputs, val_inputs, scaler


def encode_categorical_features(
    train_inputs: pd.DataFrame,
    val_inputs: pd.DataFrame,
    categorical_cols: List[str],
    encoder: Optional[OneHotEncoder] = None,
    fit_encoder: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, OneHotEncoder, List[str]]:
    """One-hot encode categorical features.

    Args:
        train_inputs: Training feature DataFrame.
        val_inputs: Validation feature DataFrame.
        categorical_cols: Categorical columns to encode.
        encoder: Existing fitted encoder. A new encoder is created if omitted.
        fit_encoder: Whether to fit the encoder on training data.

    Returns:
        Encoded training features, encoded validation features, the encoder,
        and the names of the generated one-hot columns.
    """
    train_inputs = train_inputs.copy()
    val_inputs = val_inputs.copy()

    if encoder is None:
        encoder = OneHotEncoder(
            sparse_output=False,
            handle_unknown="ignore",
        )

    if categorical_cols:
        if fit_encoder:
            encoder.fit(train_inputs[categorical_cols])

        encoded_cols = encoder.get_feature_names_out(categorical_cols).tolist()

        train_encoded = encoder.transform(train_inputs[categorical_cols])
        val_encoded = encoder.transform(val_inputs[categorical_cols])

        train_inputs[encoded_cols] = train_encoded
        val_inputs[encoded_cols] = val_encoded

        train_inputs = train_inputs.drop(columns=categorical_cols)
        val_inputs = val_inputs.drop(columns=categorical_cols)
    else:
        encoded_cols = []

    return train_inputs, val_inputs, encoder, encoded_cols


def preprocess_data(
    raw_df: pd.DataFrame,
    scaler_numeric: bool = True,
    target_col: str = "Exited",
    test_size: float = 0.25,
    random_state: int = 42,
) -> Tuple[
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.Series,
    List[str],
    Optional[MinMaxScaler],
    OneHotEncoder,
]:
    """Prepare raw bank churn data for model training.

    This function combines the small preprocessing steps above into one
    public function. It is the main function imported by the homework
    notebook.

    Args:
        raw_df: Raw training DataFrame containing the target column.
        scaler_numeric: If True, scale numeric features with MinMaxScaler.
            If False, leave numeric features unchanged (recommended for
            Decision Trees).
        target_col: Name of the target column.
        test_size: Fraction of data used for validation.
        random_state: Random seed for the train/validation split.

    Returns:
        X_train: Preprocessed training features.
        train_targets: Training target values.
        X_val: Preprocessed validation features.
        val_targets: Validation target values.
        input_cols: Names of the final model features.
        scaler: Fitted scaler, or None when scaling is disabled.
        encoder: Fitted OneHotEncoder.
    """
    train_df, val_df = split_data(
        raw_df=raw_df,
        target_col=target_col,
        test_size=test_size,
        random_state=random_state,
    )

    (
        train_inputs,
        train_targets,
        val_inputs,
        val_targets,
        _,
    ) = select_features(
        train_df=train_df,
        val_df=val_df,
        target_col=target_col,
        columns_to_drop=["CustomerId", "Surname"],
    )

    numeric_cols, categorical_cols = get_feature_types(train_inputs)

    scaler = None
    if scaler_numeric:
        train_inputs, val_inputs, scaler = scale_features(
            train_inputs=train_inputs,
            val_inputs=val_inputs,
            numeric_cols=numeric_cols,
        )

    (
        train_inputs,
        val_inputs,
        encoder,
        _,
    ) = encode_categorical_features(
        train_inputs=train_inputs,
        val_inputs=val_inputs,
        categorical_cols=categorical_cols,
    )

    input_cols = train_inputs.columns.tolist()

    return (
        train_inputs,
        train_targets,
        val_inputs,
        val_targets,
        input_cols,
        scaler,
        encoder,
    )


def preprocess_new_data(
    new_df: pd.DataFrame,
    input_cols: List[str],
    scaler: Optional[MinMaxScaler],
    encoder: OneHotEncoder,
    scaler_numeric: bool = True,
) -> pd.DataFrame:
    """Preprocess new raw data using already fitted preprocessing objects.

    Args:
        new_df: New raw data, for example test.csv.
        input_cols: Feature names expected by the trained model.
        scaler: Previously fitted MinMaxScaler, or None if scaling is disabled.
        encoder: Previously fitted OneHotEncoder.
        scaler_numeric: Whether numeric features were scaled during training.

    Returns:
        A DataFrame with the same processed feature structure used for training.
    """
    inputs = new_df.copy()

    inputs = inputs.drop(
        columns=["Exited", "CustomerId", "Surname"],
        errors="ignore",
    )

    numeric_cols = inputs.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = inputs.select_dtypes(include="object").columns.tolist()

    if scaler_numeric and scaler is not None and numeric_cols:
        inputs[numeric_cols] = scaler.transform(inputs[numeric_cols])

    if categorical_cols:
        encoded_cols = encoder.get_feature_names_out(categorical_cols).tolist()
        encoded_values = encoder.transform(inputs[categorical_cols])

        inputs[encoded_cols] = encoded_values
        inputs = inputs.drop(columns=categorical_cols)

    # Keep exactly the same columns and order as during model training.
    inputs = inputs.reindex(columns=input_cols, fill_value=0)

    return inputs
