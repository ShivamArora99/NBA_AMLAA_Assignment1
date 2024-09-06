# src/models/train_model.py

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score
import joblib
from src.features.build_features import pre_processing, apply_pca
from src.visualization.visualize import auc_roc_curve

def train_model(input_df_train, model, model_name, target_col, param_grid=None):
    """
    Train a specified machine learning model using PCA-transformed features and evaluate using ROC AUC.
    Hyperparameter tuning can be performed if a param_grid is provided.

    Args:
        input_df_train (pd.DataFrame): Input DataFrame containing training data.
        model: A machine learning model to be trained (e.g., LogisticRegression(), XGBClassifier()).
        model_name (str): The name of the model (used for saving the model and PCA).
        target_col (str): The name of the target column in the dataset.
        param_grid (dict): Hyperparameter grid for tuning using GridSearchCV.

    Returns:
        model: Trained machine learning model (after tuning if param_grid is provided).
        pca_model: Fitted PCA object.
        X_val (pd.DataFrame): Validation features.
        y_val (pd.Series): Validation labels.
    """
    # Preprocess the data and apply PCA
    input_df_train_processed = pre_processing(input_df_train)

    # Separate features and target
    X = input_df_train_processed.drop(target_col, axis=1)
    y = input_df_train_processed[target_col].astype(int)

    # Apply PCA to reduce dimensions
    X_pca, pca_model = apply_pca(X)

    # Setup Stratified K-Folds cross-validation
    y = y.reset_index(drop=True)
    skf = StratifiedKFold(n_splits=5)
    aucs = []

    print(f"Training {model_name}...")

    X_val = None
    y_val = None

    # Always apply Stratified KFold regardless of hyperparameter tuning
    for train_index, val_index in skf.split(X_pca, y):
        X_train, X_val = X_pca[train_index], X_pca[val_index]
        y_train, y_val = y[train_index], y[val_index]

        if param_grid is not None and train_index[0] == 0:
            # Hyperparameter tuning using GridSearchCV in the first fold
            print(f"Tuning hyperparameters for {model_name} using GridSearchCV...")
            grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=skf, scoring='roc_auc', n_jobs=-1, verbose=2)
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_
            print(f"Best hyperparameters found: {grid_search.best_params_}")

        # Fit the model for each fold
        model.fit(X_train, y_train)

        # Predict probabilities for validation set
        y_val_pred_proba = model.predict_proba(X_val)[:, 1]

        # Calculate AUC for validation set
        auc = roc_auc_score(y_val, y_val_pred_proba)
        aucs.append(auc)

        # Plot ROC curve for each fold
        auc_roc_curve(y_val, y_val_pred_proba)

    # Mean AUC across all folds for this model
    mean_auc = np.mean(aucs)
    print(f"Mean AUC across all folds for {model_name}: {mean_auc}")

    # Save the trained model and PCA model
    model_filepath = f'{model_name}_model.pkl'
    joblib.dump(model, model_filepath)
    print(f"Model saved as '{model_filepath}'")

    pca_filepath = f'{model_name}_pca.pkl'
    joblib.dump(pca_model, pca_filepath)
    print(f"PCA model saved as '{pca_filepath}'")

    # Return the last trained model and PCA model for further use
    return model, pca_model, X_val, y_val
