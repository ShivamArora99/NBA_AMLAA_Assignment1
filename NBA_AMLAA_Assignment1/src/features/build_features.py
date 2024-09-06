# src/features/build_features.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd

def important_features(df, target_col='drafted', n_top=20):
    """
    Identifies the top important features using a RandomForestClassifier.

    Args:
        df (pd.DataFrame): Input DataFrame with features and target column.
        target_col (str): The name of the target column in the DataFrame.
        n_top (int): Number of top features to select.

    Returns:
        pd.Index: Index of the top important feature names.
    """
    # Ensure the target column exists
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in the DataFrame.")

    # Prepare the features and target
    features = df.drop(columns=target_col)
    target = df[target_col]

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(features, target)
    feature_importances = pd.Series(rf_model.feature_importances_, index=features.columns)
    top_features = feature_importances.nlargest(n_top).index

    print("Top Features:\n", top_features)
    return top_features

def pre_processing(input_df):
    """
    Preprocess the input data by scaling and returning a DataFrame.

    Args:
        input_df (pd.DataFrame): Input DataFrame to preprocess.
        target_col (str): The name of the target column in the DataFrame.
        n_top (int): Number of top features to scale.

    Returns:
        pd.DataFrame: Scaled DataFrame with top features.
    """
    

    # Ensure not to scale the target column
    
    
    scaler = StandardScaler()
    input_df.iloc[:, :-1] = scaler.fit_transform(input_df.iloc[:, :-1])

    return input_df


def apply_pca(df, variance_threshold=0.99):
    """
    Apply PCA to reduce dimensionality of the data.

    Args:
        df (pd.DataFrame): Input DataFrame with features.
        variance_threshold (float): Threshold for cumulative explained variance to determine number of components.

    Returns:
        np.ndarray: Transformed feature matrix with reduced dimensions.
        PCA: Fitted PCA object.
    """
    pca = PCA()
    pca.fit(df)
    var_exp_cumsum = pca.explained_variance_ratio_.cumsum()
    n_components = len(var_exp_cumsum[var_exp_cumsum <= variance_threshold])
    pca = PCA(n_components=n_components)
    df_pca = pca.fit_transform(df)

    print(f"Number of PCA components used: {n_components}")
    return df_pca, pca
