# src/models/predict_model.py

import pandas as pd
import joblib
from category_encoders import TargetEncoder
import pandas as pd
import src.data.make_dataset as clean_data  # Assuming clean_data is used for data cleaning
from src.features.build_features import pre_processing, apply_pca  # Assuming these functions handle scaling and feature selection

def load_models(model_filepath, pca_filepath):
    """
    Load the trained model and PCA model from disk.

    Args:
        model_filepath (str): Path to the saved logistic regression model file.
        pca_filepath (str): Path to the saved PCA model file.

    Returns:
        model: Loaded logistic regression model.
        pca: Loaded PCA model.
    """
    model = joblib.load(model_filepath)
    pca = joblib.load(pca_filepath)
    return model, pca


def preprocess_test_data(test_df, encoder_filepath='target_encoder.pkl'):
    """
    Clean and preprocess the test data using the pre-fitted TargetEncoder.

    Args:
        test_df (pd.DataFrame): Raw test data DataFrame.
        encoder_filepath (str): Path to the saved target encoder.

    Returns:
        pd.DataFrame: Cleaned and preprocessed DataFrame.
    """
    print(f"Initial Test DataFrame shape: {test_df.shape}")

    # Impute missing values
    test_df = clean_data.impute_missing_values(test_df)
    print(f"Shape after imputing missing values: {test_df.shape}")

    # Drop unnecessary columns
    test_df = clean_data.drop_unnecessary_columns(test_df)
    print(f"Shape after dropping unnecessary columns: {test_df.shape}")

    # Drop columns with high missing values
    test_df = clean_data.drop_high_missing_columns(test_df)
    print(f"Shape after dropping high missing columns: {test_df.shape}")

                                                                                                                                                                                                                                                                                                                                                                                                                                            

    # Encode 'yr' as a category
    if 'yr' in test_df.columns:
        test_df['yr'] = test_df['yr'].astype('category').cat.codes
        print("Encoded 'yr' as categorical codes.")

    # Load the pre-fitted TargetEncoder
    target_encoder = joblib.load(encoder_filepath)
    print("Encoding categorical columns with loaded TargetEncoder...")
    
    # Apply the transform method to encode the categorical columns
    test_df[['team', 'conf']] = target_encoder.transform(test_df[['team', 'conf']])
    print(f"Shape after encoding categorical columns: {test_df.shape}")

    test_df.fillna(test_df.median(), inplace=True)
    print(test_df.isna().sum())

    print(f"Final Test DataFrame shape: {test_df.shape}")

    return test_df


def predict_test_data(test_filepath, model_filepath='trained_model.pkl', pca_filepath='pca_model.pkl'):
    """
    Make predictions on the test data using the trained model and PCA.

    Args:
        test_filepath (str): Path to the test CSV file.
        model_filepath (str): Path to the saved logistic regression model file.
        pca_filepath (str): Path to the saved PCA model file.

    Returns:
        np.ndarray: Predicted probabilities for the test data.
    """
    # Load the test data
    test_df = pd.read_csv(test_filepath)

    # Load the trained model and PCA model
    model, pca = load_models(model_filepath, pca_filepath)

    # Preprocess the test data (cleaning, feature scaling, etc.)
    test_df_processed = preprocess_test_data(test_df)

    # Apply PCA transformation
    test_df_pca = pca.transform(test_df_processed)

    # Predict probabilities for the test data
    y_test_pred = model.predict_proba(test_df_pca)[:, 1]

    return y_test_pred
