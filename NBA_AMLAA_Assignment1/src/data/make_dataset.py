from category_encoders import TargetEncoder
import pandas as pd
import numpy as np
import joblib

def report_missing_values(df):
    """
    Report the number and percentage of missing values in the DataFrame.
    """
    missing_values = df.isnull().sum()
    percentage_missing = (missing_values / df.shape[0]) * 100
    missing_values_df = pd.DataFrame({'missing_values': missing_values, 'percentage_missing': percentage_missing})
    return missing_values_df


def impute_missing_values(df):
    """
    Impute missing values in the DataFrame using median or mode.
    """
    print("Imputing missing values...")
    df['yr'].fillna(df['yr'].mode()[0], inplace=True)
    df['ast_tov'].fillna(df['ast_tov'].median(), inplace=True)
    df['rimmade'].fillna(df['rimmade'].median(), inplace=True)
    df['midmade'].fillna(df['midmade'].median(), inplace=True)
    df['dunksmade'].fillna(df['dunksmade'].median(), inplace=True)
    df['rim_ratio'].fillna(df['rim_ratio'].mean(), inplace=True)
    df['mid_ratio'].fillna(df['mid_ratio'].mean(), inplace=True)
    df['rimmade_rimmiss'].fillna(df['rimmade_rimmiss'].median(), inplace=True)
    df['midmade_midmiss'].fillna(df['midmade_midmiss'].median(), inplace=True)
    df['dunksmiss_dunksmade'].fillna(df['dunksmiss_dunksmade'].median(), inplace=True)
    return df


def drop_unnecessary_columns(df):
    """
    Drop unnecessary columns from the DataFrame.
    """
    print("Dropping unnecessary columns...")
    df.drop(['num', 'ht', 'player_id', 'type'], axis=1, inplace=True)
    return df


def drop_high_missing_columns(df, threshold=40):
    """
    Drop columns with a high percentage of missing values.
    """
    print("Dropping columns with high percentage of missing values...")
    missing_values_df = report_missing_values(df)
    columns_to_drop = missing_values_df[missing_values_df['percentage_missing'] > threshold].index
    df.drop(columns_to_drop, axis=1, inplace=True)
    return df


def handle_duplicates(df):
    """
    Remove duplicate rows from the DataFrame.
    """
    print("Handling duplicates...")
    df.drop_duplicates(inplace=True)
    return df


def encode_categorical_columns(df):
    """
    Encode categorical columns using target encoding.
    """
    print("Encoding categorical columns...")
    target_encoder = TargetEncoder(cols=['team', 'conf'])
    df[['team', 'conf']] = target_encoder.fit_transform(df[['team', 'conf']], df['drafted'])

    joblib.dump(target_encoder, 'target_encoder.pkl')
    return df


def clean_data(df):
    """
    Perform full cleaning process on the DataFrame.
    """
    # Initial shape
    print(f"Initial DataFrame shape: {df.shape}")

    # Impute missing values
    df = impute_missing_values(df)
    print(f"Shape after imputing missing values: {df.shape}")

    # Drop unnecessary columns
    df = drop_unnecessary_columns(df)
    print(f"Shape after dropping unnecessary columns: {df.shape}")

    # Drop columns with high missing values
    df = drop_high_missing_columns(df)
    print(f"Shape after dropping high missing columns: {df.shape}")

    # Handle duplicates
    df = handle_duplicates(df)
    print(f"Shape after handling duplicates: {df.shape}")

    # Filtering specific 'yr' values BEFORE encoding
    valid_yr_values = ['Jr', 'Fr', 'So', 'Sr']
    print(f"Filtering 'yr' values: {valid_yr_values}")
    df = df[df['yr'].isin(valid_yr_values)]
    print(f"Shape after filtering 'yr' values: {df.shape}")

    # Now encode 'yr' as a category (only if the filtering is applied correctly)
    if 'yr' in df.columns:
        df['yr'] = df['yr'].astype('category').cat.codes
        print("Encoded 'yr' as categorical codes.")

    # Encode categorical columns
    df = encode_categorical_columns(df)
    print(f"Shape after encoding categorical columns: {df.shape}")

    # Final DataFrame check
    print(f"Final DataFrame shape: {df.shape}")

    return df
