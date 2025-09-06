import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_data(file_path):
    """Loads data from a CSV file."""
    return pd.read_csv(file_path)

def preprocess_data(df):
    """Performs basic data preprocessing."""
    # Handle missing values
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())

    # Encode categorical features
    label_encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    return df, label_encoders

def feature_engineering(df):
    """Creates new features from existing ones."""
    df['CREDIT_INCOME_PERCENT'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    df['ANNUITY_INCOME_PERCENT'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['CREDIT_TERM'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    df['DAYS_EMPLOYED_PERCENT'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    return df

if __name__ == '__main__':
    # Load the data
    data = load_data('data/raw/application_train.csv')

    # Preprocess the data
    data, _ = preprocess_data(data)

    # Feature engineering
    data = feature_engineering(data)

    # Split the data
    train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)

    # Save the processed data
    train_df.to_csv('data/processed/train.csv', index=False)
    test_df.to_csv('data/processed/test.csv', index=False)

    print("Data preprocessing complete.")

