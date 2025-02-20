import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(filepath):
    logging.info(f"Loading data from {filepath}")
    try:
        df = pd.read_csv(filepath)
        logging.info(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def preprocess_data(df, target_column, test_size=0.2, random_state=42):
    logging.info("Starting data preprocessing...")
    
    if target_column not in df.columns:
        logging.error(f"Target column '{target_column}' not found in data.")
        raise ValueError(f"Target column '{target_column}' not found.")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Handle categorical features (simple one-hot encoding for demonstration)
    X = pd.get_dummies(X, drop_first=True)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    logging.info(f"Data split into training ({len(X_train)} samples) and testing ({len(X_test)} samples).")

    # Scale numerical features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    logging.info("Numerical features scaled.")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

if __name__ == "__main__":
    # Create a dummy CSV file for demonstration
    dummy_data = {
        'feature_1': np.random.rand(100),
        'feature_2': np.random.randint(0, 10, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100),
        'target': np.random.randint(0, 2, 100)
    }
    dummy_df = pd.DataFrame(dummy_data)
    dummy_filepath = "dummy_dataset.csv"
    dummy_df.to_csv(dummy_filepath, index=False)
    logging.info(f"Created dummy dataset at {dummy_filepath}")

    try:
        df = load_data(dummy_filepath)
        X_train, X_test, y_train, y_test, scaler = preprocess_data(df, 'target')
        logging.info("Data preprocessing complete for dummy dataset.")
        print(f"X_train shape: {X_train.shape}")
        print(f"X_test shape: {X_test.shape}")
    except Exception as e:
        logging.error(f"Script failed: {e}")
    finally:
        os.remove(dummy_filepath)
        logging.info(f"Removed dummy dataset {dummy_filepath}")
