import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import sys
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from data_preprocessing import load_data, preprocess_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_model(data_path, target_column, model_params):
    mlflow.set_experiment("MLOps-Streamline-Training")
    with mlflow.start_run():
        logging.info("MLflow run started.")
        mlflow.log_params(model_params)

        df = load_data(data_path)
        X_train, X_test, y_train, y_test, scaler = preprocess_data(df, target_column)

        model = LogisticRegression(**model_params)
        model.fit(X_train, y_train)
        logging.info("Model training complete.")

        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)

        mlflow.log_metrics({
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        })
        logging.info(f"Metrics logged: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")

        mlflow.sklearn.log_model(model, "model")
        logging.info("Model logged to MLflow.")

        # Save scaler for later use in inference
        import joblib
        scaler_path = "scaler.joblib"
        joblib.dump(scaler, scaler_path)
        mlflow.log_artifact(scaler_path)
        os.remove(scaler_path)
        logging.info("Scaler saved as artifact.")

        mlflow.end_run()
        logging.info("MLflow run ended.")

if __name__ == "__main__":
    # Dummy data path for demonstration
    dummy_data = {
        'feature_1': np.random.rand(100),
        'feature_2': np.random.randint(0, 10, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100),
        'target': np.random.randint(0, 2, 100)
    }
    dummy_df = pd.DataFrame(dummy_data)
    dummy_filepath = "dummy_dataset_train.csv"
    dummy_df.to_csv(dummy_filepath, index=False)

    model_params = {
        "solver": "liblinear",
        "random_state": 42
    }
    try:
        train_model(dummy_filepath, 'target', model_params)
        print("Training pipeline executed successfully.")
    except Exception as e:
        logging.error(f"Training pipeline failed: {e}")
    finally:
        os.remove(dummy_filepath)
