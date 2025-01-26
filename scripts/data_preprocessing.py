import pandas as pd
import numpy as np
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class DataPreprocessor:
    def __init__(self, input_path: str, output_path: str):
        """
        Initializes the DataPreprocessor.

        Args:
            input_path (str): Path to the raw data directory.
            output_path (str): Path to save the cleaned data.
        """
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.data = None

    def load_data(self, file_name: str):
        """
        Load raw data from a CSV file.

        Args:
            file_name (str): Name of the CSV file to load.
        """
        file_path = self.input_path / file_name
        logging.info(f"Loading data from {file_path}")
        self.data = pd.read_csv(file_path)

    def handle_missing_values(self):
        """
        Handle missing values in the dataset.
        """
        logging.info("Handling missing values...")
        # Example: Fill missing numerical columns with the mean
        numeric_cols = self.data.select_dtypes(include=np.number).columns
        self.data[numeric_cols] = self.data[numeric_cols].fillna(self.data[numeric_cols].mean())

        # Example: Fill missing categorical columns with the mode
        categorical_cols = self.data.select_dtypes(include='object').columns
        self.data[categorical_cols] = self.data[categorical_cols].fillna(self.data[categorical_cols].mode().iloc[0])

    def handle_outliers(self, method='iqr', threshold=3):
        """
        Detect and handle outliers in continuous numerical features.

        Args:
            method (str): Method to detect outliers ('zscore' or 'iqr').
            threshold (float): Threshold for outlier detection.
        """
        logging.info("Handling outliers...")
        # Identify continuous numerical features (exclude binary and categorical features)
        numeric_cols = self.data.select_dtypes(include=np.number).columns
        binary_cols = [col for col in numeric_cols if self.data[col].nunique() == 2]
        continuous_cols = [col for col in numeric_cols if col not in binary_cols]

        logging.info(f"Applying outlier detection to columns: {continuous_cols}")

        if method == 'zscore':
            z_scores = np.abs((self.data[continuous_cols] - self.data[continuous_cols].mean()) / self.data[continuous_cols].std())
            self.data = self.data[(z_scores < threshold).all(axis=1)]
        elif method == 'iqr':
            Q1 = self.data[continuous_cols].quantile(0.25)
            Q3 = self.data[continuous_cols].quantile(0.75)
            IQR = Q3 - Q1
            self.data = self.data[~((self.data[continuous_cols] < (Q1 - 1.5 * IQR)) | (self.data[continuous_cols] > (Q3 + 1.5 * IQR))).any(axis=1)]


    def save_cleaned_data(self, file_name: str):
        """
        Save the cleaned data to a CSV file.

        Args:
            file_name (str): Name of the file to save the cleaned data.
        """
        output_file = self.output_path / file_name
        logging.info(f"Saving cleaned data to {output_file}")
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.data.to_csv(output_file, index=False)

    def run_preprocessing(self, file_name: str, output_file: str):
        """
        Execute the entire preprocessing pipeline.

        Args:
            file_name (str): Input raw data file name.
            output_file (str): Output cleaned data file name.
        """
        self.load_data(file_name)
        self.handle_missing_values()
        self.handle_outliers()
        self.save_cleaned_data(output_file)
        logging.info("Data preprocessing completed successfully!")


if __name__ == "__main__":
    # Example usage
    preprocessor = DataPreprocessor(input_path="../data/raw", output_path="../data/preprocessed/")
    preprocessor.run_preprocessing(file_name="data.csv", output_file="cleaned_data.csv")
