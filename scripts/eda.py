# scripts/eda.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class EDA:
    def __init__(self, input_path: str, output_path: str):
        """
        Initializes the EDA class.

        Args:
            input_path (str): Path to the raw data directory.
            output_path (str): Path to save EDA results and plots.
        """
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.data = None

    def load_data(self, file_name: str):
        """
        Load data from a CSV file and ensure proper data types.
        """
        file_path = self.input_path / file_name
        logging.info(f"Loading data from {file_path}")
        self.data = pd.read_csv(file_path)

        logging.info("Data loaded successfully.")
        return self.data


    def overview(self):
        """
        Display an overview of the dataset.
        """
        logging.info("Dataset Overview:")
        info = self.data.info()
        shape = self.data.shape
        duplicates = self.data.duplicated().sum()
        logging.info(f"Shape: {shape}")
        logging.info(f"Number of duplicate rows: {duplicates}")
        return {"info": info, "shape": shape, "duplicates": duplicates}

    def summary_statistics(self):
        """
        Generate summary statistics.
        """
        logging.info("Summary Statistics:")
        stats = self.data.describe(include="all").transpose()
        logging.info(stats)
        return stats


    def visualize_numerical_distributions(self):
        """
        Visualize the distribution of numerical features.
        """
        numeric_cols = self.data.select_dtypes(include="number").columns

        for col in numeric_cols:
            plt.figure(figsize=(8, 4))
            sns.histplot(self.data[col], kde=True, bins=30)
            plt.title(f"Distribution of {col}")
            plt.show()

    def visualize_categorical_distributions(self, unique_value_threshold=30):
        """
        Visualize the distribution of categorical features with manageable unique values.
        """
        categorical_cols = self.data.select_dtypes(include="object").columns
        for col in categorical_cols:
            if self.data[col].nunique() <= unique_value_threshold:
                plt.figure(figsize=(8, 4))
                sns.countplot(data=self.data, x=col, order=self.data[col].value_counts().index)
                plt.title(f"Distribution of {col}")
                plt.xticks(rotation=45)
                plt.show()
            else:
                logging.info(f"Skipping column {col} with {self.data[col].nunique()} unique values.")


    def correlation_analysis(self):
        """
        Perform correlation analysis.
        """
        numeric_cols = self.data.select_dtypes(include="number").columns
        corr = self.data[numeric_cols].corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Heatmap")
        plt.show()

    def missing_values_analysis(self):
        """
        Analyze missing values in the dataset.
        """
        missing = self.data.isnull().sum()
        missing = missing[missing > 0]
        if not missing.empty:
            logging.info("Missing Values Summary:")
            logging.info(missing)
        else:
            logging.info("No missing values detected.")
        return missing

    def detect_outliers(self):
        """
        Detect outliers using box plots.
        """
        numeric_cols = self.data.select_dtypes(include="number").columns
        for col in numeric_cols:
            plt.figure(figsize=(8, 4))
            sns.boxplot(data=self.data, x=col)
            plt.title(f"Outliers in {col}")
            plt.show()
