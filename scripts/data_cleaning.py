import pandas as pd
import numpy as np
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class DataCleaner:
    def __init__(self, input_path: str, output_path: str):
        """
        Initializes the DataCleaner class.

        Args:
            input_path (str): Path to the preprocessed data directory.
            output_path (str): Path to save the final cleaned data.
        """
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.data = None

    def load_data(self, file_name: str):
        """
        Load data from a CSV file.

        Args:
            file_name (str): Name of the CSV file to load.
        """
        file_path = self.input_path / file_name
        logging.info(f"Loading data from {file_path}")
        self.data = pd.read_csv(file_path)
        return self.data

    def convert_data_types(self):
        """
        Convert data types based on observations.
        """
        logging.info("Converting data types...")
        # Convert TransactionStartTime to datetime
        self.data["TransactionStartTime"] = pd.to_datetime(self.data["TransactionStartTime"])
        logging.info("Converted TransactionStartTime to datetime.")

    def drop_redundant_columns(self):
        """
        Drop columns with single unique values or deemed redundant.
        """
        logging.info("Dropping redundant columns...")
        redundant_cols = ["CurrencyCode", "CountryCode"]
        self.data.drop(columns=redundant_cols, inplace=True)
        logging.info(f"Dropped columns: {redundant_cols}")

    def handle_outliers(self, columns, method="cap", cap_percentile=99):
        """
        Handle outliers in the specified columns.

        Args:
            columns (list): List of column names to handle outliers.
            method (str): Outlier handling method ('cap' or 'remove').
            cap_percentile (int): Percentile value for capping outliers.
        """
        logging.info("Handling outliers...")
        for col in columns:
            if method == "cap":
                upper_cap = np.percentile(self.data[col], cap_percentile)
                self.data[col] = np.where(self.data[col] > upper_cap, upper_cap, self.data[col])
                logging.info(f"Capped outliers in {col} at {cap_percentile}th percentile.")
            elif method == "remove":
                q1, q3 = np.percentile(self.data[col], [25, 75])
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                self.data = self.data[(self.data[col] >= lower_bound) & (self.data[col] <= upper_bound)]
                logging.info(f"Removed outliers in {col} using IQR method.")

    def handle_multicollinearity(self):
        """
        Address multicollinearity by dropping one of the highly correlated variables.
        """
        logging.info("Handling multicollinearity...")
        self.data.drop(columns=["Value"], inplace=True)  # Retain "Amount"
        logging.info("Dropped column 'Value' due to high correlation with 'Amount'.")

    def group_categorical_data(self, column, threshold=100):
        """
        Group underrepresented categories in a categorical column.

        Args:
            column (str): Name of the column to group.
            threshold (int): Minimum count threshold to keep a category.
        """
        logging.info(f"Grouping underrepresented categories in {column}...")
        value_counts = self.data[column].value_counts()
        rare_categories = value_counts[value_counts < threshold].index
        self.data[column] = self.data[column].apply(lambda x: "Other" if x in rare_categories else x)
        logging.info(f"Grouped rare categories in {column} into 'Other'.")

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

    def run_cleaning(self, file_name: str, output_file: str):
        """
        Execute the entire data cleaning pipeline.

        Args:
            file_name (str): Input preprocessed data file name.
            output_file (str): Output cleaned data file name.
        """
        self.load_data(file_name)
        self.convert_data_types()
        self.drop_redundant_columns()
        self.handle_outliers(columns=["Amount", "Value"], method="cap", cap_percentile=99)
        self.handle_multicollinearity()
        self.group_categorical_data(column="ProductId", threshold=100)
        self.group_categorical_data(column="ProductCategory", threshold=500)
        self.save_cleaned_data(output_file)
        logging.info("Data cleaning completed successfully!")

if __name__ == "__main__":
    cleaner = DataCleaner(input_path="data/raw", output_path="data/preprocessed/cleaned")
    cleaner.run_cleaning(file_name="data.csv", output_file="final_cleaned_data.csv")
