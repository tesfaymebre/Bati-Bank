import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class EDA:
    def __init__(self, file_path):
        """Initialize with the file path to the dataset."""
        self.file_path = file_path
        self.data = None

    def load_data(self):
        """Load the dataset from the file path."""
        self.data = pd.read_csv(self.file_path)
        return self.data

    def data_overview(self):
        """Provide an overview of the dataset."""
        print("\nDataset Information:\n")
        print(self.data.info())
        print("\nFirst 5 Rows:\n", self.data.head())
        print("\nBasic Statistics:\n", self.data.describe(include='all'))

    def visualize_numerical_distribution(self, columns=None):
        """Visualize distributions of numerical features."""
        numerical_data = self.data.select_dtypes(include=['float64', 'int64'])
        if columns:
            numerical_data = numerical_data[columns]

        for column in numerical_data.columns:
            plt.figure(figsize=(8, 4))
            sns.histplot(self.data[column], kde=True, bins=30)
            plt.title(f"Distribution of {column}")
            plt.show()

    def visualize_categorical_distribution(self, columns=None):
        """Visualize distributions of categorical features."""
        categorical_data = self.data.select_dtypes(include=['object', 'category'])
        columns = ["CurrencyCode", "ProviderId", "ChannelId", "ProductCategory"]
        if columns:
            categorical_data = categorical_data[columns]

        for column in categorical_data.columns:
            plt.figure(figsize=(8, 4))
            sns.countplot(data=self.data, x=column, order=self.data[column].value_counts().index)
            plt.title(f"Distribution of {column}")
            plt.xticks(rotation=45)
            plt.show()

    def correlation_analysis(self):
        """Analyze correlations between numerical features."""
        numerical_data = self.data.select_dtypes(include=['float64', 'int64'])
        correlation_matrix = numerical_data.corr()

        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title("Correlation Matrix")
        plt.show()

    def identify_missing_values(self):
        """Identify missing values in the dataset."""
        missing_values = self.data.isnull().sum()
        missing_percentage = (missing_values / len(self.data)) * 100
        missing_summary = pd.DataFrame({
            "Missing Values": missing_values,
            "% Missing": missing_percentage
        })
        print("\nMissing Values Summary:\n", missing_summary)

    def detect_outliers(self, columns=None):
        """Detect outliers using box plots."""
        numerical_data = self.data.select_dtypes(include=['float64', 'int64'])
        if columns:
            numerical_data = numerical_data[columns]

        for column in numerical_data.columns:
            plt.figure(figsize=(8, 4))
            sns.boxplot(x=self.data[column])
            plt.title(f"Outliers in {column}")
            plt.show()
