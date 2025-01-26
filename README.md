# BatiBank Credit Scoring Model

## Project Overview
The BatiBank project is aimed at developing a credit scoring model for a buy-now-pay-later service in partnership with an eCommerce platform. The model will classify users as high-risk or low-risk borrowers based on their likelihood of defaulting on payments. Additionally, the model will provide insights into optimal loan amounts and durations.

## Folder Structure
```
├── .vscode/
│   └── settings.json
├── .github/
│   └── workflows/
│       └── unittests.yml
├── .dvc/
├── data/
│   ├── preprocessed/
│   ├── raw/
├── .gitignore
├── requirements.txt
├── README.md
├── dvc.yml
├── src/
│   └── __init__.py
├── notebooks/
│   ├── __init__.py
│   └── README.md
├── tests/
│   └── __init__.py
└── scripts/
    ├── __init__.py
    └── README.md
```

## Key Components
1. **Data Ingestion**: Raw data is loaded and stored in the `data/raw/` directory.
2. **Data Preprocessing**: Data cleaning and transformation steps are performed, with outputs saved in the `data/preprocessed/` directory.
3. **Exploratory Data Analysis (EDA)**: Insights are derived from the data to inform feature engineering and model development.
4. **Feature Engineering**: New features are created to enhance the predictive power of the model.
5. **Model Training and Evaluation**: Various machine learning models are trained and fine-tuned.
6. **Model Deployment**: A REST API serves the trained models for real-time predictions.

## Prerequisites
- Python 3.8+
- Required Python libraries (see `requirements.txt`)
- DVC (Data Version Control)
- Git

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd BatiBank
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Initialize DVC:
   ```bash
   dvc init
   ```
4. Pull the dataset and preprocessed data:
   ```bash
   dvc pull
   ```

## Usage
### Running the Scripts
- **Data Ingestion**:
  ```bash
  python scripts/data_ingestion.py
  ```
- **Data Preprocessing**:
  ```bash
  python scripts/data_preprocessing.py
  ```
- **Exploratory Data Analysis (EDA)**:
  Utilize the `EDA` class from the `eda_script.py` to perform EDA interactively in a notebook.

### Notebooks
Jupyter notebooks are provided in the `notebooks/` directory for interactive exploration and visualization.

## Project Highlights
- Modular code using Object-Oriented Programming (OOP).
- Scripts and functions designed to be reusable and callable within notebooks.
- Comprehensive EDA including:
  - Summary statistics
  - Visualizations for numerical and categorical features
  - Correlation analysis
  - Outlier detection
  - Missing value identification

## Roadmap
1. Complete advanced feature engineering and RFMS-based classification.
2. Train and evaluate machine learning models.
3. Develop a model-serving API using FastAPI or Flask.
4. Deploy the solution to a cloud platform for real-time usage.

## Contributing
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch.
3. Submit a pull request with detailed information about your changes.

## License
This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Contact
For questions or feedback, please contact the project team at [team@example.com](mailto:team@example.com).

