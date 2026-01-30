# Sentiment Analysis Project

This project performs emotion classification on text data using two approaches: a traditional machine learning model (TF-IDF with Logistic Regression) and a transformer-based model (DistilBERT). The dataset is split into training, validation, and test sets, and the models are evaluated on accuracy, classification reports, and confusion matrices. Exploratory Data Analysis (EDA) is conducted to understand the data distribution.

The project includes Jupyter notebooks for EDA and model comparison, Python scripts for training and prediction, and a Streamlit web app for interactive inference.

## Table of Contents
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Training Models](#training-models)
  - [Making Predictions](#making-predictions)
  - [Running the Streamlit App](#running-the-streamlit-app)
- [Notebooks](#notebooks)
- [Contributing](#contributing)
- [License](#license)

## Project Structure
```
sentiment-analysis/
│
├── data/                  # Directory for dataset files (e.g., training.csv, validation.csv, test.csv)
│
├── src/                   # Source code for models and utilities
│   ├── config.py          # Configuration file (e.g., paths, hyperparameters)
│   │
│   ├── tf_idf/            # TF-IDF + Logistic Regression module
│   │   ├── train.py       # Script to train the TF-IDF model
│   │   ├── predict.py     # Script to make predictions with the TF-IDF model
│   │   └── preprocessing.py # Data preprocessing utilities
│   │
│   └── bert/              # DistilBERT module
│       ├── train.py       # Script to train the DistilBERT model
│       └── predict.py     # Script to make predictions with the DistilBERT model
│
├── models/                # Directory to save trained models
│   ├── tf_idf/            # TF-IDF model artifacts
│   │   ├── sentiment_model.pkl    # Trained Logistic Regression model
│   │   └── tfidf_vectorizer.pkl   # TF-IDF vectorizer
│   │
│   └── bert/              # DistilBERT model artifacts
│       └── distilbert/    # Saved DistilBERT model directory
│
├── app.py                 # Streamlit web app for interactive predictions
├── README.md              # Project documentation
└── requirements.txt       # List of Python dependencies
```

## Requirements
- Python 3.10+
- Key libraries: pandas, scikit-learn, transformers, torch, streamlit, matplotlib, seaborn

See `requirements.txt` for the full list.

## Installation
1. Clone the repository:
   ```
   git clone https://github.com/kamalmustafayev07/sentiment-analysis-ferrum-capital
   cd sentiment-analysis
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Training Models
Train the models using the provided scripts. Ensure your dataset is placed in the `data/` directory.

- Train the TF-IDF + Logistic Regression model:
  ```
  python -m src.tf_idf.train
  ```

- Train the DistilBERT model:
  ```
  python -m src.bert.train
  ```

Trained models will be saved in the `models/` directory.

### Making Predictions
Run predictions on the test data or new inputs.

- Predict with the TF-IDF model:
  ```
  python -m src.tf_idf.predict
  ```

- Predict with the DistilBERT model:
  ```
  python -m src.bert.predict
  ```

These scripts will load the test data, make predictions, and output evaluation metrics (e.g., accuracy, classification report).

### Running the Streamlit App
Launch the interactive web app to classify emotions in custom text inputs.

```
streamlit run app.py
```

Open your browser at `http://localhost:8501` to use the app.

## Notebooks
- `01_eda.ipynb`: Exploratory Data Analysis, including data loading, basic statistics, visualizations (e.g., label distribution, text length boxplots).
- `02_model_training.ipynb`: Model training, evaluation, and comparison between TF-IDF + Logistic Regression and DistilBERT. Includes confusion matrices and performance metrics.

To run the notebooks, ensure Jupyter is installed (`pip install notebook`) and launch with:
```
jupyter notebook
```

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
