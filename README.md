# Telecom Customer Churn Prediction

This repository contains a comprehensive solution to predict customer churn in the telecom sector using machine learning. The project includes data preprocessing, exploratory data analysis, feature engineering, model building, evaluation, and interpretability of results. The primary goal is to help telecom companies understand why customers leave and proactively reduce churn rates.

## Features

- **Preprocessing & EDA:** Clean and explore the raw telecom customer dataset.
- **Model Building:** Multiple ML models (Logistic Regression, Decision Trees, Random Forest, XGBoost, etc.) are trained and compared.
- **Evaluation:** Performance metrics like accuracy, precision, recall, F1-score, and ROC-AUC are used to select the best model.
- **Interpretability:** Feature importance and SHAP analysis to interpret model predictions.
- **Notebook-based Workflow:** All steps are documented in a Jupyter Notebook for reproducibility and clarity.

## Project Structure

```
.
├── Notebook/
│   └── customer_churn_prediction.ipynb  # Main notebook with code and analysis
├── data/                               # (Not included) Place for your dataset(s)
├── README.md                           # Project overview and instructions
└── requirements.txt                    # Python dependencies (if available)
└── app.py                              # Used StreamLit for UI & Backend
```

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/OkYashGajjar/telecom-customer-churn-prediction.git
cd telecom-customer-churn-prediction
```

### 2. Prepare the Environment

It is recommended to use a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```
> If requirements.txt is missing, install main packages:
> `pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap`

### 3. Data Preparation

- Download the telecom customer churn dataset (for example, from [Kaggle](https://www.kaggle.com/blastchar/telco-customer-churn) or your own source).
- Place it in the `data/` directory or update the notebook's data path accordingly.

### 4. Run the Notebook

Open the notebook and follow along (place the path of the dataset):

```bash
jupyter notebook Notebook/customer_churn_prediction.ipynb
```
### 4. Run the app.py

StreaLit Used for UI & Integration of Model
```bash
python -m streamlit run app.py
```
## Results

- **Best Model:** The notebook compares several models and highlights the one with the best performance.
- **Insights:** Key drivers of churn are identified, and visualizations are provided for business understanding.

## Contributing

Feel free to open issues or submit pull requests for improvements, new features, or bug fixes.

## License

This project is open-source and available under the [MIT License](LICENSE).

## Acknowledgements

- [Telco Customer Churn Dataset](https://www.kaggle.com/blastchar/telco-customer-churn) (if used)
- Scikit-learn, XGBoost, SHAP, and related open-source libraries.

---

> **Author:** [Yash Gajjar](https://github.com/OkYashGajjar)
