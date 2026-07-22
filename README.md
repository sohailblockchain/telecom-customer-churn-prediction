# Telecom Customer Churn Prediction

An end-to-end machine learning project that predicts whether a telecom customer is likely to leave a service provider. The project includes data preprocessing, model training, evaluation, model persistence, and an interactive Streamlit application for real-time predictions.

## Project Overview

Customer churn is an important business problem because retaining an existing customer is often more cost-effective than acquiring a new one. This project uses historical customer information to estimate churn risk and help identify customers who may require retention support.

## Key Features

- Data cleaning and preprocessing
- Missing-value handling
- Numerical feature scaling
- Categorical feature encoding with OneHotEncoder
- Random Forest classification
- Train-test split with stratification
- Evaluation using Accuracy, Precision, Recall, F1-score, Confusion Matrix, and ROC-AUC
- Saved machine learning pipeline using Joblib
- Interactive Streamlit user interface
- Real-time churn probability prediction

## Model Performance

| Metric | Result |
|---|---:|
| Accuracy | 78.07% |
| ROC-AUC | 83.47% |
| Precision for Churn | 58% |
| Recall for Churn | 66% |
| F1-score for Churn | 61% |
| Test Records | 1,409 |

## Confusion Matrix

```text
[[854 181]
 [128 246]]
```

Interpretation:

- 854 customers who stayed were correctly classified.
- 246 customers who churned were correctly classified.
- 181 customers were incorrectly predicted to churn.
- 128 churned customers were missed by the model.

## Technology Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Joblib

## Dataset

The project uses the IBM Telco Customer Churn dataset. It contains customer demographic, account, billing, contract, and service-related information.

Important features include:

- Gender
- Senior citizen status
- Partner and dependents
- Tenure
- Phone and internet services
- Online security and technical support
- Contract type
- Payment method
- Monthly charges
- Total charges
- Churn status

## Machine Learning Workflow

1. Load and validate the dataset.
2. Convert `TotalCharges` to numeric values.
3. Remove `customerID` because it is only an identifier.
4. Convert the target variable from `Yes/No` to `1/0`.
5. Separate numerical and categorical columns.
6. Handle missing values using Scikit-learn imputers.
7. Scale numerical features.
8. Encode categorical features with one-hot encoding.
9. Split data into training and testing sets using stratification.
10. Train a balanced Random Forest classifier.
11. Evaluate the model with multiple classification metrics.
12. Save the complete preprocessing and prediction pipeline.
13. Use the saved pipeline in a Streamlit application.

## Project Structure

```text
telecom-customer-churn-prediction/
├── data/
│   └── Telco-Customer-Churn.csv
├── models/
│   └── churn_model.joblib
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

## Installation

Clone the repository:

```bash
git clone https://github.com/sohailblockchain/telecom-customer-churn-prediction.git
cd telecom-customer-churn-prediction
```

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Train the Model

```bash
python train_model.py
```

The trained pipeline will be saved as:

```text
models/churn_model.joblib
```

## Run the Streamlit Application

```bash
streamlit run app.py
```

Open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

## How the Application Works

The user enters customer information such as tenure, contract type, internet service, monthly charges, and payment method. The application then returns:

- Predicted churn class
- Estimated churn probability
- A clear risk message

The result is decision-support information and should not be treated as a guaranteed business outcome.

## Skills Demonstrated

- Data preprocessing
- Feature engineering
- Classification
- Imbalanced-class handling
- Scikit-learn pipelines
- Model evaluation
- Model serialization
- Streamlit deployment
- Git and GitHub project documentation

## Future Improvements

- Compare Random Forest with Logistic Regression and XGBoost
- Add cross-validation and hyperparameter tuning
- Add feature-importance visualization
- Add ROC curve and confusion-matrix charts
- Improve churn-class recall
- Add batch CSV predictions
- Deploy the application on Streamlit Community Cloud
- Add automated tests

## Author

**Sohail Ahmed**  
Senior Software Engineer | Blockchain Engineer | Data Science and AI Practitioner
