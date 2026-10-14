# Bank Loan Default Prediction

Predicting whether a loan applicant is likely to default, using customer loan application data. The project covers data cleaning, exploratory analysis, handling a heavily imbalanced target, and comparing several classification models. A small Flask script is included for scoring applications.

![Loan default prediction](Payday-Loan.gif)

## Dataset

`application_train.csv` and `application_test.csv` hold one row per loan application: applicant demographics, income, loan amount and contract type, housing and family status, and counts of defaults in the applicant's social circle. The target flags whether the client had payment difficulties.

## Project structure

| File | What it does |
|---|---|
| `Data Preprocessing.ipynb` | Missing-value analysis, cleaning, one-hot encoding, scaling and train/test split |
| `Exploratory Data Analysis (EDA).ipynb` | Target balance, income distribution, gender, family situation, car and realty ownership, loan types |
| `Machine Learning Models.ipynb` | Resampling, model training and evaluation |
| `flask_file.py` | Flask app for scoring a loan application |
| `images/` | Plots and result screenshots |

## Approach

1. Clean the data and handle missing values
2. Explore the main drivers of default with EDA
3. Handle class imbalance with RandomOverSampler, SMOTE, Tomek Links and NearMiss
4. Train Logistic Regression, Gaussian Naive Bayes, Decision Tree and Random Forest models, tuning with GridSearchCV
5. Compare models with confusion matrices, classification reports and ROC/AUC curves

## Results

ROC curves and performance plots for each model are in [`images/`](images). Example, Random Forest with SMOTE:

![Random Forest with SMOTE ROC curves](images/RF%20SMOTE%20AUC%20Curves.png)

## Tech stack

Python, pandas, NumPy, scikit-learn, imbalanced-learn, Matplotlib, seaborn, missingno, Flask, Jupyter

## Run locally

```bash
git clone https://github.com/himhjr1018/Bank-Loan-Default-Prediction.git
cd Bank-Loan-Default-Prediction
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn missingno flask jupyter
jupyter notebook
```

Run the notebooks in order: preprocessing, EDA, then models.

## Author

**Himanshu Rao** · [LinkedIn](https://www.linkedin.com/in/himanshu-rao-/)
