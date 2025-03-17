# ❤️ Heart Disease Prediction Using Machine Learning

This project is an end-to-end machine learning pipeline designed to predict the likelihood of heart disease based on patient data. It includes data preprocessing, model selection, hyperparameter tuning, evaluation, and deployment using Python and popular ML libraries.

## 📂 Dataset
The dataset used in this project comes from Kaggle:
- [Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)

It includes multiple health indicators for patients, such as:
- Age
- Sex
- Chest pain type (cp)
- Resting blood pressure (trestbps)
- Serum cholesterol (chol)
- Fasting blood sugar (fbs)
- Resting electrocardiographic results (restecg)
- Maximum heart rate achieved (thalach)
- Exercise-induced angina (exang)
- ST depression induced by exercise relative to rest (oldpeak)
- Slope of the peak exercise ST segment (slope)
- Number of major vessels (ca)
- Thalassemia (thal)
- Target (1 = presence of heart disease, 0 = absence of heart disease)

---

## 🛠️ Tools & Libraries
- **Python 3.x**
- **Pandas**, **NumPy** for data manipulation
- **Matplotlib**, **Seaborn** for data visualization
- **Scikit-learn** for machine learning models and pipelines
- **XGBoost** for advanced gradient boosting models
- **Imbalanced-learn (SMOTE)** for handling class imbalance
- **Joblib** for saving and loading the model pipeline

---

## ⚙️ Workflow

1. **Dataset Download & Load**
   - The dataset is automatically downloaded from Kaggle and loaded into a Pandas DataFrame.

2. **Exploratory Data Analysis (EDA)**
   - Overview of the dataset
   - Missing value detection
   - Correlation heatmap for understanding feature relationships

3. **Data Preprocessing**
   - Splitting data into train and test sets
   - Feature scaling using `StandardScaler`
   - Addressing class imbalance with SMOTE

4. **Model Training & Hyperparameter Tuning**
   - **Logistic Regression** (L1 & L2 regularization)
   - **Random Forest** (various depths and estimators)
   - **XGBoost** (with tuning of estimators, learning rate, depth)

5. **Evaluation**
   - Accuracy score
   - Confusion matrix
   - ROC-AUC score & ROC Curve
   - Feature importance (for tree-based models)

6. **Deployment**
   - Best model (XGBoost) wrapped in a Scikit-learn `Pipeline`
   - Saved pipeline using `joblib`
   - Predictions demonstrated on custom patient data

---

## 🚀 Results

- Best performing model: **XGBoost Classifier**
- Cross-validated accuracy (on training): 85.39%
- Test accuracy: 98.34%
- ROC-AUC score: 98.46%
- Feature importance visualization shows which factors most impact heart disease predictions.

---
