# Titanic Data Exploration and Model Building

## Project Overview

This project performs **Exploratory Data Analysis (EDA)** on the Titanic dataset and builds two predictive models (Logistic Regression and Decision Tree) to predict passenger survival based on various features.

### Key Steps:
1. **Data Preprocessing**: Handle missing values, encode categorical variables, and prepare the dataset for model training. 
2. **Exploratory Data Analysis (EDA)**: Analyze relationships between features and the target variable (survival).
3. **Model Building**: Train two models: 
   - **Logistic Regression**: A simple yet effective linear model for binary classification.
   - **Decision Tree**: A non-linear model that provides more interpretability.

### Key Libraries Used:
- `pandas`: For data manipulation.
- `numpy`: For numerical operations.
- `matplotlib`, `seaborn`: For data visualization.
- `scikit-learn`: For machine learning algorithms and model evaluation.

## Dataset
The dataset used is the Titanic dataset, available at [Kaggle Titanic dataset](https://www.kaggle.com/c/titanic/data).

## Steps Involved

1. **Data Preprocessing**:
   - Filled missing 'Age' values with the median.
   - Dropped rows with missing 'Embarked' values.
   - Encoded categorical features ('Sex' and 'Embarked') as numeric values.

2. **Model Building**:
   - **Logistic Regression** and **Decision Tree** classifiers were trained to predict passenger survival.
   - The models were evaluated based on accuracy, confusion matrix, and classification report.

## Results
- The models achieved a solid accuracy on the Titanic dataset, with evaluation metrics provided for both Logistic Regression and Decision Tree classifiers.

## Future Improvements
- Try advanced models like Random Forest or XGBoost.
- Perform hyperparameter tuning using GridSearchCV for better performance.
- Experiment with more feature engineering and missing value handling strategies.

## Installation and Usage

### Prerequisites:
- Python 3.x
- Required libraries can be installed via `pip`:
  ```bash
  pip install pandas numpy scikit-learn matplotlib seaborn
