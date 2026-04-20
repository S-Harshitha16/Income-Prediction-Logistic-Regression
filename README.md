# Income Prediction using Logistic Regression

## Overview
This project predicts whether a person's income exceeds 50K per year using machine learning techniques. It is built using the Adult Income Dataset and applies Logistic Regression for classification.

## Tech Stack
- Python  
- Pandas  
- NumPy  
- Scikit-learn  

## Dataset
- **Name:** income.csv  
- **Description:** Adult Income dataset used to predict whether income is greater than 50K or not.  

### Features:
- age  
- JobType  
- EdType  
- maritalstatus  
- occupation  
- relationship  
- race  
- gender  
- capitalgain  
- capitalloss  
- hoursperweek  
- nativecountry  

### Target Variable:
- SalStat (less than or equal to 50K / greater than 50K)
## Model
- Logistic Regression (Scikit-learn)

## Data Preprocessing
- Handled missing values  
- Encoded categorical variables  
- Feature scaling applied  
- Train-test split performed  

## Output / Result
![Income Graph](IncomeGraph.png)
- Model Accuracy: 85% (update with your actual value)  
- Model predicts whether income is >50K or <=50K  
- Performance evaluated using classification report  
