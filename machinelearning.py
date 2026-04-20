import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 1: Read the CSV file
df = pd.read_csv("income.csv")

# Step 2: Show first 5 rows
print(df.head())

# Step 3: Check total rows and columns
print("Shape of Dataset:", df.shape)

# Step 4: Check missing values
print(df.isnull().sum())

# Step 5: Replace ? with empty values
df.replace("?", np.nan, inplace=True)

# Step 6: Remove missing rows
df.dropna(inplace=True)

# Step 7: Remove spaces from text columns
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].str.strip()

# Step 8: Show income column values
print(df['SalStat'].value_counts())

# Step 9: Graph 1 - Income Count
sns.countplot(x='SalStat', data=df)
plt.title("Income Distribution")
plt.show()

# Step 10: Graph 2 - Age vs Income
sns.boxplot(x='SalStat', y='age', data=df)
plt.title("Age vs Income")
plt.show()

# Step 11: Convert text into numbers
label_encoder = LabelEncoder()

for col in df.select_dtypes(include='object').columns:
    df[col] = label_encoder.fit_transform(df[col])

# Step 12: Separate input and output
X = df.drop('SalStat', axis=1)
y = df['income']

# Step 13: Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 14: Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Step 15: Predict
y_pred = model.predict(X_test)

# Step 16: Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Step 17: Full Report
print(classification_report(y_test, y_pred))

# Step 18: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
