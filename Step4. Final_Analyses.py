import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

# Load the dataset
file_path = '/Users/paritagala/Desktop/Website/5. HR Analytics Dataset from Kaggle/HR_Analytics.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the dataset
data.head()

# Drop EmpID as it is not useful for prediction
data = data.drop('EmpID', axis=1)

# Drop rows with missing values for simplicity
data = data.dropna()

# Encode categorical variables
data = pd.get_dummies(data, drop_first=True)

# Split the data into features and target variable
X = data.drop('Attrition_Yes', axis=1)  # Attrition_Yes is the target variable after encoding
y = data['Attrition_Yes']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
classification_rep = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Feature importance
importances = model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values(by='importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='importance', y='feature', data=feature_importance_df)
plt.title('Feature Importance')
plt.show()

classification_rep
