import pandas as pd

# Load the dataset
file_path = '/Users/paritagala/Desktop/Website/5. HR Analytics Dataset from Kaggle/HR_Analytics.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the dataset
data.head()

print (data)