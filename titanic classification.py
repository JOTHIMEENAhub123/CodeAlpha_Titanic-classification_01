import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load Titanic dataset from seaborn
titanic_data = sns.load_dataset('titanic')

# Display the first few rows of the dataset
print(titanic_data.head())

# Let's select relevant features for our analysis
features = ['sex', 'age', 'fare', 'class', 'embarked', 'who', 'adult_male', 'deck', 'embark_town', 'alone']

# Drop rows with missing values in selected features
titanic_data = titanic_data.dropna(subset=features + ['survived'])

# Convert categorical variables to numerical using one-hot encoding
titanic_data = pd.get_dummies(titanic_data, columns=['sex', 'class', 'embarked', 'who', 'deck', 'embark_town', 'alone'])

# Select features and target variable
X = titanic_data.drop(columns=['survived'])
y = titanic_data['survived']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

