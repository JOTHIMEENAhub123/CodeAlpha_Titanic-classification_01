import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Generate synthetic data (replace this with your real dataset)
np.random.seed(0)
n_samples = 1000
age = np.random.randint(18, 80, n_samples)
gender = np.random.choice(['male', 'female'], n_samples)
socioeconomic_status = np.random.choice(['low', 'medium', 'high'], n_samples)
survived = np.random.randint(0, 2, n_samples)

# Encode categorical variables
gender_encoded = np.where(gender == 'male', 0, 1)
socioeconomic_status_encoded = np.where(socioeconomic_status == 'low', 0,
                                         np.where(socioeconomic_status == 'medium', 1, 2))

# Prepare feature matrix and target vector
X = np.column_stack((age, gender_encoded, socioeconomic_status_encoded))
y = survived

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)