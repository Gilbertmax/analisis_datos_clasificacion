# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
           'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
           'hours-per-week', 'native-country', 'income']

data = pd.read_csv(url, names=columns, na_values=" ?", skipinitialspace=True)

# Step 2: Data Cleaning and Preprocessing
# Fill missing values with mode (most frequent value)
data.fillna(data.mode().iloc[0], inplace=True)

# Convert categorical columns to numeric using one-hot encoding
data = pd.get_dummies(data, drop_first=True)

# Step 3: Feature Scaling
scaler = StandardScaler()
scaled_features = scaler.fit_transform(data.drop('income_>50K', axis=1))

# Convert scaled features back to a DataFrame
scaled_data = pd.DataFrame(scaled_features, columns=data.columns[:-1])

# Step 4: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(scaled_data, data['income_>50K'], test_size=0.3, random_state=42)

# Step 5: Model Training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 6: Predictions and Evaluation
y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Step 7: Visualization of Feature Importance
feature_importances = model.feature_importances_
features = X_train.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plotting the feature importances
plt.figure(figsize=(10, 15))  
top_n = 20  
sns.barplot(x='Importance', y='Feature', data=importance_df.head(top_n))  
plt.title('Top 20 Feature Importance in RandomForest Model')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()
