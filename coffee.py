# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import math

# Load the dataset
file_name = '/content/coffee_recommendation_dataset.xlsx'  # Make sure this file is uploaded in Colab
df = pd.read_excel(file_name)

# Data Preprocessing

# Encode categorical variables
label_encoder = LabelEncoder()

# Apply label encoding to the target column (Label)
df['Label'] = label_encoder.fit_transform(df['Label'])

# Apply one-hot encoding to the feature columns (Token_0 to Token_9)
df_encoded = pd.get_dummies(df, columns=[f'Token_{i}' for i in range(10)], drop_first=True)

# Splitting the dataset into training and test sets
X = df_encoded.drop('Label', axis=1)
y = df_encoded['Label']

# Split data into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing the data (optional, for certain models like SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Machine Learning Models
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=500),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "SVM": SVC(random_state=42)
}

# Training and evaluating the models
results = {}

for model_name, model in models.items():
    print(f"Training {model_name}...")
    # Use scaled data for SVM, for others, use non-scaled data
    if model_name == "SVM":
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
    # Evaluation metrics
    r2 = r2_score(y_test, y_pred)
    rmse = math.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    results[model_name] = {
        "R2 Score": r2,
        "RMSE": rmse,
        "MAE": mae
    }
    
    print(f"{model_name} Performance:")
    print(f"R2 Score: {r2}")
    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}\n")

# Display the results
print("Summary of Model Performance:")
for model_name, metrics in results.items():
    print(f"{model_name}:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value}")
    print()
