import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Load dataset
df = pd.read_csv('./datasets/Rural_LGD_Dataset.csv')

# Separating Features and Target
X = df.drop(columns=['LGD', 'Id'])
y = df['LGD']

# Identify categorical and numerical columns
categorical_features = ['sex']
numerical_features = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
numerical_features = [col for col in numerical_features if col not in categorical_features]

# Preprocessing for numerical and categorical features
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(drop='first')

# Combine preprocessors in a column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the base models (now just RandomForest and DecisionTree)
base_models = [
    ('random_forest', RandomForestRegressor(n_estimators=100, random_state=42)),
    ('decision_tree', DecisionTreeRegressor(random_state=42))
]

# Create a Voting Regressor model
voting_model = VotingRegressor(estimators=base_models)

# Create a pipeline with preprocessing and voting regressor
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', voting_model)
])

# Fit the model
pipeline.fit(X_train, y_train)

# Predict and evaluate
y_train_pred = pipeline.predict(X_train)
y_test_pred = pipeline.predict(X_test)

# Metrics
print("Voting Model Training MSE:", mean_squared_error(y_train, y_train_pred))
print("Voting Model Training R²:", r2_score(y_train, y_train_pred))
print("Voting Model Test MSE:", mean_squared_error(y_test, y_test_pred))
print("Voting Model Test R²:", r2_score(y_test, y_test_pred))

# Save the pipeline
with open('./models/voting_lgd_pipeline.pkl', 'wb') as file:
    pickle.dump(pipeline, file)

print("Voting model pipeline saved as 'voting_lgd_pipeline.pkl'")
