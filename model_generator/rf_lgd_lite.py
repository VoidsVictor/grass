import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

if __name__ == "__main__":
    # Load dataset
    df = pd.read_csv('./datasets/Rural_LGD_Dataset.csv')

    # Separating Features and Target
    X = df.drop(columns=['LGD', 'Id'])
    y = df['LGD']

    # Identify categorical and numerical columns
    categorical_features = ['sex']  # Add more categorical features if present
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

    # Create a pipeline that includes preprocessing and model training
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the pipeline
    pipeline.fit(X_train, y_train)

    # Predictions
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)

    # Metrics
    print("Random Forest Training MSE:", mean_squared_error(y_train, y_train_pred))
    print("Random Forest Training R²:", r2_score(y_train, y_train_pred))
    print("Random Forest Test MSE:", mean_squared_error(y_test, y_test_pred))
    print("Random Forest Test R²:", r2_score(y_test, y_test_pred))

    # Save the pipeline to a file
    with open('./models/rf_lgd_lite_pipeline.pkl', 'wb') as file:
        pickle.dump(pipeline, file)

    print("Model pipeline saved as 'rf_lgd_lite_pipeline.pkl'")
