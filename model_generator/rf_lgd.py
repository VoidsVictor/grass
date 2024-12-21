import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
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
        ('model', RandomForestRegressor(random_state=42))  # Keep base model for grid search
    ])

    # Define parameter grid for GridSearchCV
    param_grid = {
        'model__n_estimators': [50, 100],
        'model__max_depth': [10],
        'model__min_samples_split': [10],
        'model__min_samples_leaf': [5],
        'model__max_features': ['sqrt']
    }

    # Use GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Perform grid search for best hyperparameters
    grid_search.fit(X_train, y_train)

    # Get the best pipeline from grid search
    best_pipeline = grid_search.best_estimator_

    # Predictions using the best pipeline
    y_train_pred = best_pipeline.predict(X_train)
    y_test_pred = best_pipeline.predict(X_test)

    # Metrics
    print("Random Forest Training MSE:", mean_squared_error(y_train, y_train_pred))
    print("Random Forest Training R²:", r2_score(y_train, y_train_pred))
    print("Random Forest Test MSE:", mean_squared_error(y_test, y_test_pred))
    print("Random Forest Test R²:", r2_score(y_test, y_test_pred))

    # Cross-validation score (to assess generalization)
    cv_scores = cross_val_score(best_pipeline, X, y, cv=5, scoring='neg_mean_squared_error')
    print("Cross-validation MSE:", -np.mean(cv_scores))

    # Save the best pipeline to a file
    with open('./models/rf_lgd_pipeline.pkl', 'wb') as file:
        pickle.dump(best_pipeline, file)

    print("Optimized model pipeline saved as 'rf_lgd_pipeline.pkl'")
