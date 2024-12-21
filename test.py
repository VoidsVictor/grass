import pandas as pd
import pickle

# Load the pipeline
with open('./models/rf_lgd_pipeline.pkl', 'rb') as file:
    pipeline = pickle.load(file)

data_high_lgd = pd.DataFrame({
    'age': [35],
    'sex': ['M'],
    'annual_income': [15000.0],  # Very low annual income
    'monthly_expenses': [4000.0],  # High monthly expenses relative to income
    'old_dependents': [2],
    'young_dependents': [3],
    'occupants_count': [6],
    'house_area': [50.0],  # Small house area
    'loan_tenure': [36],  # Long loan tenure
    'loan_installments': [36],  # Spread over a long period
    'loan_amount': [25000.0],  # High loan amount relative to income
    'LTI': [20],  # Extremely high Loan-to-Income ratio
    'LSR': [0.8000],  # High Loan-to-Size ratio
    'specific_net_income': [-33000.0],  # Negative specific net income
    'LGD': [0.95]  # Very high LGD
})

to_test = pd.DataFrame(data = {
    'age': [40],
    'sex': ['M'],
    'annual_income': [700000.0],
    'monthly_expenses': [5000.0],
    'old_dependents': [0],
    'young_dependents': [2],
    'occupants_count': [4],
    'house_area': [1000.0],
    'loan_tenure': [12],
    'loan_installments': [12],
    'loan_amount': [10.0],
    'LTI': [1.66],
    'LSR': [0.82],
    'specific_net_income': [100000.0],
})

# Predict LGD using the loaded model
predictions = pipeline.predict(data_high_lgd)
print("Predicted LGD:", predictions)
