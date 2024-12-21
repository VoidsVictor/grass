"""
    Generates the loan parameters and other related insights.
"""
import pandas as pd
import pickle

def load_model(path):
    with open(path, 'rb') as file:
        return pickle.load(file)
    
def predict(params_df, model):
    return model.predict(params_df)

def lti(loan_amt, annual_income, loan_tenure):
    return loan_amt / ((annual_income/12) * loan_tenure)

def lsr(loan_amt, loan_tenure, monthly_expenses):
    return (loan_amt/loan_tenure) / monthly_expenses

def specific_net_income(annual_income, monthly_expenses, loan_tenure):
    return ((annual_income/12) - monthly_expenses) * loan_tenure

def lgd(df, model):
    return model.predict(df)

if __name__ == '__main__':
    age = int(input("Enter borrower age: "))
    gender = input("Enter gender (F/M/TG): ")
    annual_income = float(input("Enter annual income: "))
    monthly_expenses = float(input("Enter monthly expenses: "))
    old_dependents = int(input("Number of old people dependent on borrower: "))
    young_dependents = int(input("Number of young people dependent on borrower: "))
    occupants_count = int(input("Number of occupants in household: "))
    house_area = float(input("Enter house area (sq ft): "))
    loan_amount = float(input("Enter loan amount: "))
    loan_tenure = int(input("Enter number loan term (months): "))

    features = pd.DataFrame({
        "age": [age],
        "sex": [gender],
        "annual_income": [annual_income],
        "monthly_expenses": [monthly_expenses],
        "old_dependents": [old_dependents],
        "young_dependents": [young_dependents],
        "occupants_count": [occupants_count],
        "house_area": [house_area],
        "loan_tenure": [loan_tenure],
        "loan_installments": [loan_tenure],
        "loan_amount": [loan_amount],
        "LTI": [lti(loan_amount, annual_income, loan_tenure)],
        "LSR": [lsr(loan_amount, loan_tenure, monthly_expenses)],
        "specific_net_income": [specific_net_income(annual_income, monthly_expenses, loan_tenure)]
    })

    model = load_model('./models/rf_lgd_pipeline.pkl')
    y_pred = lgd(features, model)
    print("LGD: ", y_pred)


