"""
    Generates the loan parameters and other related insights.
"""
import pandas as pd
import numpy as np
import pickle

from scipy.stats import norm

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

def df_merton(path):
    data = pd.read_csv(path)

    # Preprocess numeric columns
    data["WITHDRAWAL_AMT"] = data["WITHDRAWAL_AMT"].fillna(0)
    data["DEPOSIT_AMT"] = data["DEPOSIT_AMT"].fillna(0)

    # Calculate net cash flow
    data["NET CASH FLOW"] = data["DEPOSIT_AMT"] - data["WITHDRAWAL_AMT"]

    # Calculate cumulative asset value (starting with an initial value, e.g., 10,000,000)
    data["ASSET VALUE"] = data["NET CASH FLOW"].cumsum()

    # Calculate daily growth rates
    data["GROWTH RATE"] = np.log(data["ASSET VALUE"] / data["ASSET VALUE"].shift(1))
    
    # Drop NaN values caused by shift
    data = data.dropna(subset=["GROWTH RATE"])

    return data

def merton_params(df):
    volatility = df["GROWTH RATE"].std()
    growth_rate = df["GROWTH RATE"].mean()

    return volatility, growth_rate

def pd_merton(volatility, growth_rate, tenure, net_worth, debt_point):
    # Tenure should be in age
    d2 = (np.log(net_worth / debt_point) + (growth_rate - (volatility**2) / 2) * tenure) / (volatility * np.sqrt(tenure))
    return norm.cdf(-d2)

def risk_premium(pd, lgd):
    return pd*lgd

def credit_score(pd, lgd, ead):
    beta = (300 - 850)/ead
    ecl = pd * lgd * ead
    return 850 + (beta * ecl)

def annual_interest_rate(risk_premium, profit_rate):
    return risk_premium + profit_rate

if __name__ == '__main__':
    # age = int(input("Enter borrower age: "))
    # gender = input("Enter gender (F/M/TG): ")
    # asset_val = float(input("Enter net worth: "))
    # annual_income = float(input("Enter annual income: "))
    # monthly_expenses = float(input("Enter monthly expenses: "))
    # old_dependents = int(input("Number of old people dependent on borrower: "))
    # young_dependents = int(input("Number of young people dependent on borrower: "))
    # occupants_count = int(input("Number of occupants in household: "))
    # house_area = float(input("Enter house area (sq ft): "))
    # loan_amount = float(input("Enter loan amount: "))
    # loan_tenure = int(input("Enter number loan term (months): "))

    # features = pd.DataFrame({
    #     "age": [age],
    #     "sex": [gender],
    #     "annual_income": [annual_income],
    #     "monthly_expenses": [monthly_expenses],
    #     "old_dependents": [old_dependents],
    #     "young_dependents": [young_dependents],
    #     "occupants_count": [occupants_count],
    #     "house_area": [house_area],
    #     "loan_tenure": [loan_tenure],
    #     "loan_installments": [loan_tenure],
    #     "loan_amount": [loan_amount],
    #     "LTI": [lti(loan_amount, annual_income, loan_tenure)],
    #     "LSR": [lsr(loan_amount, loan_tenure, monthly_expenses)],
    #     "specific_net_income": [specific_net_income(annual_income, monthly_expenses, loan_tenure)]
    # })

    # model = load_model('./models/rf_lgd_pipeline.pkl')
    # y_pred = lgd(features, model)
    # print("LGD: ", y_pred)

    lgd = 0.2
    ead = 100000
    df = df_merton("./datasets/bank.csv")
    volatility, growth_rate = merton_params(df)
    pd = pd_merton(volatility, growth_rate, 1, 10000, 800)
    rp = risk_premium(pd, lgd)
    cs = credit_score(pd, lgd, ead)
    ir = annual_interest_rate(rp, 0.03)

    print(volatility, growth_rate, pd, rp, cs, ir)