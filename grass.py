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

def monthly_installment(loan_amt, rate, tenure_months):
    """
    Calculates monthly installment for given loan parameters.
    """
    if rate == 0:
        return loan_amt / tenure_months
    monthly_rate = rate / 12
    return loan_amt * (monthly_rate * (1 + monthly_rate) ** tenure_months) / (
        (1 + monthly_rate) ** tenure_months - 1
    )

def optimize_loan_tenure(
    loan_amt,
    annual_income,
    monthly_expenses,
    profit_rate,
    net_worth,
    debt_point,
    lgd,
    ead,
    max_tenure_years=30,
):
    """
    Optimizes the loan tenure to minimize monthly installments, considering detailed installment calculations.

    Args:
        loan_amt (float): Loan amount.
        annual_income (float): Annual income of the borrower.
        monthly_expenses (float): Monthly expenses of the borrower.
        profit_rate (float): Desired profit rate for the loan.
        net_worth (float): Net worth of the borrower.
        debt_point (float): Debt threshold for calculating Merton probability.
        lgd (float): Loss given default.
        ead (float): Exposure at default.
        max_tenure_years (int): Maximum loan tenure in years (default is 30).

    Returns:
        dict: Optimized tenure, monthly installment, and associated metrics.
    """
    from scipy.optimize import minimize

    # Define constraints for tenure
    def constraints(tenure_months):
        # Example constraints: LTI and monthly expenses coverage
        lti_value = lti(loan_amt, annual_income, tenure_months / 12)
        min_lti, max_lti = 0.2, 0.5  # Example range
        return [
            tenure_months - 12,  # Minimum 1 year tenure
            max_tenure_years * 12 - tenure_months,  # Maximum tenure
            max_lti - lti_value,  # Ensure LTI is within range
            lti_value - min_lti,  # Ensure LTI is within range
        ]

    # Define the objective function (minimize monthly installment)
    def objective_function(tenure_months):
        rate = annual_interest_rate(
            risk_premium(pd_merton(0.2, 0.03, tenure_months / 12, net_worth, debt_point), lgd),
            profit_rate,
        )
        return monthly_installment(loan_amt, rate, tenure_months)

    # Optimize tenure
    initial_tenure_months = max_tenure_years * 12 / 2  # Initial guess: halfway tenure
    bounds = [(12, max_tenure_years * 12)]  # Between 1 and max years in months

    result = minimize(
        lambda x: objective_function(x[0]),
        x0=[initial_tenure_months],
        bounds=bounds,
        constraints=[
            {"type": "ineq", "fun": lambda x: c} for c in constraints(initial_tenure_months)
        ],
        method="SLSQP",
    )

    # Extract optimized tenure and recalculate metrics
    optimized_tenure_months = result.x[0]
    rate = annual_interest_rate(
        risk_premium(pd_merton(0.2, 0.03, optimized_tenure_months / 12, net_worth, debt_point), lgd),
        profit_rate,
    )
    installment = monthly_installment(loan_amt, rate, optimized_tenure_months)

    return {
        "optimized_tenure_years": optimized_tenure_months / 12,
        "monthly_installment": installment,
        "interest_rate": rate,
        "loan_to_income_ratio": lti(loan_amt, annual_income, optimized_tenure_months / 12),
    }

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

    # lgd = 0.2
    # ead = 100000
    # df = df_merton("./datasets/bank.csv")
    # volatility, growth_rate = merton_params(df)
    # pd = pd_merton(volatility, growth_rate, 1, 10000, 800)
    # rp = risk_premium(pd, lgd)
    # cs = credit_score(pd, lgd, ead)
    # ir = annual_interest_rate(rp, 0.03)

    # print(volatility, growth_rate, pd, rp, cs, ir)

    result = optimize_loan_tenure(
        loan_amt=100000,
        annual_income=600000,
        monthly_expenses=30000,
        profit_rate=0.03,
        net_worth=1000000,
        debt_point=800000,
        lgd=0.2,
        ead=100000,
        max_tenure_years=1
    )

    print(result)