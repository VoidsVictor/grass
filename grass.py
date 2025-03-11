"""
Grass: Loan Risk Analysis and Optimization Module

This module generates loan parameters and related insights using financial models,
machine learning predictions, and optimization techniques. Key calculations include:
- Loan-to-Income Ratio (LTI)
- Loan Servicing Ratio (LSR)
- Specific Net Income over the loan tenure
- Loss Given Default (LGD) prediction via a pre-trained model
- Merton Model parameters (volatility and growth rate) for calculating probability of default (PD)
- Risk premium, credit scoring, and interest rate calculations
- Optimization of loan tenure to minimize monthly installments

Note: Some functions assume specific units for input (e.g., annual income, loan tenure in years or months).
Ensure consistency when integrating into your system.
"""

import pandas as pd
import numpy as np
import pickle
from scipy.stats import norm

def load_model(path):
    """
    Load a pickled model from the specified file path.
    
    Args:
        path (str): Path to the pickle file containing the model.
    
    Returns:
        model: Loaded machine learning model.
    """
    with open(path, 'rb') as file:
        return pickle.load(file)
    
def predict(params_df, model):
    """
    Predict output using the provided model on the given parameters DataFrame.
    
    Args:
        params_df (pd.DataFrame): DataFrame containing feature values.
        model: Trained machine learning model.
    
    Returns:
        array: Prediction results from the model.
    """
    return model.predict(params_df)

def lti(loan_amt, annual_income, loan_tenure):
    """
    Calculate the Loan-to-Income Ratio (LTI).
    
    LTI is computed as:
        LTI = loan_amt / (monthly_income * loan_tenure)
    where monthly_income is derived as annual_income/12.
    
    Args:
        loan_amt (float): Loan amount.
        annual_income (float): Annual income of the borrower.
        loan_tenure (float): Loan tenure (expressed in years).
    
    Returns:
        float: Calculated LTI.
    """
    return loan_amt / ((annual_income/12) * loan_tenure)

def lsr(loan_amt, loan_tenure, monthly_expenses):
    """
    Calculate the Loan Servicing Ratio (LSR).
    
    LSR is computed as:
        LSR = (loan_amt / loan_tenure) / monthly_expenses
    
    Args:
        loan_amt (float): Loan amount.
        loan_tenure (float): Loan tenure (expressed consistently with loan_amt units).
        monthly_expenses (float): Monthly expenses of the borrower.
    
    Returns:
        float: Calculated LSR.
    """
    return (loan_amt/loan_tenure) / monthly_expenses

def specific_net_income(annual_income, monthly_expenses, loan_tenure):
    """
    Calculate the specific net income over the loan tenure.
    
    This is computed as:
        Specific Net Income = (monthly_income - monthly_expenses) * loan_tenure,
    where monthly_income is annual_income/12.
    
    Args:
        annual_income (float): Annual income of the borrower.
        monthly_expenses (float): Monthly expenses of the borrower.
        loan_tenure (float): Loan tenure (expressed in years).
    
    Returns:
        float: Specific net income over the tenure.
    """
    return ((annual_income/12) - monthly_expenses) * loan_tenure

def lgd(df, model):
    """
    Predict the Loss Given Default (LGD) using a pre-trained model.
    
    Args:
        df (pd.DataFrame): DataFrame containing features.
        model: Pre-trained model for predicting LGD.
    
    Returns:
        array: Predicted LGD values.
    """
    return model.predict(df)

def df_merton(path):
    """
    Load and preprocess data required for Merton's model calculations.
    
    The function performs the following steps:
      - Reads CSV data from the specified path.
      - Fills missing values in 'WITHDRAWAL_AMT' and 'DEPOSIT_AMT' with 0.
      - Computes net cash flow as DEPOSIT_AMT - WITHDRAWAL_AMT.
      - Calculates cumulative asset value (ASSET VALUE) as the cumulative sum of net cash flow.
      - Computes daily growth rates (GROWTH RATE) as the log ratio of consecutive asset values.
      - Drops rows with NaN growth rates.
    
    Args:
        path (str): Path to the CSV dataset.
    
    Returns:
        pd.DataFrame: Preprocessed DataFrame with asset growth information.
    """
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
    """
    Calculate the volatility and average growth rate from the asset's daily growth rates.
    
    These parameters are essential for Merton's model:
      - Volatility: Standard deviation of GROWTH RATE.
      - Growth Rate: Mean of GROWTH RATE.
    
    Args:
        df (pd.DataFrame): DataFrame containing the 'GROWTH RATE' column.
    
    Returns:
        tuple: (volatility (float), growth_rate (float))
    """
    volatility = df["GROWTH RATE"].std()
    growth_rate = df["GROWTH RATE"].mean()
    return volatility, growth_rate

def pd_merton(volatility, growth_rate, tenure, net_worth, debt_point):
    """
    Calculate the Probability of Default (PD) using Merton's model.
    
    The model computes d2 as:
        d2 = (ln(net_worth / debt_point) + (growth_rate - 0.5 * volatility^2) * tenure) / (volatility * sqrt(tenure))
    and returns PD as norm.cdf(-d2).
    
    Args:
        volatility (float): Asset volatility.
        growth_rate (float): Average asset growth rate.
        tenure (float): Time horizon (in years).
        net_worth (float): Net worth of the borrower.
        debt_point (float): Debt threshold for default.
    
    Returns:
        float: Probability of default.
    """
    d2 = (np.log(net_worth / debt_point) + (growth_rate - (volatility**2) / 2) * tenure) / (volatility * np.sqrt(tenure))
    return norm.cdf(-d2)

def risk_premium(pd_value, lgd_value):
    """
    Calculate the risk premium as the product of the probability of default and loss given default.
    
    Args:
        pd_value (float): Probability of default.
        lgd_value (float): Loss Given Default.
    
    Returns:
        float: Risk premium.
    """
    return pd_value * lgd_value

def credit_score(pd_value, lgd_value, ead):
    """
    Calculate a credit score based on PD, LGD, and Exposure at Default (EAD).
    
    The formula used here is illustrative:
        beta = (300 - 850) / ead
        Expected Credit Loss (ECL) = pd_value * lgd_value * ead
        Credit Score = 850 + (beta * ECL)
    
    Args:
        pd_value (float): Probability of default.
        lgd_value (float): Loss Given Default.
        ead (float): Exposure at default.
    
    Returns:
        float: Calculated credit score.
    """
    beta = (300 - 850) / ead
    ecl = pd_value * lgd_value * ead
    return 850 + (beta * ecl)

def annual_interest_rate(risk_prem, profit_rate):
    """
    Calculate the annual interest rate based on the risk premium and a desired profit rate.
    
    Args:
        risk_prem (float): Calculated risk premium.
        profit_rate (float): Base profit rate desired.
    
    Returns:
        float: Annual interest rate.
    """
    return risk_prem + profit_rate

def monthly_installment(loan_amt, rate, tenure_months):
    """
    Calculate the monthly installment for a loan using the annuity formula.
    
    If the interest rate is zero, a simple division of loan_amt by tenure_months is performed.
    
    Args:
        loan_amt (float): Total loan amount.
        rate (float): Annual interest rate (as a decimal).
        tenure_months (int): Loan tenure in months.
    
    Returns:
        float: Calculated monthly installment.
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
    profit_rate,
    net_worth,
    debt_point,
    lgd,
    max_tenure_years=30,
):
    """
    Optimize the loan tenure to minimize monthly installments while considering constraints on the
    Loan-to-Income (LTI) ratio.
    
    The function uses a numerical optimization (SLSQP) to find the tenure (in months) that minimizes
    the monthly installment. Constraints ensure that the LTI remains within a specified range.
    
    Args:
        loan_amt (float): Loan amount.
        annual_income (float): Annual income of the borrower.
        profit_rate (float): Desired profit rate for the loan.
        net_worth (float): Net worth of the borrower.
        debt_point (float): Debt threshold used for calculating PD via Merton's model.
        lgd (float): Loss Given Default.
        max_tenure_years (int, optional): Maximum loan tenure in years (default is 30).
    
    Returns:
        dict: A dictionary containing:
            - optimized_tenure_years (float): Optimized loan tenure in years.
            - monthly_installment (float): Corresponding monthly installment.
            - interest_rate (float): Annual interest rate used.
            - loan_to_income_ratio (float): LTI at optimized tenure.
    """
    from scipy.optimize import minimize

    def constraints(tenure_months):
        """
        Define inequality constraints based on the LTI value.
        Ensures tenure is at least 1 year and within the maximum allowed, and LTI remains within [min_lti, max_lti].
        """
        # Convert tenure in months to years for LTI calculation
        lti_value = lti(loan_amt, annual_income, tenure_months / 12)
        min_lti, max_lti = 0.2, 0.5  # Example acceptable range for LTI
        return [
            tenure_months - 12,               # Minimum tenure: 12 months (1 year)
            max_tenure_years * 12 - tenure_months,  # Maximum tenure in months
            max_lti - lti_value,              # Upper bound on LTI
            lti_value - min_lti,              # Lower bound on LTI
        ]

    def objective_function(tenure_months):
        """
        Objective function that returns the monthly installment for a given tenure.
        """
        rate = annual_interest_rate(
            risk_premium(pd_merton(0.2, 0.03, tenure_months / 12, net_worth, debt_point), lgd),
            profit_rate,
        )
        return monthly_installment(loan_amt, rate, tenure_months)

    # Initial guess: halfway tenure in months
    initial_tenure_months = max_tenure_years * 12 / 2
    bounds = [(12, max_tenure_years * 12)]  # Tenure between 1 year and max_tenure_years

    # Optimize using SLSQP method
    result = minimize(
        lambda x: objective_function(x[0]),
        x0=[initial_tenure_months],
        bounds=bounds,
        constraints=[
            {"type": "ineq", "fun": lambda x: c} for c in constraints(initial_tenure_months)
        ],
        method="SLSQP",
    )

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
    # Example usage of optimize_loan_tenure function.
    # (Input values are illustrative. In practice, obtain these from user inputs or data.)
    result = optimize_loan_tenure(
        loan_amt=100000,
        annual_income=600000,
        profit_rate=0.03,
        net_worth=1000000,
        debt_point=800000,
        lgd=0.2,
        max_tenure_years=1
    )

    print(result)
