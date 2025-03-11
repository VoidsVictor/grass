import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from grass import *

def get_borrower_info() -> Dict[str, Any]:
    """Get basic borrower information from user input."""
    print("\n=== Borrower Information ===")
    return {
        "age": int(input("Enter borrower age: ")),
        "gender": input("Enter gender (F/M/TG): "),
        "annual_income": float(input("Enter annual income: ")),
        "monthly_expenses": float(input("Enter monthly expenses: ")),
        "old_dependents": int(input("Number of old people dependent on borrower: ")),
        "young_dependents": int(input("Number of young people dependent on borrower: ")),
        "occupants_count": int(input("Number of occupants in household: ")),
        "house_area": float(input("Enter house area (sq ft): ")),
        "net_worth": float(input("Enter total net worth: "))
    }

def get_loan_request() -> Dict[str, Any]:
    """Get loan request details from user input."""
    print("\n=== Loan Request Details ===")
    return {
        "loan_amount": float(input("Enter desired loan amount: ")),
        "loan_tenure": int(input("Enter desired loan tenure (months): "))
    }

def calculate_loan_metrics(borrower: Dict[str, Any], loan: Dict[str, Any], model_path: str) -> Dict[str, Any]:
    """Calculate various loan metrics based on borrower and loan information."""
    # Create features DataFrame for model prediction
    features = pd.DataFrame({
        "age": [borrower["age"]],
        "sex": [borrower["gender"]],
        "annual_income": [borrower["annual_income"]],
        "monthly_expenses": [borrower["monthly_expenses"]],
        "old_dependents": [borrower["old_dependents"]],
        "young_dependents": [borrower["young_dependents"]],
        "occupants_count": [borrower["occupants_count"]],
        "house_area": [borrower["house_area"]],
        "loan_tenure": [loan["loan_tenure"]],
        "loan_installments": [loan["loan_tenure"]],
        "loan_amount": [loan["loan_amount"]],
        "LTI": [lti(loan["loan_amount"], borrower["annual_income"], loan["loan_tenure"])],
        "LSR": [lsr(loan["loan_amount"], loan["loan_tenure"], borrower["monthly_expenses"])],
        "specific_net_income": [specific_net_income(borrower["annual_income"], 
                                                  borrower["monthly_expenses"], 
                                                  loan["loan_tenure"])]
    })
    
    # Load model and predict LGD
    model = load_model(model_path)
    predicted_lgd = float(lgd(features, model))
    
    # Calculate Merton model parameters
    df = df_merton("./datasets/bank.csv")
    volatility, growth_rate = merton_params(df)
    
    # Calculate risk metrics
    prob_default = pd_merton(volatility, growth_rate, loan["loan_tenure"]/12, 
                            borrower["net_worth"], loan["loan_amount"])
    risk_prem = risk_premium(prob_default, predicted_lgd)
    credit_score_val = credit_score(prob_default, predicted_lgd, loan["loan_amount"])
    interest = annual_interest_rate(risk_prem, 0.03)  # Using 3% as base profit rate
    monthly_payment = monthly_installment(loan["loan_amount"], interest, loan["loan_tenure"])
    
    return {
        "lgd": predicted_lgd,
        "probability_of_default": prob_default,
        "risk_premium": risk_prem,
        "credit_score": credit_score_val,
        "interest_rate": interest,
        "monthly_payment": monthly_payment
    }

def display_loan_analysis(metrics: Dict[str, Any]):
    """Display loan analysis results in a formatted manner."""
    print("\n=== Loan Analysis Results ===")
    print(f"Credit Score: {metrics['credit_score']:.2f}")
    print(f"Annual Interest Rate: {metrics['interest_rate']*100:.2f}%")
    print(f"Monthly Payment: {metrics['monthly_payment']:.2f}")
    print(f"Risk Metrics:")
    print(f"  - Loss Given Default: {metrics['lgd']*100:.2f}%")
    print(f"  - Probability of Default: {metrics['probability_of_default']*100:.2f}%")
    print(f"  - Risk Premium: {metrics['risk_premium']*100:.2f}%")

def optimize_existing_loan(borrower: Dict[str, Any], current_loan: Dict[str, Any]):
    """Optimize an existing loan terms."""
    print("\n=== Loan Optimization Analysis ===")
    
    result = optimize_loan_tenure(
        loan_amt=current_loan["loan_amount"],
        annual_income=borrower["annual_income"],
        monthly_expenses=borrower["monthly_expenses"],
        profit_rate=0.03,  # Base profit rate
        net_worth=borrower["net_worth"],
        debt_point=current_loan["loan_amount"],
        lgd=0.2,  # Using a conservative estimate
        ead=current_loan["loan_amount"],
        max_tenure_years=30
    )
    
    print("\nOptimized Loan Terms:")
    print(f"Recommended Tenure: {result['optimized_tenure_years']:.1f} years")
    print(f"Monthly Installment: {result['monthly_installment']:.2f}")
    print(f"Interest Rate: {result['interest_rate']*100:.2f}%")
    print(f"Loan-to-Income Ratio: {result['loan_to_income_ratio']:.2f}")

def main():
    while True:
        print("\n=== Loan Analysis System ===")
        print("1. New Loan Analysis")
        print("2. Optimize Existing Loan")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ")
        
        if choice == "1":
            try:
                # Get borrower and loan information
                borrower = get_borrower_info()
                loan = get_loan_request()
                
                # Calculate and display metrics
                metrics = calculate_loan_metrics(borrower, loan, './models/rf_lgd_pipeline.pkl')
                display_loan_analysis(metrics)
                
                # Ask if user wants to optimize
                if input("\nWould you like to see optimized loan terms? (y/n): ").lower() == 'y':
                    optimize_existing_loan(borrower, loan)
                    
            except Exception as e:
                print(f"\nError occurred: {str(e)}")
                print("Please try again with valid inputs.")
                
        elif choice == "2":
            try:
                print("\n=== Existing Loan Optimization ===")
                borrower = get_borrower_info()
                current_loan = get_loan_request()
                optimize_existing_loan(borrower, current_loan)
                
            except Exception as e:
                print(f"\nError occurred: {str(e)}")
                print("Please try again with valid inputs.")
                
        elif choice == "3":
            print("\nThank you for using the Loan Analysis System!")
            break
            
        else:
            print("\nInvalid choice. Please select 1, 2, or 3.")

if __name__ == "__main__":
    main()