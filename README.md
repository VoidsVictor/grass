# Plan of Action

## Calculating the Loss Given Default (LGD)

- From the Dataset,
	- Normalise and standardise the dataset
	- Split the data into training (with validation set as well) and testing
	- Calculate Loan To Income (LTI) = (Loan Amt)/(Annual Income)
	- Calculate Loan Servicing Ration (LSR) = (Montly Installment)/(Monthly Expense)
	- Heuristically calculate proxy_LGD:
		- Calculate Net Annual Income = Income * (12*monthly_expence)
		- Calculate Total Payable for Loan = Loan_installments * Loan_term
		- Calculate LGD = (Total_loan - Net_income)/Net_income
	- Calculate correlations and other statistics
	- Drop irrelevant tables
- Find and train model with optimised parameters
- Save the model
- Write the Python program that uses this model to predict LGD and calculates Adjusted LGD by equating Collateral LGD (CLGD) and Estimated LGD (ELGD)
	- Adjusted LGD = a * CLGD + (1 - a) * ELGD
	- a = (collateral_value)/(loan_amt)

## Calculating the Probability of Default (PD)

NOTE: Dataset need to be changed containing only concise information in order to derive the the mu and sigma as the available dataset is leaning towards statistical estimation.

- From the dataset,
	- Separate the dataset based on account no
	- Keep date, withdrawal_amt and deposit_amt
	- Group data into daily and calculate daily withdrawals and daily deposits
	- Calculate column Daily Net Change = Deposit - Withdrawal
	- Calculate Rolling Standard Deviation of the Daily Net Change (volatility) in percentage (percentiles)
	- Calculate the following columns
		- Avg_Daily_Withdrawal
		- Avg_Daily_Deposits
		- Withdrawal_V_Deposit_Ratio
		- Estimated_balance = data['Net Change'].cumsum()
		- Daily_Growth_Rate = 
			```
				data['Previous Balance'] = data['Estimated Balance'].shift(1)
				data['Growth Rate'] = data['Net Change'] / data['Previous Balance']
			```
	- Split the data into training (with validation set as well) and testing
	- Normalise and stardise the dataset
	- Calculate correlations and other statistics
	- Drop irrelevant tables

### Estimating Volatility

- Find and train model with optimised parameters
- Save the model

### Estimating Growth Rate
- Find and train model with optimised parameters
- Save the model

### Merton Model

- Put the parameters in the Merton Model
- Get the Probability of Default

## Calculate the Credit Score

- Calculate PD and LGD with 100k as EAD
- Calculate EL = PD * LGD * EAD
- 
