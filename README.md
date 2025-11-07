# Grass - Creditworthiness Evaluation System

> **Available for Consultation:** As the developer of this system, I'm available for freelance projects implementing similar financial technology solutions for businesses. [Contact me](#contact) to discuss your needs.

## Project Overview

Grass is an advanced financial analytics tool that addresses one of the most significant challenges in global finance: **evaluating creditworthiness for the underbanked and those without formal credit histories**. Using sophisticated statistical models and machine learning techniques, this system provides holistic credit evaluation that goes beyond traditional credit scores.

## Key Capabilities

- **Alternative Credit Scoring**: Evaluates creditworthiness without requiring traditional credit history
- **Risk Assessment**: Implements the Black-Scholes-Merton model to calculate Probability of Default (PD)
- **Loss Prediction**: Advanced algorithms to predict Loss Given Default (LGD)
- **Loan Optimisation**: AI-driven optimisation of loan structures and installment plans
- **Customisable Parameters**: Adaptable to various demographic and regional contexts
- **Research-Backed**: Built on methodologies documented in published research

## Technical Implementation

The system currently leverages:
- **Statistical Modeling**: Implementation of the Black-Scholes-Merton model
- **Machine Learning**: Random forest regression models with scikit-learn
- **Data Pipeline Architecture**: End-to-end processing from raw financial data to credit decisions
- **Planned PyTorch Integration**: Transitioning to deep learning for enhanced prediction accuracy

## Real-World Applications

- **Microfinance Institutions**: Enable lending to clients without formal credit histories
- **Rural Banking**: Expand financial inclusion in developing regions
- **SME Lending**: Evaluate small business creditworthiness with limited financial records
- **Alternative Lending Platforms**: Power new credit products for underserved markets

## Installation and Setup

1. **Clone the Repository:**
   ```sh
   git clone https://github.com/YourUsername/grass.git
   cd grass
   ```

2. **Install Requirements:**
   ```sh
   pip install -r requirements.txt
   ```

3. **Generate the Model:**
   ```sh
   mkdir models
   python model_generator/rf_lgd.py
   ```

## Implementation Example

```python
from grass import calculate_loan_metrics

# Example borrower with limited credit history
borrower = {
    "age": 30,
    "gender": "M",
    "annual_income": 50000,
    "monthly_expenses": 2000,
    "net_worth": 100000,
    "old_dependents": 1,
    "young_dependents": 2,
    "occupants_count": 3,
    "house_area": 1200
}

loan = {
    "loan_amount": 20000,
    "loan_tenure": 60
}

# Calculate comprehensive credit metrics
metrics = calculate_loan_metrics(borrower, loan, "./models/rf_lgd_pipeline.pkl")
print(metrics)
```

## Research Foundation

This project applies methodologies explored in published research on alternative credit scoring. The implementation leverages stochastic processes to model financial risk in ways that traditional credit systems cannot.

For more technical details on the models:
- `notebooks/lgd_EDA.ipynb`: Loss Given Default analysis methodology
- `notebooks/merton_params.ipynb`: Black-Scholes-Merton Model implementation for PD estimation

## Customisation Options

The system can be customised for specific:
- Regional markets
- Demographic segments
- Financial product types
- Risk tolerance levels
- Regulatory requirements

## License

This project is licensed under the GNU General Public License v3.0. See the [LICENSE](./LICENSE) file for details.
