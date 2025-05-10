# Grass - Holistic Creditworthiness Evaluation System

> **Note:** This is an ongoing project. In its current stage, it is highly experimental and unstable. It should not be used in production environments. Open sourced components used in this project are governed by their respective licenses.

## Problem at Hand

A significant portion of the world's population lacks a formal credit history, despite being active participants in the economy. Examples include street vendors and farmers in developing nations. The critical challenge is: **How do we assess their creditworthiness?** This project aims to address that question.

### Key Areas Addressed by this Project

- Recognizing that creditworthiness evaluation is a holistic process—a single credit score does not define a person.
- Assessing creditworthiness under the assumption that no formal credit history exists.
- Incorporating stochastic evaluation methods for creditworthiness.
- Researching and understanding innovative approaches to credit evaluation.

## Present Overview

Grass is a financial analytics tool that calculates key loan parameters, assesses credit risk, and optimizes loan structures using statistical and machine learning techniques. The system implements models such as the Black-Scholes-Merton (BSM) model for estimating Probability of Default (PD), predicts Loss Given Default (LGD), computes credit scores, and optimizes loan installment plans.

For more insights on current methods for evaluating LGD and PD (which may evolve over time), please refer to the notebooks:
- `lgd_EDA.ipynb` (LGD Exploratory Data Analysis)
- `merton_params.ipynb` ("Black-Scholes-Merton Model For Estimation of Probability of Default")  
both located in the `notebooks` directory.

Currently, this project utilizes `scikit-learn` for its machine learning components, but it will soon transition to `pytorch` for more advanced modeling.

## Installation and Setup

1. **Clone the Repository:**

   ```sh
   git clone https://github.com/VoidsVictor/grass.git
   cd grass
   ```

2. **Install Requirements:**

   ```sh
   pip install -r requirements.txt
   ```

3. **Generate an Appropriate Model:**

   The random forest regressor model is preferred. Create the models directory and run the model generator:

   ```sh
   mkdir models
   py model_generator/rf_lgd.py
   ```

## Usage

### Running the Application

Execute the main script:

```bash
python main.py
```

### Sample Usage in Code

To calculate loan metrics for a borrower:

```python
from grass import calculate_loan_metrics

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

metrics = calculate_loan_metrics(borrower, loan, "./models/rf_lgd_pipeline.pkl")
print(metrics)
```

## Notes

1. Dataset is representative of rural India and may not be appropriate for other regions.
2. `datasets/bank.csv` should contain, in the same format, bank statement of the respective entity.

## License

This project is licensed under the GNU General Public License v3.0. See the [LICENSE](./LICENSE) file for details.
