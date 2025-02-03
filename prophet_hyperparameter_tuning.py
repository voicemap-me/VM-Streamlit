import itertools
import numpy as np
import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics



##########################################
# 1) LOAD YOUR MONTHLY DATA (ds, y)
##########################################
# 1) Import your utility functions from the separate file
from utils import (
    build_monthly_revenue_df, 
    build_monthly_users_df
)
###############################################################################
# 2) Build monthly data for Revenue / Users
###############################################################################
monthly_revenue = build_monthly_revenue_df()  # => DataFrame with ds, y
monthly_users   = build_monthly_users_df()    # => DataFrame with ds, y

# (Optional) Print tails to confirm
print("Revenue Data (tail):\n", monthly_revenue.tail())
print("\nUsers Data (tail):\n", monthly_users.tail())


##########################################
# 2) DEFINE THE PARAMETER GRID
##########################################
param_grid = {
    'changepoint_prior_scale': [0.05, 0.1, 0.3],
    'seasonality_prior_scale': [5, 10],
    'seasonality_mode': ['additive', 'multiplicative']
}
# Feel free to add 'holidays_prior_scale': [5,10] or other keys if relevant.

all_params = [
    dict(zip(param_grid.keys(), v))
    for v in itertools.product(*param_grid.values())
]

##########################################
# 3) FUNCTION TO RUN CROSS-VALIDATION
##########################################
def tune_prophet_hyperparams(df, horizon='90 days', period='90 days', initial='540 days'):
    """
    Runs a grid search over all_params on the given dataframe df (['ds','y']).
    Returns a DataFrame of results with columns:
      [param_combo, changepoint_prior_scale, seasonality_prior_scale,
       seasonality_mode, mape, mae, rmse, coverage]
    Sorted by MAPE ascending.
    """
    results = []

    # Loop over each combo of hyperparams
    for params in all_params:
        # 1) Build the model with these params
        m = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            **params  # Unpacks our param grid combos
        )
        m.fit(df)

        # 2) Perform cross-validation
        # horizon: how far out we forecast per fold
        # period: spacing between cutoffs
        # initial: how much data in the first training set
        df_cv = cross_validation(
            model=m,
            horizon=horizon,
            period=period,
            initial=initial
        )

        # 3) Compute performance metrics
        df_metrics = performance_metrics(df_cv, rolling_window=1)
        mape  = df_metrics['mape'].mean()
        mae   = df_metrics['mae'].mean()
        rmse  = df_metrics['rmse'].mean()
        cov   = df_metrics['coverage'].mean()

        # 4) Store results
        results.append({
            'param_combo': params,
            'changepoint_prior_scale': params['changepoint_prior_scale'],
            'seasonality_prior_scale': params['seasonality_prior_scale'],
            'seasonality_mode': params['seasonality_mode'],
            'mape': mape,
            'mae': mae,
            'rmse': rmse,
            'coverage': cov
        })

    # Convert to DataFrame, sort by MAPE ascending
    results_df = pd.DataFrame(results).sort_values(by='mape', ascending=True)
    return results_df


##########################################
# 4) RUN TUNING FOR NEW USERS
##########################################
print("=== TUNING FOR NEW USERS ===")
results_users = tune_prophet_hyperparams(
    df=monthly_users,
    horizon='180 days',    # e.g. 6 months forecast horizon
    period='180 days',     # step between each fold
    initial='720 days'     # about 2 years of initial training
)
print("Top results (Users):")
print(results_users.head(5))  # show best combos

best_users_params = results_users.iloc[0]['param_combo']
print("\nBest param combo for New Users:", best_users_params)
print("MAPE:", results_users.iloc[0]['mape'])

##########################################
# 5) RUN TUNING FOR REVENUE
##########################################
print("\n=== TUNING FOR REVENUE ===")
results_revenue = tune_prophet_hyperparams(
    df=monthly_revenue,
    horizon='180 days',
    period='180 days',
    initial='720 days'
)
print("Top results (Revenue):")
print(results_revenue.head(5))

best_revenue_params = results_revenue.iloc[0]['param_combo']
print("\nBest param combo for Revenue:", best_revenue_params)
print("MAPE:", results_revenue.iloc[0]['mape'])
