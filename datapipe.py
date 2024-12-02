import pandas as pd

# Read CSV files and drop the index column
big_purchase = pd.read_csv('Purchase Data (1).csv', index_col=0)
big_user = pd.read_csv('Users (1).csv', index_col=0)

big_purchase.reset_index(drop=True, inplace=True)
big_user.reset_index(drop=True, inplace=True)

small_purchase = pd.read_csv('route_purchase_2024-12-02_09h25m55.csv')
small_user = pd.read_csv('user_2024-12-02_09h26m35.csv')

# Append small_purchase to big_purchase
bigger_purchase = pd.concat([big_purchase, small_purchase], ignore_index=True)

# Append small_user to big_user
bigger_user = pd.concat([big_user, small_user], ignore_index=True)

# Save CSV files without index column
bigger_purchase.to_csv('New Purchase Data.csv', index=False)
bigger_user.to_csv('New Users.csv', index=False)
