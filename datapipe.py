import pandas as pd

big_purchase = pd.read_csv('Purchase Data.csv')
big_user = pd.read_csv('Users.csv')

small_purchase = pd.read_csv('route_purchase_2024-11-25_17h38m08.csv')
small_user = pd.read_csv('user_2024-11-25_17h38m50.csv')

# Append small_purchase to big_purchase
bigger_purchase = pd.concat([big_purchase, small_purchase], ignore_index=True)

# Append small_user to big_user
bigger_user = pd.concat([big_user, small_user], ignore_index=True)

bigger_purchase.to_csv('New Purchase Data.csv')
bigger_user.to_csv('New User Data.csv')