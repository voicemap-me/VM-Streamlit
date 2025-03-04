import pandas as pd

# Read CSV files
print("Reading big data")
big_purchase = pd.read_csv('New Purchase Data.csv')
big_user = pd.read_csv('New Users.csv')

print("Reading new months data")
small_purchase = pd.read_csv('route_purchase_feb25.csv')
small_user = pd.read_csv('user_feb25.csv')

# Get common columns between purchase DataFrames
common_columns = list(set(big_purchase.columns) & set(small_purchase.columns))

# Convert datetime columns with mixed formats and handle time zones
big_purchase['Created at [Route Purchase]'] = pd.to_datetime(big_purchase['Created at [Route Purchase]'], format='mixed', utc=True)
small_purchase['Created at [Route Purchase]'] = pd.to_datetime(small_purchase['Created at [Route Purchase]'], format='mixed', utc=True)

# Convert user datetime columns
big_user['Created at'] = pd.to_datetime(big_user['Created at'], format='mixed', utc=True)
small_user['Created at'] = pd.to_datetime(small_user['Created at'], format='mixed', utc=True)

print("Grouping..")
# Keep only common columns and concatenate purchase data
bigger_purchase = pd.concat([
    big_purchase[common_columns], 
    small_purchase[common_columns]
], ignore_index=True)

# Append small_user to big_user (keeping only Id and Created at)
bigger_user = pd.concat([
    big_user[['Id', 'Created at']], 
    small_user[['Id', 'Created at']]
], ignore_index=True)

print("Saving files New Purchase Data.csv and New Users.csv")
# Save CSV files without index column
bigger_purchase.to_csv('Purchase Data 4 March 2025.csv', index=False)
bigger_user.to_csv('User Data 4 March 2025.csv', index=False)
