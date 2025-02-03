# utils.py
import pandas as pd
from prophet import Prophet
import datetime

def load_data():
    users = pd.read_csv("New Users.csv")
    purchases = pd.read_csv("New Purchase Data.csv")
    
    # Convert dates with more robust parsing and ensure timezone-naive
    users['Created at'] = pd.to_datetime(users['Created at'], format='mixed', utc=True).dt.tz_localize(None)
    purchases['Created at [Route Purchase]'] = pd.to_datetime(purchases['Created at [Route Purchase]'], format='mixed', utc=True).dt.tz_localize(None)
    
    # Optimize datatypes
    users['Id'] = pd.to_numeric(users['Id'], errors='coerce', downcast='integer')
    purchases['Id [User]'] = pd.to_numeric(purchases['Id [User]'], errors='coerce', downcast='integer')
    purchases['Price [Route Purchase]'] = pd.to_numeric(purchases['Price [Route Purchase]'], errors='coerce', downcast='float')
    
    return users, purchases


def process_filtered_data(
    payment_types_key: tuple,
    start_date: datetime,
    end_date: datetime,
    period: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    users, purchases = load_data()
    
    # Convert date objects to timezone-naive pandas datetime
    start_date = pd.to_datetime(start_date).tz_localize(None)
    end_date = pd.to_datetime(end_date).tz_localize(None)
    
    # Add time component based on period granularity
    if period == 'D':
        start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_date = end_date.replace(hour=23, minute=59, second=59, microsecond=999999)
    elif period == 'M':
        start_date = start_date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        end_date = (end_date + pd.offsets.MonthEnd(0)).replace(hour=23, minute=59, second=59, microsecond=999999)
    elif period == 'Q':
        start_date = (start_date - pd.offsets.QuarterBegin(startingMonth=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        end_date = (end_date + pd.offsets.QuarterEnd()).replace(hour=23, minute=59, second=59, microsecond=999999)
    elif period == 'Y':
        start_date = start_date.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
        end_date = end_date.replace(month=12, day=31, hour=23, minute=59, second=59, microsecond=999999)
    
    # Filter users based on date range
    users = users[(users['Created at'] >= start_date) & (users['Created at'] <= end_date)]
    
    # For purchases, we need all historical purchases for correct "returning" calculations
    # but filter by payment type
    filtered_purchases = purchases[purchases['Type [Payment]'].isin(payment_types_key)].copy()
    
    # Then create a separate filtered set for revenue calculations
    revenue_purchases = filtered_purchases[
        (filtered_purchases['Created at [Route Purchase]'] >= start_date) & 
        (filtered_purchases['Created at [Route Purchase]'] <= end_date)
    ].copy()
    
    # Ensure dates are properly sorted before processing
    users = users.sort_values('Created at')
    filtered_purchases = filtered_purchases.sort_values('Created at [Route Purchase]')
    
    # Process metrics using all historical purchase data for user metrics
    user_metrics = process_user_data(users, filtered_purchases, payment_types_key, period)
    # But use date-filtered purchases for revenue metrics
    revenue_metrics = process_revenue_data(users, revenue_purchases, payment_types_key, period)
    
    # Ensure proper date handling for each period type
    if period == 'D':
        user_metrics['Created at'] = user_metrics['Created at'].dt.to_period('D').dt.to_timestamp()
        revenue_metrics['Created at'] = revenue_metrics['Created at'].dt.to_period('D').dt.to_timestamp()
    elif period == 'M':
        user_metrics['Created at'] = user_metrics['Created at'].dt.to_period('M').dt.to_timestamp()
        revenue_metrics['Created at'] = revenue_metrics['Created at'].dt.to_period('M').dt.to_timestamp()
    elif period == 'Q':
        user_metrics['Created at'] = user_metrics['Created at'].dt.to_period('Q').dt.to_timestamp()
        revenue_metrics['Created at'] = revenue_metrics['Created at'].dt.to_period('Q').dt.to_timestamp()
    else:  # 'Y'
        user_metrics['Created at'] = user_metrics['Created at'].dt.to_period('Y').dt.to_timestamp()
        revenue_metrics['Created at'] = revenue_metrics['Created at'].dt.to_period('Y').dt.to_timestamp()
    
    # Ensure complete periods
    user_metrics = user_metrics[
        (user_metrics['Created at'] >= start_date) &
        (user_metrics['Created at'] <= end_date)
    ]
    revenue_metrics = revenue_metrics[
        (revenue_metrics['Created at'] >= start_date) &
        (revenue_metrics['Created at'] <= end_date)
    ]
    
    # Sort by date to ensure proper ordering
    user_metrics = user_metrics.sort_values('Created at')
    revenue_metrics = revenue_metrics.sort_values('Created at')
    
    return user_metrics, revenue_metrics



def get_user_id_range(users, month):
    month_users = users[users['Created at'].dt.to_period('M') == month]
    if len(month_users) > 0:
        numeric_ids = pd.to_numeric(month_users['Id'], errors='coerce')
        return numeric_ids.min(), numeric_ids.max()
    return np.nan, np.nan



def build_monthly_revenue_df():
    users, purchases = load_data()
    # Filter to 2020 onward
    purchases = purchases[purchases['Created at [Route Purchase]'] >= '2020-01-01'].copy()
    purchases['Created at [Route Purchase]'] = pd.to_datetime(purchases['Created at [Route Purchase]'], utc=True).dt.tz_localize(None)
    purchases.set_index('Created at [Route Purchase]', inplace=True)
    monthly_rev = purchases['Price [Route Purchase]'].resample('M').sum().reset_index()
    monthly_rev.columns = ['ds', 'y']
    return monthly_rev

def build_monthly_users_df():
    users, _ = load_data()
    users = users[users['Created at'] >= '2020-01-01'].copy()
    users['Created at'] = pd.to_datetime(users['Created at'], utc=True).dt.tz_localize(None)
    users.set_index('Created at', inplace=True)
    monthly_users = users['Id'].resample('M').count().reset_index()
    monthly_users.columns = ['ds', 'y']
    return monthly_users

def build_monthly_conversion_df():
    # Uses your existing process_filtered_data over the FULL range from 2020 onward
    df, _ = process_filtered_data(
        payment_types_key=('InAppPurchase','AndroidPayment','StripePayment'),  # or ALL_PAYMENT_TYPES
        start_date='2020-01-01',
        end_date='2100-01-01',
        period='M'
    )
    df['Conversion Rate'] = (df['New Paying Users'] / df['New Users'] * 100).fillna(0)
    return df[['Created at','Conversion Rate']].rename(columns={'Created at':'ds','Conversion Rate':'y'}).dropna()

def build_monthly_repeat_df():
    df, _ = process_filtered_data(
        payment_types_key=('InAppPurchase','AndroidPayment','StripePayment'),
        start_date='2020-01-01',
        end_date='2100-01-01',
        period='M'
    )
    df['Repeat Rate'] = (df['Returning, Repeat'].cumsum() / df['All New Paying'].cumsum() * 100).fillna(0)
    return df[['Created at','Repeat Rate']].rename(columns={'Created at':'ds','Repeat Rate':'y'}).dropna()
