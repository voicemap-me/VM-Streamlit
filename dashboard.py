import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
from prophet import Prophet
from prophet.plot import plot_components
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(page_title="VM-Streamlit", layout="wide")

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["User Acquisition", "Payment Metrics", "Conversion Analysis", "Revenue Metrics", "Unit Economics", "Forecasting", "LTV"])



# Define all payment types
ALL_PAYMENT_TYPES = ['InAppPurchase', 'AndroidPayment', 'StripePayment', 'CouponRedemptionCreditReseller', 'FreePayment', 'CouponRedemptionReseller', 'CouponRedemptionResellerReply', 'CouponRedemptionPaid', 'CouponRedemption', 'SwfRedemption', 'BraintreePayment', 'PaypalPayment']
DEFAULT_PAYMENT_TYPES = ['InAppPurchase', 'AndroidPayment', 'StripePayment']

# def create_prophet_forecast(data, column_name, periods=12):
#     """
#     Create a forecast using Facebook Prophet
    
#     Parameters:
#     data: DataFrame with 'Created at' column and the target column
#     column_name: Name of the column to forecast
#     periods: Number of periods to forecast
    
#     Returns:
#     DataFrame with the forecast and confidence intervals
#     """
#     df_prophet = pd.DataFrame({
#         'ds': data['Created at'],
#         'y': data[column_name]
#     })
    
#     # Configure seasonality based on granularity
#     if selected_period == 'D':
#         m = Prophet(
#             yearly_seasonality=True,
#             weekly_seasonality=True,
#             daily_seasonality=False,
#             seasonality_mode='multiplicative'
#         )
#     elif selected_period == 'M':
#         m = Prophet(
#             yearly_seasonality=True,
#             weekly_seasonality=False,
#             daily_seasonality=False,
#             seasonality_mode='multiplicative'
#         )
#     else:  # 'Q' or 'Y'
#         m = Prophet(
#             yearly_seasonality=True,
#             weekly_seasonality=False,
#             daily_seasonality=False,
#             seasonality_mode='additive'
#         )
    
#     m.fit(df_prophet)
    
#     # Get the last date from actual data and ensure we start forecasting from the next period
#     last_date = df_prophet['ds'].max()
    
#     # Create future dataframe starting after the last actual date
#     if selected_period == 'D':
#         next_start = last_date + pd.Timedelta(days=1)
#         future_dates = pd.date_range(start=next_start, periods=periods, freq='D')
#     elif selected_period == 'M':
#         # Ensure we start from the next month after the last actual data
#         next_month_start = (last_date + pd.offsets.MonthEnd(0) + pd.Timedelta(days=1))
#         future_dates = pd.date_range(start=next_month_start, periods=periods, freq='ME')
#     elif selected_period == 'Q':
#         next_quarter_start = (last_date + pd.offsets.QuarterEnd(0) + pd.Timedelta(days=1))
#         future_dates = pd.date_range(start=next_quarter_start, periods=periods, freq='Q')
#     else:  # 'Y'
#         next_year_start = pd.Timestamp(last_date.year + 1, 1, 1)
#         future_dates = pd.date_range(start=next_year_start, periods=periods, freq='Y')
    
#     future = pd.DataFrame({'ds': future_dates})
    
#     # Make forecast
#     forecast = m.predict(future)
    
#     # Combine with historical data for plotting
#     historical = pd.DataFrame({
#         'ds': df_prophet['ds'],
#         'yhat': df_prophet['y'],
#         'yhat_lower': df_prophet['y'],
#         'yhat_upper': df_prophet['y']
#     })
    
#     forecast = pd.concat([historical, forecast])
    
#     return forecast, m


# def create_prophet_forecast(data, column_name, periods=12):
#     """
#     Create a forecast using Facebook Prophet
    
#     Parameters:
#     data: DataFrame with 'Created at' column and the target column
#     column_name: Name of the column to forecast
#     periods: Number of periods to forecast
    
#     Returns:
#     DataFrame with the forecast and confidence intervals
#     """
#     # Filter data to a fixed date range (e.g., 2020 onwards)
#     data = data[data['Created at'] >= '2020-01-01']
    
#     df_prophet = pd.DataFrame({
#         'ds': data['Created at'],
#         'y': data[column_name]
#     })
    
#     # Cap and floor for growth (adjust based on your data)
#     df_prophet['cap'] = df_prophet['y'].max() * 2  # Adjust multiplier as needed
#     df_prophet['floor'] = df_prophet['y'].min() * 0.8  # Adjust multiplier as needed
    
#     # Configure seasonality based on granularity
#     if selected_period == 'D':
#         m = Prophet(
#             yearly_seasonality=True,
#             weekly_seasonality=True,
#             daily_seasonality=False,
#             seasonality_mode='multiplicative',
#             growth='logistic'  # Use logistic growth to cap growth
#         )
#     elif selected_period == 'M':
#         m = Prophet(
#             yearly_seasonality=True,
#             weekly_seasonality=False,
#             daily_seasonality=False,
#             seasonality_mode='multiplicative',
#             growth='logistic'  # Use logistic growth to cap growth
#         )
#     else:  # 'Q' or 'Y'
#         m = Prophet(
#             yearly_seasonality=True,
#             weekly_seasonality=False,
#             daily_seasonality=False,
#             seasonality_mode='additive',
#             growth='logistic'  # Use logistic growth to cap growth
#         )
    
#     # Add custom seasonality if needed
#     m.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    
#     m.fit(df_prophet)
    
#     # Get the last date from actual data and ensure we start forecasting from the next period
#     last_date = df_prophet['ds'].max()
    
#     # Create future dataframe starting after the last actual date
#     if selected_period == 'D':
#         next_start = last_date + pd.Timedelta(days=1)
#         future_dates = pd.date_range(start=next_start, periods=periods, freq='D')
#     elif selected_period == 'M':
#         # Ensure we start from the next month after the last actual data
#         next_month_start = (last_date + pd.offsets.MonthEnd(0) + pd.Timedelta(days=1))
#         future_dates = pd.date_range(start=next_month_start, periods=periods, freq='ME')
#     elif selected_period == 'Q':
#         next_quarter_start = (last_date + pd.offsets.QuarterEnd(0) + pd.Timedelta(days=1))
#         future_dates = pd.date_range(start=next_quarter_start, periods=periods, freq='Q')
#     else:  # 'Y'
#         next_year_start = pd.Timestamp(last_date.year + 1, 1, 1)
#         future_dates = pd.date_range(start=next_year_start, periods=periods, freq='Y')
    
#     future = pd.DataFrame({'ds': future_dates})
    
#     # Add cap and floor to future dataframe
#     future['cap'] = df_prophet['cap'].max()
#     future['floor'] = df_prophet['floor'].min()
    
#     # Make forecast
#     forecast = m.predict(future)
    
#     # Combine with historical data for plotting
#     historical = pd.DataFrame({
#         'ds': df_prophet['ds'],
#         'yhat': df_prophet['y'],
#         'yhat_lower': df_prophet['y'],
#         'yhat_upper': df_prophet['y']
#     })
    
#     forecast = pd.concat([historical, forecast])
    
#     return forecast, m
    

# def create_components_forecast(data, column_name):
#     """
#     Create a forecast with all seasonality components for visualization
#     """
#     df_prophet = pd.DataFrame({
#         'ds': data['Created at'],
#         'y': data[column_name]
#     })
    
#     # Configure Prophet with all seasonality components
#     m = Prophet(
#         yearly_seasonality=True,
#         weekly_seasonality=True,
#         daily_seasonality=True,
#         seasonality_mode='multiplicative'
#     )
    
#     m.fit(df_prophet)
    
#     # Create future dataframe for components
#     future = m.make_future_dataframe(periods=0, freq='D')  # No future periods needed for components
#     forecast = m.predict(future)
    
#     return forecast, m

def aggregate_by_period(df, period='M'):
    """
    Aggregate data by specified time period (Daily, Monthly, Quarterly, or Yearly)
    """
    temp_df = df.copy()
    
    if not pd.api.types.is_datetime64_any_dtype(temp_df['Created at']):
        temp_df['Created at'] = pd.to_datetime(temp_df['Created at'])
    
    # Create period column and ensure proper ordering
    if period == 'D':
        temp_df['Period'] = temp_df['Created at'].dt.to_period('D')
    elif period == 'M':
        temp_df['Period'] = temp_df['Created at'].dt.to_period('M')
    elif period == 'Q':
        temp_df['Period'] = temp_df['Created at'].dt.to_period('Q')
    else:  # 'Y'
        temp_df['Period'] = temp_df['Created at'].dt.to_period('Y')
    
    # Sort by period before aggregating
    temp_df = temp_df.sort_values('Period')
    
    # Aggregate numeric columns
    numeric_columns = temp_df.select_dtypes(include=[np.number]).columns
    agg_df = temp_df.groupby('Period').agg({col: 'sum' for col in numeric_columns}).reset_index()
    
    # Convert period to timestamp for plotting
    agg_df['Created at'] = agg_df['Period'].astype(str).apply(pd.to_datetime)
    
    # Recalculate derived metrics if they exist
    if 'Percentage Returning, Repeat' in df.columns:
        agg_df['Percentage Returning, Repeat'] = (agg_df['Returning, Repeat'] / agg_df['Total Paying'] * 100).round(2)
    if 'Cumulative All New Paying' in df.columns:
        agg_df['Cumulative All New Paying'] = agg_df['All New Paying'].cumsum()
    
    return agg_df.sort_values('Created at')

def format_date_axis(fig, period):
    """Update date axis format based on selected period"""
    if isinstance(fig.data[0].x[0], str):
        start_date = pd.to_datetime(fig.data[0].x[0])
        end_date = pd.to_datetime(fig.data[0].x[-1])
    else:
        start_date = pd.to_datetime(fig.data[0].x[0])
        end_date = pd.to_datetime(fig.data[0].x[-1])
        
    # Convert to pandas Timestamp if needed and calculate days
    if isinstance(start_date, (np.datetime64, pd.Timestamp)):
        start_date = pd.Timestamp(start_date)
    if isinstance(end_date, (np.datetime64, pd.Timestamp)):
        end_date = pd.Timestamp(end_date)
        
    days_in_range = (end_date - start_date).days
    
    if period == 'Y':
        date_format = '%Y'
        dtick = 'M12'
    elif period == 'Q':
        date_format = 'Q%q %Y'
        dtick = 'M3'
    elif period == 'M':
        # Adjust format and tick frequency based on date range
        if days_in_range > 1825:  # More than 5 years
            date_format = '%Y'  # Just show year
            dtick = 'M12'  # Show yearly ticks
        elif days_in_range > 730:  # More than 2 years
            date_format = '%b %Y'
            dtick = 'M2'  # Show every 6 months
        elif days_in_range > 365:  # More than 1 year
            date_format = '%b %Y'
            dtick = 'M1'  # Show quarterly
        else:
            date_format = '%b %Y'
            dtick = 'M1'  # Show every month
    else:  # 'D'
        if days_in_range > 90:  # More than 90 days
            date_format = '%b %Y'
            dtick = 'M1'
        else:
            date_format = '%Y-%m-%d'
            dtick = 'D7'
    
    fig.update_xaxes(
        type='date',
        tickformat=date_format,
        dtick=dtick,
        tickangle=90,
        tickmode='linear',
        showgrid=False,  # Turn off x-axis grid
        automargin=True,
        tickfont=dict(size=10)
    )
    
    # Add padding and adjust layout
    fig.update_layout(
        margin=dict(b=70, l=50, r=50, t=50),
        xaxis=dict(
            rangeslider=dict(visible=False),
            showspikes=True,
            spikemode='across',
            spikesnap='cursor',
            showline=True,
            
        ),
        yaxis=dict(showgrid=False),  # Turn off y-axis grid
        hovermode='x unified'
    )
    
    return fig

@st.cache_data
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



@st.cache_data
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



def get_user_id_range(users, month):
    month_users = users[users['Created at'].dt.to_period('M') == month]
    if len(month_users) > 0:
        numeric_ids = pd.to_numeric(month_users['Id'], errors='coerce')
        return numeric_ids.min(), numeric_ids.max()
    return np.nan, np.nan

def process_user_data(users, purchases, selected_payment_types, period='M'):
    # Define paid payment types that disqualify a user from being "free"
    paid_payment_types = ['AndroidPayment', 'BraintreePayment', 'CouponRedemptionCreditReseller', 
                         'CouponRedemptionReseller', 'CouponRedemptionResellerReply', 
                         'InAppPurchase', 'PaypalPayment', 'StripePayment']
    
    def calculate_monthly_metrics(month):
        month_start = month.to_timestamp()
        month_end = month.to_timestamp(how='end')
        
        # Get users for this specific period only
        period_users_mask = (users['Created at'] >= month_start) & (users['Created at'] <= month_end)
        period_users = users[period_users_mask]
        period_user_ids = set(period_users['Id'])
        
        # Get paid users for this period
        paid_users_this_period = set(purchases[
            (purchases['Type [Payment]'].isin(paid_payment_types)) & 
            (purchases['Created at [Route Purchase]'] >= month_start) &
            (purchases['Created at [Route Purchase]'] <= month_end) &
            (purchases['Id [User]'].isin(period_user_ids))
        ]['Id [User]'].unique())
        
        # Free users are those who haven't made a paid purchase in this period
        free_users = len(period_user_ids - paid_users_this_period)
        
        # New users for this period
        new_users = len(period_user_ids)
        
        # First-time purchasers in this period
        first_time_purchasers = purchases[
            (purchases['Type [Payment]'].isin(selected_payment_types)) &
            (purchases['Created at [Route Purchase]'] >= month_start) & 
            (purchases['Created at [Route Purchase]'] <= month_end) &
            (~purchases['Id [User]'].isin(
                purchases[
                    (purchases['Type [Payment]'].isin(selected_payment_types)) &
                    (purchases['Created at [Route Purchase]'] < month_start)
                ]['Id [User]']
            ))
        ]['Id [User]'].unique()
        
        # New paying users (signed up and paid in same period)
        new_paying = len(set(first_time_purchasers) & period_user_ids)
        
        # Returning first purchase (users who made their first purchase but didn't sign up this period)
        returning_first = len(set(first_time_purchasers) - period_user_ids)
        
        # All purchasers this period
        all_purchasers = purchases[
            (purchases['Type [Payment]'].isin(selected_payment_types)) &
            (purchases['Created at [Route Purchase]'] >= month_start) & 
            (purchases['Created at [Route Purchase]'] <= month_end)
        ]['Id [User]'].unique()
        
        # All returning (all purchasers minus new users)
        all_returning = len(set(all_purchasers) - period_user_ids)
        
        # Total paying
        total_paying = len(all_purchasers)
        
        return pd.Series({
            'New Users': new_users,
            'Free Users': free_users,
            'New Paying Users': new_paying,
            'Returning, First Purchase': returning_first,
            'All Returning': all_returning,
            'Total Paying': total_paying
        })
    
    # Calculate metrics for each period
    if period == 'D':
        period_index = users['Created at'].dt.to_period('D').unique()
    elif period == 'Q':
        period_index = users['Created at'].dt.to_period('Q').unique()
    elif period == 'Y':
        period_index = users['Created at'].dt.to_period('Y').unique()
    else:  # 'M'
        period_index = users['Created at'].dt.to_period('M').unique()

    period_metrics = pd.DataFrame([calculate_monthly_metrics(month) for month in period_index],
                                 index=period_index)
    
    # Calculate derived metrics
    period_metrics['All New Paying'] = period_metrics['New Paying Users'] + period_metrics['Returning, First Purchase']
    period_metrics['Returning, Repeat'] = period_metrics['All Returning'] - period_metrics['Returning, First Purchase']
    period_metrics['Percentage Returning, Repeat'] = (period_metrics['Returning, Repeat'] / period_metrics['Total Paying'] * 100).round(2)
    period_metrics['Cumulative All New Paying'] = period_metrics['All New Paying'].cumsum()
    
    # Convert to timestamps
    period_metrics = period_metrics.reset_index(drop=True)
    period_metrics['Created at'] = [period.to_timestamp() for period in period_index]
    
    return period_metrics

def process_revenue_data(users, purchases, selected_payment_types, period='M'):
    filtered_purchases = purchases[purchases['Type [Payment]'].isin(selected_payment_types)]
    
    def calculate_monthly_revenue(month):
        month_start = month.to_timestamp()
        month_end = month.to_timestamp(how='end')
        
        # New users for this period
        new_users_mask = calculate_period_mask(users, period, month)
        new_users_ids = users[new_users_mask]['Id']
        
        # Total revenue from new users
        new_users_revenue = filtered_purchases[
            filtered_purchases['Id [User]'].isin(new_users_ids)
        ]['Price [Route Purchase]'].sum()
        
        # New paying users revenue (first month)
        new_paying_revenue = filtered_purchases[
            (filtered_purchases['Id [User]'].isin(new_users_ids)) & 
            (filtered_purchases['Created at [Route Purchase]'] >= month_start) & 
            (filtered_purchases['Created at [Route Purchase]'] <= month_end)
        ]['Price [Route Purchase]'].sum()
        
        # First-time purchasers this period
        first_time_purchasers = filtered_purchases[
            (filtered_purchases['Created at [Route Purchase]'] >= month_start) & 
            (filtered_purchases['Created at [Route Purchase]'] <= month_end) &
            (~filtered_purchases['Id [User]'].isin(
                filtered_purchases[filtered_purchases['Created at [Route Purchase]'] < month_start]['Id [User]']
            ))
        ]
        
        # Revenue from returning first-time purchasers
        previous_users = users[users['Created at'] < month_start]['Id']
        returning_first_revenue = first_time_purchasers[
            first_time_purchasers['Id [User]'].isin(previous_users)
        ]['Price [Route Purchase]'].sum()
        
        # All revenue this period
        month_revenue = filtered_purchases[
            (filtered_purchases['Created at [Route Purchase]'] >= month_start) & 
            (filtered_purchases['Created at [Route Purchase]'] <= month_end)
        ]
        
        returning_revenue = month_revenue[~month_revenue['Id [User]'].isin(new_users_ids)]['Price [Route Purchase]'].sum()
        total_revenue = month_revenue['Price [Route Purchase]'].sum()
        total_downloads = month_revenue['Price [Route Purchase]'].count()
        
        return pd.Series({
            'New Users Revenue': new_users_revenue,
            'New Paying Users Revenue': new_paying_revenue,
            'Returning First Purchase Revenue': returning_first_revenue,
            'All Returning Revenue': returning_revenue,
            'Total Revenue': total_revenue,
            'Total Downloads': total_downloads 
        })
    
    # Calculate metrics for each month
    if period == 'D':
        period_index = users['Created at'].dt.to_period('D').unique()
    elif period == 'Q':
        period_index = users['Created at'].dt.to_period('Q').unique()
    elif period == 'Y':
        period_index = users['Created at'].dt.to_period('Y').unique()
    else:  # 'M'
        period_index = users['Created at'].dt.to_period('M').unique()

    monthly_revenue = pd.DataFrame([
        calculate_monthly_revenue(month) 
        for month in period_index
    ], index=period_index)
    
    # Calculate derived metrics
    monthly_revenue['All New Paying Revenue'] = monthly_revenue['New Paying Users Revenue'] + monthly_revenue['Returning First Purchase Revenue']
    monthly_revenue['Returning Repeat Revenue'] = monthly_revenue['All Returning Revenue'] - monthly_revenue['Returning First Purchase Revenue']
    monthly_revenue['Percentage Returning Repeat Revenue'] = (monthly_revenue['Returning Repeat Revenue'] / monthly_revenue['Total Revenue'] * 100).round(2)
    monthly_revenue['Cumulative All New Paying Revenue'] = monthly_revenue['All New Paying Revenue'].cumsum()
    
    # Convert the period index to timestamp
    monthly_revenue = monthly_revenue.reset_index(drop=True)  # Drop the old index
    monthly_revenue['Created at'] = [period.to_timestamp() for period in period_index]
    
    return monthly_revenue

# Load base data
users, purchases = load_data()

# Sidebar filters
st.sidebar.title("Data Filtering")

st.sidebar.caption("Free payment types are not included in Revenue figures. Payment filtering only works for Payment and Revenue related metrics.")

# Add Select All checkbox
select_all = st.sidebar.checkbox("Select All Payment Types")

# Add payment type filter
selected_payment_types = st.sidebar.multiselect(
    "Select Payment Types",
    options=ALL_PAYMENT_TYPES,
    default=DEFAULT_PAYMENT_TYPES if not select_all else ALL_PAYMENT_TYPES,
)

# Update selected_payment_types if select_all is True
if select_all:
    selected_payment_types = ALL_PAYMENT_TYPES

if not selected_payment_types:
    st.warning("Please select at least one payment type.")
    st.stop()

# st.sidebar.caption("Select whether you want to look at the data on a Daily, Monthly, Quarterly or Yearly granularity.")

st.sidebar.title("Date Range Selection")

# Add time period selector to sidebar
time_period = st.sidebar.selectbox(
    "Select Date Granularity",
    options=['Daily', 'Monthly', 'Quarterly', 'Yearly'],  # Added 'Daily'
    index=1  # Default to Monthly
)

# Convert selection to period code
period_map = {
    'Daily': 'D',    # Add daily option
    'Monthly': 'M',
    'Quarterly': 'Q',
    'Yearly': 'Y'
}
selected_period = period_map[time_period]

if not selected_payment_types:
    st.warning("Please select at least one payment type.")
    st.stop()

# Convert selected payment types to tuple for caching
payment_types_key = tuple(sorted(selected_payment_types))

def calculate_period_mask(users, period_type, month):
    """Helper function to create the correct period mask for any granularity"""
    if isinstance(month, pd.Period):
        # Match the period type exactly to avoid any granularity mismatch
        return users['Created at'].dt.to_period(month.freq) == month
    else:
        # Fallback for timestamp comparison
        month_start = month
        if period_type == 'D':
            return users['Created at'].dt.date == month_start.date()
        elif period_type == 'Q':
            return users['Created at'].dt.to_period('Q') == month_start.to_period('Q')
        elif period_type == 'Y':
            return users['Created at'].dt.to_period('Y') == month_start.to_period('Y')
        else:  # 'M'
            return users['Created at'].dt.to_period('M') == month_start.to_period('M')


# Get initial date range from raw data (we already have users loaded)
min_date = users['Created at'].min().date()
max_date = users['Created at'].max().date()


def get_date_ranges(min_date, max_date):
    last_data_date = max_date
    return {
        "Custom": {
            "start": min_date,
            "end": max_date,
        },
        "Last 7 Days": {
            "start": max_date - pd.Timedelta(days=7),
            "end": max_date,
        },
        "Last 30 Days": {
            "start": max_date - pd.Timedelta(days=30),
            "end": max_date,
        },
        "Last 90 Days": {
            "start": max_date - pd.Timedelta(days=90),
            "end": max_date,
        },
        "Year to Date": {
            "start": pd.Timestamp(max_date.year, 1, 1).date(),
            "end": max_date,
        },
        "Last 12 Months": {
            "start": max_date - pd.Timedelta(days=365),
            "end": max_date,
        }
    }

date_ranges = get_date_ranges(min_date, max_date)
# selected_range = st.sidebar.selectbox(
#     "Select Date Range",
#     options=list(date_ranges.keys()),
#     index=0  # Default to Custom
# )



col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input(
            "Start Date",
            value=datetime(2020, 2, 1).date(),  # Set default to Jan 1, 2020
            min_value=min_date,
            max_value=max_date
    )
with col2:
    end_date = st.date_input(
            "End Date",
            value=max_date,
            min_value=min_date,
            max_value=max_date
    )


# Validate date range
if start_date > end_date:
    st.error("End date must be after start date")
    st.stop()

# Ensure dates are within available data range
start_date = max(start_date, min_date)
end_date = min(end_date, max_date)

if time_period == 'Daily' and (end_date - start_date).days > 90:
    st.warning("Daily view is showing more than 90 days of data. Consider using a shorter date range or switching to monthly view for better performance.")




# Then process the filtered data
df, df_revenue = process_filtered_data(
    payment_types_key=tuple(sorted(selected_payment_types)),
    start_date=start_date,
    end_date=end_date,
    period=selected_period
)

filtered_df = df
filtered_df_revenue = df_revenue

fig_rev_breakdown = px.bar(filtered_df_revenue, x='Created at',
    y=['New Paying Users Revenue', 'Returning First Purchase Revenue', 'Returning Repeat Revenue'],
    title=f'{time_period} Revenue Breakdown by User Type',
    labels={'value': 'Revenue ($)', 'variable': 'Revenue Type'},
    color_discrete_sequence=['#87CEEB', '#1E90FF', 'rgb(255, 171, 171)']  # Light blue, Dark blue, Light salmon
)

fig_rev_breakdown.update_layout(
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=1.05
    )
)

# Add 100% stacked revenue chart
# Calculate percentages for stacked chart
revenue_pct = filtered_df_revenue.copy()
total_revenue = revenue_pct['Total Revenue']
revenue_pct['New Paying %'] = (revenue_pct['New Paying Users Revenue'] / total_revenue * 100)
revenue_pct['First Purchase %'] = (revenue_pct['Returning First Purchase Revenue'] / total_revenue * 100)
revenue_pct['Repeat %'] = (revenue_pct['Returning Repeat Revenue'] / total_revenue * 100)

fig_rev_pct = go.Figure()
fig_rev_pct.add_trace(go.Bar(
    x=revenue_pct['Created at'],
    y=revenue_pct['New Paying %'],
    name='New Paying Users Revenue',
    marker_color='#87CEEB'  # Light blue
))
fig_rev_pct.add_trace(go.Bar(
    x=revenue_pct['Created at'],
    y=revenue_pct['First Purchase %'],
    name='Returning First Purchase Revenue',
    marker_color='#1E90FF'  # Dark blue
))
fig_rev_pct.add_trace(go.Bar(
    x=revenue_pct['Created at'],
    y=revenue_pct['Repeat %'],
    name='Returning Repeat Revenue',
    marker_color='rgb(255, 171, 171)'  # Light salmon
))

fig_rev_pct.update_layout(
    barmode='stack',
    title=f'{time_period} Revenue Distribution by User Type (%)',
    yaxis_title='Percentage of Total Revenue',
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=1.05
    ),
    hovermode='x unified'
)

fig_rev_pct = format_date_axis(fig_rev_pct, selected_period)

hover_settings = dict(
    hoverlabel=dict(
        bgcolor="rgba(44, 47, 51, 0.85)", 
        font=dict(color="white", size=12, family="Arial"),
        bordercolor="rgba(255, 255, 255, 0.2)",
        namelength=-1  # This prevents truncation of names
    ),
    hoverdistance=100,  # Increases the hover "snap" distance
    hovermode='x unified'
)

with tab1:
    st.markdown("""
    ## 📊 Key Metrics
    """)
    # Top level metrics in columns
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total New Users", f"{filtered_df['New Users'].sum():,}")
    with col2:
        st.metric("Total Paying Users", f"{filtered_df['Total Paying'].sum():,}")
    with col3:
        conversion_rate = (filtered_df['All New Paying'].sum() / filtered_df['New Users'].sum() * 100)
        st.metric("Overall Conversion Rate", f"{conversion_rate:.1f}%")
    with col4:
        repeat_rate = filtered_df['Percentage Returning, Repeat'].mean()
        st.metric("Avg Repeat Purchase Rate", f"{repeat_rate:.1f}%")
    with col5:
        st.metric("Total Downloads", f"{int(filtered_df_revenue['Total Downloads'].sum()):,}")
    
    st.markdown("""
    ## 📈 User Distribution Analysis
    This chart shows how our user base is distributed across different types over time:
    
    - **Free Users:** The total number of existing and New Users, who have never made a direct paid purchase. Users who have only made purchases with free payment types (FreePayment, SwfRedemption, CouponRedemption, CouponRedemptionPaid) are still considered Free Users.
    - **New Paying Users:** The total number of users who have signed up and made their first payment within the sign up month.
    - **Returning, First Purchase:** The number of existing users who have returned to make their first payment.
    - **Returning, Repeat:** The number of existing users who have returned to make another payment outside of their first payment month.
                """ )
        
    # Calculate percentages for each user type
    user_percentages = pd.DataFrame({
    'Created at': filtered_df['Created at'],
    'New Paying Users': filtered_df['New Paying Users'],
    'Free Users': filtered_df['Free Users'],
    'Returning First Purchase': filtered_df['Returning, First Purchase'],
    'Returning Repeat': filtered_df['Returning, Repeat']
})

    # Calculate total for each period
    user_percentages['Total'] = user_percentages['New Paying Users'] + user_percentages['Free Users'] + \
                            user_percentages['Returning First Purchase'] + user_percentages['Returning Repeat']

    # Convert to percentages
    for col in ['New Paying Users', 'Free Users', 'Returning First Purchase', 'Returning Repeat']:
        user_percentages[col] = (user_percentages[col] / user_percentages['Total'] * 100)

    # Create 100% stacked bar chart
    fig1 = go.Figure()

    # Add traces for each user type with custom colors
    fig1.add_trace(go.Bar(
        x=user_percentages['Created at'],
        y=user_percentages['New Paying Users'],
        name='New Paying Users',
        marker_color='#2E86C1'
    ))

    fig1.add_trace(go.Bar(
        x=user_percentages['Created at'],
        y=user_percentages['Free Users'],
        name='Free Users',
        marker_color='#28B463'
    ))

    fig1.add_trace(go.Bar(
        x=user_percentages['Created at'],
        y=user_percentages['Returning First Purchase'],
        name='Returning First Purchase',
        marker_color='#AF7AC5'
    ))

    fig1.add_trace(go.Bar(
        x=user_percentages['Created at'],
        y=user_percentages['Returning Repeat'],
        name='Returning Repeat',
        marker_color='#E59866'
    ))

    fig1.update_layout(
        title=f'{time_period} User Type Distribution',
        xaxis_title='Date',
        yaxis_title='Percentage of Users',
        barmode='relative',
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )


    fig1 = format_date_axis(fig1, selected_period)
    st.plotly_chart(fig1, use_container_width=True)

with tab2:
    # First, prepare the data
    payment_dist = purchases[
        (purchases['Created at [Route Purchase]'] >= pd.to_datetime(start_date)) & 
        (purchases['Created at [Route Purchase]'] <= pd.to_datetime(end_date)) &
        (purchases['Type [Payment]'].isin(selected_payment_types))
    ].copy()
    
    # Create period column based on selected granularity
    if selected_period == 'D':
        payment_dist['Period'] = payment_dist['Created at [Route Purchase]'].dt.to_period('D')
    elif selected_period == 'M':
        payment_dist['Period'] = payment_dist['Created at [Route Purchase]'].dt.to_period('M')
    elif selected_period == 'Q':
        payment_dist['Period'] = payment_dist['Created at [Route Purchase]'].dt.to_period('Q')
    else:  # 'Y'
        payment_dist['Period'] = payment_dist['Created at [Route Purchase]'].dt.to_period('Y')

    # Group and pivot data
    payment_counts = payment_dist.groupby(['Period', 'Type [Payment]']).size().reset_index(name='Count')
    payment_pivot = payment_counts.pivot(
        index='Period',
        columns='Type [Payment]',
        values='Count'
    ).fillna(0)
    
    # Convert period to timestamp for plotting
    payment_pivot.index = payment_pivot.index.map(lambda x: x.to_timestamp())
    
    # Create stacked bar chart
    fig_payment_dist = go.Figure()
    
    for payment_type in payment_pivot.columns:
        fig_payment_dist.add_trace(
            go.Bar(
                name=payment_type,
                x=payment_pivot.index,
                y=payment_pivot[payment_type],
                hovertemplate="<br>".join([
                    "Downloads: %{y:,.0f}",
                ])
            )
        )
    
    fig_payment_dist.update_layout(
        title=f'{time_period} Distribution of Payment Types',
        xaxis_title='Date',
        yaxis_title='Number of Downloads',
        barmode='stack',
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    fig_payment_dist = format_date_axis(fig_payment_dist, selected_period)

     
    st.markdown("""
    ### 💳 Payment Methods Distribution
    Which payment methods do users prefer?
    """)

    st.plotly_chart(fig_payment_dist, use_container_width=True)

    st.markdown("""
    ### 👥 User Payment Behavior
    This breakdown helps us understand the mix between:
    - **New Paying Users:** The total number of users who have signed up and made their first payment within the sign up month.
    - **Returning, First Purchase Users:** The number of existing users who have returned to make their first payment.
    - **Returning, Repeat Users:** The number of existing users who have returned to make another payment outside of their first payment month.
    """)
    
    # User Payment Behavior Breakdown
    fig3 = px.bar(filtered_df, x='Created at',
                 y=['New Paying Users', 'Returning, First Purchase', 'Returning, Repeat'],
                 title=f'{time_period} User Payment Behavior Breakdown',
                 labels={'value': 'Number of Users', 'variable': 'User Type'})
    fig3 = format_date_axis(fig3, selected_period)
    st.plotly_chart(fig3, use_container_width=True)
    
    # Distribution of User Payment Types
    avg_percentages = pd.DataFrame({
        'Category': ['New Paying', 'Returning (First)', 'Returning (Repeat)'],
        'Percentage': [
            filtered_df['New Paying Users'].sum() / filtered_df['Total Paying'].sum() * 100,
            filtered_df['Returning, First Purchase'].sum() / filtered_df['Total Paying'].sum() * 100,
            filtered_df['Returning, Repeat'].sum() / filtered_df['Total Paying'].sum() * 100
        ]
    })
    fig4 = px.pie(avg_percentages, values='Percentage', names='Category',
                 title=f'Distribution of User Payment Types ({time_period})')
    st.plotly_chart(fig4, use_container_width=True)

    

with tab3:
    st.markdown("""
    ### 🎯 Conversion Analysis
    Monitor how effectively we convert new users into paying customers.
    ```
    Conversion Rate = (All New Paying Users / Total New Users) × 100
    ```
    """)

    # Conversion metrics
    monthly_conversion = (filtered_df['All New Paying'] / filtered_df['New Users'] * 100).round(2)
    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(x=filtered_df['Created at'],
                             y=monthly_conversion,
                             mode='lines+markers',
                             name=f'{time_period} Conversion Rate',
                             line=dict(width=2)))
    fig5.update_layout(
        title=f'{time_period} Conversion Rate Trend',
        xaxis_title='Date',
        yaxis_title='Conversion Rate (%)',
        yaxis_range=[0, max(monthly_conversion) * 1.1],
        hovermode='x unified'
    )
    fig5 = format_date_axis(fig5, selected_period)
    st.plotly_chart(fig5, use_container_width=True)

with tab4:
    st.markdown("""
    ### 💰 Revenue Growth Trends
    
    Track both period revenue and cumulative growth trends.
    """)
    # Revenue Growth
    fig_revenue = go.Figure()
    fig_revenue.add_trace(go.Bar(
        x=filtered_df_revenue['Created at'],
        y=filtered_df_revenue['Total Revenue'],
        name=f'{time_period} Revenue',
        marker_color='#87CEEB'  # Light blue to match other charts
    ))
    fig_revenue.add_trace(go.Scatter(
        x=filtered_df_revenue['Created at'],
        y=filtered_df_revenue['Total Revenue'].cumsum(),
        name='Cumulative Revenue',
        line=dict(color='#1E90FF', width=3),  # Darker blue for contrast
        yaxis='y2'
    ))
    fig_revenue.update_layout(
        title=f'{time_period} Revenue Growth',
        yaxis_title='Revenue per Period ($)',
        yaxis2=dict(title='Cumulative Revenue ($)', overlaying='y', side='right'),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode='x unified'
    )
    fig_revenue = format_date_axis(fig_revenue, selected_period)
    st.plotly_chart(fig_revenue, use_container_width=True)

    st.markdown("""
    ### 📊 Revenue Source Analysis
    
    Understanding revenue composition helps identify our most valuable user segments:
    - **New User Revenue:** Revenue from New Users
    - **Returning, First Purchase Revenue :** Revenue from Returning, First Purchase Users
    - **Returning, Repeat Revenue:** Revenue from Returning, Repeat Users
    """)
    # Revenue Breakdown
    fig_rev_breakdown = px.bar(filtered_df_revenue, x='Created at',
        y=['New Paying Users Revenue', 'Returning First Purchase Revenue', 'Returning Repeat Revenue'],
        title=f'{time_period} Revenue Breakdown by User Type',
        labels={'value': 'Revenue ($)', 'variable': 'Revenue Type'},
        color_discrete_sequence=['#87CEEB', '#1E90FF', 'rgb(255, 171, 171)']  # Light blue, Dark blue, Light salmon
    )
    fig_rev_breakdown.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        showlegend=True,
        hovermode='x unified'
    )
    fig_rev_breakdown = format_date_axis(fig_rev_breakdown, selected_period)
    st.plotly_chart(fig_rev_breakdown, use_container_width=True)

    
    # Revenue Distribution Percentage
    fig_rev_pct = go.Figure()
    fig_rev_pct.add_trace(go.Bar(
        x=revenue_pct['Created at'],
        y=revenue_pct['New Paying %'],
        name='New Paying Users Revenue',
        marker_color='#87CEEB'  # Light blue
    ))
    fig_rev_pct.add_trace(go.Bar(
        x=revenue_pct['Created at'],
        y=revenue_pct['First Purchase %'],
        name='Returning First Purchase Revenue',
        marker_color='#1E90FF'  # Dark blue
    ))
    fig_rev_pct.add_trace(go.Bar(
        x=revenue_pct['Created at'],
        y=revenue_pct['Repeat %'],
        name='Returning Repeat Revenue',
        marker_color='rgb(255, 171, 171)'  # Light salmon
    ))
    fig_rev_pct.update_layout(
        barmode='stack',
        title=f'{time_period} Revenue Distribution by User Type (%)',
        yaxis_title='Percentage of Total Revenue',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode='x unified'
    )
    fig_rev_pct = format_date_axis(fig_rev_pct, selected_period)
    st.plotly_chart(fig_rev_pct, use_container_width=True)

    # Revenue per User Trend
    monthly_arpu = filtered_df_revenue['Total Revenue'] / filtered_df['New Users']
    monthly_arppu = filtered_df_revenue['Total Revenue'] / filtered_df['Total Paying']
    
    st.markdown("""
    ### 📈 Revenue per User Metrics           
    ```
    ARPU = Revenue / Total New Users (includes non-paying users)
    ```
    ```
    ARPPU = Revenue / Paying Users (paying user value)
    ```
    """)
    fig_arpu = go.Figure()
    fig_arpu.add_trace(go.Scatter(
        x=filtered_df['Created at'], 
        y=monthly_arpu, 
        name='ARPU',
        line=dict(color='#87CEEB', width=2)  # Light blue
    ))
    fig_arpu.add_trace(go.Scatter(
        x=filtered_df['Created at'], 
        y=monthly_arppu, 
        name='ARPPU',
        line=dict(color='#1E90FF', width=2)  # Dark blue
    ))
    fig_arpu.update_layout(
        title=f'{time_period} Revenue per User Trends',
        yaxis_title='Revenue per User ($)',
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    fig_arpu = format_date_axis(fig_arpu, selected_period)
    st.plotly_chart(fig_arpu, use_container_width=True)

with tab5:
    st.markdown("""
    ### 💵 Price Analysis & Trends
    
    Track average purchase prices and their evolution over time to understand our pricing effectiveness.
    """)
    # Calculate key metrics first
    filtered_df_revenue['Average Price'] = filtered_df_revenue['Total Revenue'] / filtered_df_revenue['Total Downloads']
    first_price = filtered_df_revenue['Average Price'].iloc[0]
    last_price = filtered_df_revenue['Average Price'].iloc[-1]
    price_change = ((last_price - first_price) / first_price * 100)
    
    st.markdown(f"""
    - Starting Price: ${first_price:.2f}
    - Current Price: ${last_price:.2f}
    - Total Price Change: {price_change:+.1f}%
    """)
    
    # Price trend chart
    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(
        x=filtered_df_revenue['Created at'],
        y=filtered_df_revenue['Average Price'],
        name='Average Price per Download',
        line=dict(color='#2E86C1', width=2)
    ))
    fig_price.update_layout(
        title='Average Price per Download Over Time',
        yaxis_title='Price ($)',
        showlegend=True,
        hovermode='x unified'
    )
    fig_price = format_date_axis(fig_price, selected_period)
    st.plotly_chart(fig_price, use_container_width=True)

    # 2. Growth Metrics Section
    # st.subheader("2. Growth Analysis")
    # col1, col2 = st.columns(2)
    
    # with col1:
    #     # CAGR calculations
    #     first_downloads = filtered_df_revenue['Total Downloads'].iloc[0]
    #     last_downloads = filtered_df_revenue['Total Downloads'].iloc[-1]
    #     n_periods = len(filtered_df_revenue) - 1
        
    #     if n_periods > 0 and first_downloads > 0:
    #         try:
    #             cagr_downloads = (((last_downloads / first_downloads) ** (1/n_periods)) - 1) * 100
    #             future_downloads = last_downloads * (1 + (cagr_downloads/100)) ** 5
                
    #             st.metric("Downloads CAGR", f"{cagr_downloads:.1f}%")
    #             st.metric("5Y Projection", f"{int(future_downloads):,}",
    #                      f"{(future_downloads/last_downloads):.1f}x current")
                
    #             st.markdown("""
    #             **What this means:**
    #             - CAGR = Compound Annual Growth Rate
    #             - Shows average yearly growth rate
    #             - Projection assumes same growth continues
    #             """)
    #         except:
    #             st.warning("Unable to calculate growth rates")
    
    # with col2:
    #     st.markdown("""
    #     **Growth Formulas:**
    #     ```
    #     CAGR = (Final/Initial)^(1/periods) - 1
        
    #     5Y Projection = Current * (1 + CAGR)^5
    #     ```
    #     """)
    
    st.markdown("""
    ### 📊 Detailed Performance Metrics
    
    The metrics below provide a comprehensive view of our unit economics:
    - **Downloads:** Number of successful purchases
    - **Revenue:** Total money received
    - **Avg Price:** Revenue per download
    - **Growth:** Period-over-period changes
    """)

    # Format and display metrics table
    if selected_period == 'Y':
        period_col = filtered_df_revenue['Created at'].dt.strftime('%Y')
    elif selected_period == 'Q':
        period_col = filtered_df_revenue['Created at'].dt.quarter.map(lambda x: f'Q{x}') + ' ' + filtered_df_revenue['Created at'].dt.year.astype(str)
    elif selected_period == 'D':
        period_col = filtered_df_revenue['Created at'].dt.strftime('%Y-%m-%d')
    else:  # 'M'
        period_col = filtered_df_revenue['Created at'].dt.strftime('%B %Y')

    unit_metrics = pd.DataFrame({
        'Period': period_col,
        'Downloads': filtered_df_revenue['Total Downloads'],
        'Revenue': filtered_df_revenue['Total Revenue'],
        'Avg Price': filtered_df_revenue['Average Price'],
        'Downloads Growth': filtered_df_revenue['Total Downloads'].pct_change() * 100,
        'Revenue Growth': filtered_df_revenue['Total Revenue'].pct_change() * 100,
        'Price Growth': filtered_df_revenue['Average Price'].pct_change() * 100
    }).set_index('Period')
    
    st.dataframe(
        unit_metrics.style.format({
            'Downloads': '{:,.0f}',
            'Revenue': '${:,.2f}',
            'Avg Price': '${:,.2f}',
            'Downloads Growth': '{:+.2f}%',
            'Revenue Growth': '{:+.2f}%',
            'Price Growth': '{:+.2f}%'
        }),
        use_container_width=True
    )

###############################
# REPLACE YOUR TAB 6 WITH THIS
###############################
with tab6:
    st.markdown("""
    ## 📈 Revenue & User Forecasts 

    This tab always uses **fixed monthly Revenue and User data from 2020 onward** to train the Prophet model, and will ignore any filtering or granularity selected in the sidebar.
    """)

    # 1) Helper Functions to Build Forecast and Actual Data from 2020+ (Ignoring User Filters)
    @st.cache_data
    def build_forecast_users_df():
        """
        Loads entire 'users' dataset from CSV,
        filters to 2020 onward,
        resamples monthly for Prophet (ds,y).
        (Ignores user filters)
        """
        users, _ = load_data()
        # Keep only 2020+
        users = users[users['Created at'] >= '2020-01-01'].copy()
        
        # Convert to datetime (no tz), set as index
        users['Created at'] = pd.to_datetime(users['Created at'], utc=True).dt.tz_localize(None)
        users.set_index('Created at', inplace=True)

        # Resample monthly and count user IDs
        monthly = users['Id'].resample('M').count().reset_index()
        monthly.columns = ['ds','y']
        return monthly

    @st.cache_data
    def build_forecast_revenue_df():
        """
        Loads entire 'purchases' dataset,
        filters to 2020 onward and direct payment types (InApp,Android,Stripe),
        resamples monthly for Prophet (ds,y).
        (Ignores user filters)
        """
        _, purchases = load_data()
        direct_types = ['InAppPurchase','AndroidPayment','StripePayment']

        # Filter 2020+ and direct types
        purchases = purchases[
            (purchases['Created at [Route Purchase]'] >= '2020-01-01') &
            (purchases['Type [Payment]'].isin(direct_types))
        ].copy()

        # Convert to datetime, set as index
        purchases['Created at [Route Purchase]'] = pd.to_datetime(
            purchases['Created at [Route Purchase]'], utc=True
        ).dt.tz_localize(None)
        purchases.set_index('Created at [Route Purchase]', inplace=True)

        # Resample monthly sum of Price
        monthly = purchases['Price [Route Purchase]'].resample('M').sum().reset_index()
        monthly.columns = ['ds','y']
        return monthly

    @st.cache_data
    def build_actual_users_df():
        """
        Similar to build_forecast_users_df but we rename the column to 'actual_users'
        for plotting. This is the "Actual" line on the chart (ignores user filters).
        """
        users, _ = load_data()
        users = users[users['Created at'] >= '2020-01-01'].copy()

        users['Created at'] = pd.to_datetime(users['Created at'], utc=True).dt.tz_localize(None)
        users.set_index('Created at', inplace=True)

        monthly = users['Id'].resample('M').count().reset_index()
        monthly.columns = ['ds','actual_users']
        return monthly

    @st.cache_data
    def build_actual_revenue_df():
        """
        Similar to build_forecast_revenue_df but we rename the column to 'actual_revenue'
        for plotting. This is the "Actual" line on the chart (ignores user filters).
        """
        _, purchases = load_data()
        direct_types = ['InAppPurchase','AndroidPayment','StripePayment']

        purchases = purchases[
            (purchases['Created at [Route Purchase]'] >= '2020-01-01') &
            (purchases['Type [Payment]'].isin(direct_types))
        ].copy()

        purchases['Created at [Route Purchase]'] = pd.to_datetime(
            purchases['Created at [Route Purchase]'], utc=True
        ).dt.tz_localize(None)
        purchases.set_index('Created at [Route Purchase]', inplace=True)

        monthly = purchases['Price [Route Purchase]'].resample('M').sum().reset_index()
        monthly.columns = ['ds','actual_revenue']
        return monthly

    # 2) Prophet Forecast Function with Fixed Hyperparams (Ignore User Filters)
    def create_fixed_prophet_forecast(df, forecast_periods=12):

        # Example hyperparameters:
        m = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode='multiplicative',
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=5
        )
        m.fit(df)
        future = m.make_future_dataframe(periods=forecast_periods, freq='M')
        forecast = m.predict(future)
        return forecast, m

    # 3) Build (A) Forecast Data and (B) Actual Data 
    forecast_users_df   = build_forecast_users_df()
    forecast_revenue_df = build_forecast_revenue_df()
    actual_users_df     = build_actual_users_df()
    actual_revenue_df   = build_actual_revenue_df()

    # 4) Let user pick forecast horizon
    forecast_periods = st.slider(
        "How many months into the future to forecast?",
        min_value=1, max_value=24, value=12,
        help="How many months into the future to forecast"
    )

    # 5) Create Forecasts
    users_forecast, _   = create_fixed_prophet_forecast(forecast_users_df, forecast_periods)
    revenue_forecast, _ = create_fixed_prophet_forecast(forecast_revenue_df, forecast_periods)

    # 6) Plot Revenue
    st.subheader("Direct-Revenue Forecast")
    fig_revenue = go.Figure()

    # Actual line in BLUE
    fig_revenue.add_trace(go.Scatter(
        x=actual_revenue_df['ds'],
        y=actual_revenue_df['actual_revenue'],
        mode='lines+markers',
        name='Actual Revenue',
        line=dict(color='blue', width=2)
    ))

    # Forecast line in RED dash
    fig_revenue.add_trace(go.Scatter(
        x=revenue_forecast['ds'],
        y=revenue_forecast['yhat'],
        mode='lines+markers',
        name='Forecasted Revenue',
        line=dict(color='red', dash='dash', width=2)
    ))

    # Confidence interval
    fig_revenue.add_trace(go.Scatter(
        x=revenue_forecast['ds'].tolist() + revenue_forecast['ds'][::-1].tolist(),
        y=revenue_forecast['yhat_upper'].tolist() + revenue_forecast['yhat_lower'][::-1].tolist(),
        fill='toself',
        fillcolor='rgba(255,0,0,0.1)',
        line=dict(color='rgba(255,0,0,0)'),
        name='Confidence'
    ))

    fig_revenue.update_layout(
        title='Revenue Forecast (Trained on data from 2020+)',
        xaxis_title='Date',
        yaxis_title='Revenue ($)',
        hovermode='x unified',
        height=500
    )
    st.plotly_chart(fig_revenue, use_container_width=True)

    # 7) Plot Users
    st.subheader("New Users Forecast")
    fig_users = go.Figure()

    # Actual line in GREEN
    fig_users.add_trace(go.Scatter(
        x=actual_users_df['ds'],
        y=actual_users_df['actual_users'],
        mode='lines+markers',
        name='Actual Users',
        line=dict(color='green', width=2)
    ))

    # Forecast line in RED dash
    fig_users.add_trace(go.Scatter(
        x=users_forecast['ds'],
        y=users_forecast['yhat'],
        mode='lines+markers',
        name='Forecasted Users',
        line=dict(color='red', dash='dash', width=2)
    ))

    # Confidence interval
    fig_users.add_trace(go.Scatter(
        x=users_forecast['ds'].tolist() + users_forecast['ds'][::-1].tolist(),
        y=users_forecast['yhat_upper'].tolist() + users_forecast['yhat_lower'][::-1].tolist(),
        fill='toself',
        fillcolor='rgba(255,0,0,0.1)',
        line=dict(color='rgba(255,0,0,0)'),
        name='Confidence'
    ))

    fig_users.update_layout(
        title='New Users Forecast (Trained on data from 2020+)',
        xaxis_title='Date',
        yaxis_title='Users',
        hovermode='x unified',
        height=500
    )
    st.plotly_chart(fig_users, use_container_width=True)

    # # Combine into a table
    # forecast_table = pd.DataFrame({
    #     'Date': users_forecast['ds'],
    #     'Users': users_forecast['yhat'].round(0),
    #     'Users_Lower': users_forecast['yhat_lower'].round(0),
    #     'Users_Upper': users_forecast['yhat_upper'].round(0),
    #     'Conv %': conversion_forecast['yhat'].round(2),
    #     'Conv Lower': conversion_forecast['yhat_lower'].round(2),
    #     'Conv Upper': conversion_forecast['yhat_upper'].round(2),
    #     'Repeat %': repeat_forecast['yhat'].round(2),
    #     'Repeat Lower': repeat_forecast['yhat_lower'].round(2),
    #     'Repeat Upper': repeat_forecast['yhat_upper'].round(2)
    # })

    # forecast_table['Date'] = forecast_table['Date'].dt.strftime('%b %Y')
    # st.dataframe(forecast_table.set_index('Date'), use_container_width=True)

    # 8) Combine into a "Forecast Summary Table" for both Revenue & Users
    st.subheader("Forecast Summary Table")

    # Create DataFrame for final output.
    # We'll align them by 'ds' (the date).
    # Both forecasts have the same ds range, so we can merge on ds.
    df_forecast = pd.DataFrame({
        'ds': revenue_forecast['ds'],
        'Revenue': revenue_forecast['yhat'].round(2),
        'Revenue Lower': revenue_forecast['yhat_lower'].round(2),
        'Revenue Upper': revenue_forecast['yhat_upper'].round(2),
        'New Users': users_forecast['yhat'].round(0),
        'New Users Lower': users_forecast['yhat_lower'].round(0),
        'New Users Upper': users_forecast['yhat_upper'].round(0)
    })

    # Format 'ds' as e.g. "Jan 2023"
    df_forecast['Date'] = df_forecast['ds'].dt.strftime('%b %Y')

    # Reorder columns: 'Date', 'Revenue', 'Users', ...
    df_forecast = df_forecast[['Date', 'Revenue', 'Revenue Lower', 'Revenue Upper', 
                               'New Users', 'New Users Lower', 'New Users Upper']]

    # Use set_index('Date') so Date is the row label
    st.dataframe(
        df_forecast.set_index('Date').style.format({
            'Revenue': '${:,.2f}',
            'Revenue Lower': '${:,.2f}',
            'Revenue Upper': '${:,.2f}',
            'New Users': '{:,.0f}',
            'New Users Lower': '{:,.0f}',
            'New Users Upper': '{:,.0f}'
        }),
        use_container_width=True
    )

##############################################
# TAB 7: LIFETIME VALUE (LTV) by USER TYPE
##############################################
with tab7:
    st.markdown("""
    ## 📈 User Lifetime Value (LTV) by User Type
    
    This chart shows how much revenue each user cohort generates over time, 
    **taking into account the date range and payment type filters** on the left.
    
    - **Months Since Sign-Up** (X-axis): The number of months from each user's sign-up date
    - **Average LTV** (Y-axis): The cumulative paid revenue per user (average within each cohort)
    
    **Cohorts**:
    - **New Paying**: Made at least one paid purchase *within the sign-up month*
    - **Returning Repeat**: Did not pay in the sign-up month, but eventually made more than 1 paid purchase 
    """)

    @st.cache_data
    def build_ltv_data_filtered(users_df, purchases_df):
        """
        1) Classify each user into: Free, New Paying, Returning First, Returning Repeat
        2) Compute 'months_since_signup' for each PAID purchase
        3) Calculate each user's cumulative paid spend by month_since_signup
        4) Average those cumulative spends across all users in each user_type & month_since_signup
        5) Exclude 'Returning First Purchase' from final results

        Returns a DataFrame with columns:
          [user_type, months_since_signup, avg_ltv]
        """
        ###################################################
        # A) PREP & MERGE
        ###################################################
        # Copy to avoid modifying original
        u = users_df.copy()
        p = purchases_df.copy()

        # Rename columns for clarity
        u.rename(columns={'Id': 'user_id'}, inplace=True)
        p.rename(columns={'Id [User]': 'user_id',
                          'Price [Route Purchase]': 'price'}, 
                 inplace=True)

        # Define which payment types count as 'paid'
        paid_types = [
            'AndroidPayment','BraintreePayment','CouponRedemptionCreditReseller',
            'CouponRedemptionReseller','CouponRedemptionResellerReply',
            'InAppPurchase','PaypalPayment','StripePayment'
        ]
        p['is_paid'] = p['Type [Payment]'].isin(paid_types)

        # Force datetime
        u['Created at'] = pd.to_datetime(u['Created at'], utc=True).dt.tz_localize(None)
        p['Created at [Route Purchase]'] = pd.to_datetime(p['Created at [Route Purchase]'], utc=True).dt.tz_localize(None)

        # Sign-up month (to check if user paid during that month)
        u['signup_month'] = u['Created at'].dt.to_period('M')

        ###################################################
        # B) CLASSIFY USERS
        ###################################################
        # For each user, count total paid purchases & how many in sign-up month
        merged = pd.merge(
            p[['user_id','is_paid','Created at [Route Purchase]']],
            u[['user_id','signup_month','Created at']], 
            on='user_id', how='right'
        )
        # Some users might have no purchase row -> fill is_paid as False
        merged['is_paid'] = merged['is_paid'].fillna(False)
        # Month of each purchase
        merged['purchase_month'] = merged['Created at [Route Purchase]'].dt.to_period('M')

        def user_summary(sub):
            paid_count = sub['is_paid'].sum()
            su_month   = sub['signup_month'].iloc[0]
            in_signup  = ((sub['is_paid'] == True) & (sub['purchase_month'] == su_month)).sum()
            return pd.Series({'total_paid_count':paid_count, 'signup_paid_count':in_signup})

        user_type_info = merged.groupby('user_id').apply(user_summary).reset_index()

        # Classify each user
        def classify_user(row):
            if row['total_paid_count'] == 0:
                return "Free User"
            elif row['signup_paid_count'] > 0:
                return "New Paying"
            elif row['total_paid_count'] == 1:
                return "Returning First"   # We'll omit from final chart
            else:
                return "Returning Repeat"

        user_type_info['user_type'] = user_type_info.apply(classify_user, axis=1)

        # Merge classification back into 'users'
        u = pd.merge(
            u, 
            user_type_info[['user_id','user_type']], 
            on='user_id', how='left'
        )
        u['user_type'] = u['user_type'].fillna("Free User")

        ###################################################
        # C) BUILD CUMULATIVE PAID REVENUE
        ###################################################
        # Only keep paid purchases
        paid_purchases = p[p['is_paid'] == True].copy()

        # Merge with user sign-up date + user_type
        merged_paid = pd.merge(
            paid_purchases, 
            u[['user_id','Created at','user_type']], 
            on='user_id', how='inner'
        )

        # months_since_signup
        merged_paid['months_since_signup'] = (
            (merged_paid['Created at [Route Purchase]'] - merged_paid['Created at'])
            / np.timedelta64(30,'D')
        ).apply(np.floor).astype(int)

        # Discard negative
        merged_paid = merged_paid[merged_paid['months_since_signup'] >= 0]

        # group by (user_id, user_type, months_since_signup), sum price
        user_monthly = merged_paid.groupby(
            ['user_id','user_type','months_since_signup'], as_index=False
        )['price'].sum()

        # cumsum across each user
        user_monthly.sort_values(['user_id','months_since_signup'], inplace=True)
        user_monthly['cumulative_spend'] = user_monthly.groupby('user_id')['price'].cumsum()

        # Average by (user_type, months_since_signup)
        final = user_monthly.groupby(['user_type','months_since_signup'], as_index=False)['cumulative_spend'].mean()
        final.rename(columns={'cumulative_spend':'avg_ltv'}, inplace=True)

        # Omit the "Returning First" user_type from final
        final = final[final['user_type'] != "Returning First"].copy()

        return final

    ###################################################
    # 1) Build LTV data from the *filtered* dataset
    ###################################################
    # We'll use the same row-level filtering logic your code uses:
    #  - 'filtered_df' is aggregated. We need the raw row-level user/purchase data
    #  - But we want to apply the same "time range + payment type" filters.
    # So let's re-run the row-level filtering with 'process_filtered_data', then do LTV.

    # We'll replicate the raw filtering: 
    # 'process_filtered_data' already gave us aggregated DF (filtered_df), 
    # but let's get the actual row-level (we do NOT have that in 'filtered_df').
    # We'll do it ourselves by using load_data() + the same filtering approach:
    # 
    # Actually, an easy approach: we *already* have 'users' and 'purchases' fully loaded,
    # but we want to restrict to the same date range & payment types:
    # 
    # We'll do exactly what 'process_filtered_data' does, but keep row-level.

    # Filter date range + payment types in the same way:
    date_start = pd.to_datetime(start_date).tz_localize(None)
    date_end   = pd.to_datetime(end_date).tz_localize(None)

    # 2) Subset 'users' to that date range
    users_filtered = users[
        (users['Created at'] >= date_start) &
        (users['Created at'] <= date_end)
    ].copy()

    # 3) Subset 'purchases' to only chosen payment types
    pmt_filtered = purchases[purchases['Type [Payment]'].isin(selected_payment_types)].copy()

    # 4) We'll keep only purchases that happened between date_start & date_end
    pmt_filtered = pmt_filtered[
        (pmt_filtered['Created at [Route Purchase]'] >= date_start) &
        (pmt_filtered['Created at [Route Purchase]'] <= date_end)
    ].copy()

    # Now we have row-level 'users_filtered' and 'purchases_filtered' that match the side filters
    ltv_data = build_ltv_data_filtered(users_filtered, pmt_filtered)

    ###################################################
    # 2) Plot the result
    ###################################################
    # We'll have lines for: ["Free User","New Paying","Returning Repeat"]
    # because we omitted "Returning First" from the final DataFrame
    fig_ltv = go.Figure()
    for user_type in ["Free User","New Paying","Returning Repeat"]:
        subset = ltv_data[ltv_data['user_type'] == user_type].copy()
        subset.sort_values('months_since_signup', inplace=True)
        if len(subset) == 0:
            continue
        fig_ltv.add_trace(go.Scatter(
            x=subset['months_since_signup'],
            y=subset['avg_ltv'],
            mode='lines+markers',
            name=user_type,
            marker=dict(size=6),
            line=dict(width=2)
        ))

    fig_ltv.update_layout(
        title="Average LTV by Months Since Sign Up (Filtered)",
        xaxis_title="Months Since Sign-Up",
        yaxis_title="Avg LTV ($)",
        hovermode='x unified',
        height=550,
        legend=dict(
            yanchor="top", y=0.99,
            xanchor="left", x=0.01,
            bordercolor="grey", borderwidth=1
        )
    )
    st.plotly_chart(fig_ltv, use_container_width=True)

    ###################################################
    # 3) Table preview
    ###################################################
    st.markdown("### LTV Table (Filtered)")

    # Pivot so each row is months_since_signup, columns are user_type
    pivoted = ltv_data.pivot(
        index='months_since_signup',
        columns='user_type',
        values='avg_ltv'
    ).fillna(0.0).round(2)

    st.dataframe(
        pivoted.style.format("${:,.2f}"),
        use_container_width=True
    )


# Detailed metrics table
st.subheader("Detailed Metrics")

for fig in [fig1, fig_payment_dist, fig3, fig5, fig_revenue, fig_rev_breakdown, fig_rev_pct, fig_arpu, fig_price]:
    fig.update_layout(**hover_settings)

# Create a copy for display
display_df = filtered_df.copy()
display_revenue_df = filtered_df_revenue.copy()

# Format the 'Created at' column based on the selected period
if selected_period == 'Y':
    date_format = '%Y'
    display_df['Created at'] = display_df['Created at'].dt.strftime(date_format)
    display_revenue_df['Created at'] = display_revenue_df['Created at'].dt.strftime(date_format)
elif selected_period == 'Q':
    display_df['Created at'] = filtered_df['Created at'].dt.quarter.map(lambda x: f'Q{x}') + ' ' + filtered_df['Created at'].dt.year.astype(str)
    display_revenue_df['Created at'] = filtered_df_revenue['Created at'].dt.quarter.map(lambda x: f'Q{x}') + ' ' + filtered_df_revenue['Created at'].dt.year.astype(str)
elif selected_period == 'D':
    date_format = '%Y-%m-%d'
    display_df['Created at'] = display_df['Created at'].dt.strftime(date_format)
    display_revenue_df['Created at'] = display_revenue_df['Created at'].dt.strftime(date_format)
else:  # 'M'
    date_format = '%B %Y'
    display_df['Created at'] = display_df['Created at'].dt.strftime(date_format)
    display_revenue_df['Created at'] = display_revenue_df['Created at'].dt.strftime(date_format)


# display_df['Created at'] = display_df['Created at'].dt.strftime(date_format)
# display_revenue_df['Created at'] = display_revenue_df['Created at'].dt.strftime(date_format)

display_df = display_df.set_index('Created at')
display_revenue_df = display_revenue_df.set_index('Created at')

# Add revenue columns to display_df
display_df['Total Revenue'] = display_revenue_df['Total Revenue']
display_df['Total Downloads'] = display_revenue_df['Total Downloads']
display_df['ARPU'] = display_revenue_df['Total Revenue'] / display_df['New Users']
display_df['ARPPU'] = display_revenue_df['Total Revenue'] / display_df['Total Paying']

# Reorder the columns
display_df = display_df[['New Users', 'Free Users', 'New Paying Users', 'Returning, First Purchase', 
                        'All New Paying', 'Cumulative All New Paying', 'All Returning', 
                        'Returning, Repeat', 'Total Paying', 'Percentage Returning, Repeat',
                        'Total Revenue', 'Total Downloads', 'ARPU', 'ARPPU']]

# Display the dataframe with index (which will be frozen)
st.dataframe(
    data=display_df.style.format({
        'New Users': '{:,.0f}',
        'Free Users': '{:,.0f}',
        'New Paying Users': '{:,.0f}',
        'Returning, First Purchase': '{:,.0f}',
        'All New Paying': '{:,.0f}',
        'Cumulative All New Paying': '{:,.0f}',
        'All Returning': '{:,.0f}',
        'Returning, Repeat': '{:,.0f}',
        'Total Paying': '{:,.0f}',
        'Percentage Returning, Repeat': '{:.1f}%',
        'Total Revenue': '${:,.2f}',
        'Total Downloads': '{:,.0f}',
        'ARPU': '${:,.2f}',
        'ARPPU': '${:,.2f}'
    }),
    use_container_width=True)