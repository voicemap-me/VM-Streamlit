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

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["User Acquisition", "Payment Metrics", "Conversion Analysis", "Revenue Metrics", "Unit Economics", "Forecasting"])

# Define all payment types
ALL_PAYMENT_TYPES = ['InAppPurchase', 'AndroidPayment', 'StripePayment', 'CouponRedemptionCreditReseller', 'FreePayment', 'CouponRedemptionReseller', 'CouponRedemptionResellerReply', 'CouponRedemptionPaid', 'CouponRedemption', 'SwfRedemption', 'BraintreePayment', 'PaypalPayment']
DEFAULT_PAYMENT_TYPES = ['InAppPurchase', 'AndroidPayment', 'StripePayment']

def create_prophet_forecast(data, column_name, periods=12):
    """
    Create a forecast using Facebook Prophet
    
    Parameters:
    data: DataFrame with 'Created at' column and the target column
    column_name: Name of the column to forecast
    periods: Number of periods to forecast
    
    Returns:
    DataFrame with the forecast and confidence intervals
    """
    df_prophet = pd.DataFrame({
        'ds': data['Created at'],
        'y': data[column_name]
    })
    
    # Configure seasonality based on granularity
    if selected_period == 'D':
        m = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode='multiplicative'
        )
    elif selected_period == 'M':
        m = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode='multiplicative'
        )
    else:  # 'Q' or 'Y'
        m = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode='additive'
        )
    
    m.fit(df_prophet)
    
    # Get the last date from actual data
    last_date = df_prophet['ds'].max()
    
    # Create future dataframe starting after the last actual date
    if selected_period == 'D':
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods, freq='D')
    elif selected_period == 'M':
        # Use 'ME' for date_range to get month-end dates
        future_dates = pd.date_range(start=last_date + pd.offsets.MonthEnd(1), periods=periods, freq='ME')
    elif selected_period == 'Q':
        future_dates = pd.date_range(start=last_date + pd.offsets.QuarterBegin(1), periods=periods, freq='Q')
    else:  # 'Y'
        future_dates = pd.date_range(start=last_date + pd.offsets.YearBegin(1), periods=periods, freq='Y')
    
    future = pd.DataFrame({'ds': future_dates})
    
    # Make forecast
    forecast = m.predict(future)
    
    # Combine with historical data for plotting
    historical = pd.DataFrame({
        'ds': df_prophet['ds'],
        'yhat': df_prophet['y'],
        'yhat_lower': df_prophet['y'],
        'yhat_upper': df_prophet['y']
    })
    
    forecast = pd.concat([historical, forecast])
    
    return forecast, m

def create_components_forecast(data, column_name):
    """
    Create a forecast with all seasonality components for visualization
    """
    df_prophet = pd.DataFrame({
        'ds': data['Created at'],
        'y': data[column_name]
    })
    
    # Configure Prophet with all seasonality components
    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
        seasonality_mode='multiplicative'
    )
    
    m.fit(df_prophet)
    
    # Create future dataframe for components
    future = m.make_future_dataframe(periods=0, freq='D')  # No future periods needed for components
    forecast = m.predict(future)
    
    return forecast, m

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
        start_date = fig.data[0].x[0]
        end_date = fig.data[0].x[-1]
        
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
    users = pd.read_csv("Users.csv")
    purchases = pd.read_csv("Purchase Data.csv")
    
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
            value=datetime(2020, 1, 1).date(),  # Set default to Jan 1, 2020
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

# Forecasting Tab
with tab6:
    st.markdown("""
    ## 📈 Forecasting Analysis
    
    This tab provides forecasts for key metrics using Facebook's Prophet model. The forecasts take into account:
    - Important Note: This model works best when filtering 2020 onwards and selecting monthly granularity
    - Historical trends
    - Yearly seasonality
    - Growth patterns
    
    Note: Forecasts are estimates and should be used as directional guidance rather than exact predictions.
    """)
    
    # Forecast period selector
    forecast_periods = st.slider("Number of periods to forecast", 
                               min_value=1, 
                               max_value=24, 
                               value=12,
                               help="Select how many periods ahead you want to forecast")
    
    # Revenue Forecast
    st.markdown("### Revenue Forecast")
    # Create revenue forecast
    revenue_forecast, revenue_model = create_prophet_forecast(filtered_df_revenue, 'Total Revenue', forecast_periods)
    
    # Plot revenue forecast
    fig_revenue_forecast = go.Figure()
    
    # Add actual values
    fig_revenue_forecast.add_trace(go.Scatter(
        x=filtered_df_revenue['Created at'],
        y=filtered_df_revenue['Total Revenue'],
        name='Actual Revenue',
        line=dict(color='blue', width=2)
    ))
    
    # Add forecasted values
    fig_revenue_forecast.add_trace(go.Scatter(
        x=revenue_forecast['ds'],
        y=revenue_forecast['yhat'],
        name='Forecast',
        line=dict(color='red', dash='dash', width=2)
    ))
    
    # Add confidence interval
    fig_revenue_forecast.add_trace(go.Scatter(
        x=revenue_forecast['ds'].tolist() + revenue_forecast['ds'].tolist()[::-1],
        y=revenue_forecast['yhat_upper'].tolist() + revenue_forecast['yhat_lower'].tolist()[::-1],
        fill='toself',
        fillcolor='rgba(255,0,0,0.1)',
        line=dict(color='rgba(255,0,0,0)'),
        name='Confidence Interval'
    ))
    
    fig_revenue_forecast.update_layout(
        title='Revenue Forecast',
        xaxis_title='Date',
        yaxis_title='Revenue ($)',
        hovermode='x unified',
        height=500,  # Increased height
        margin=dict(t=50, b=50, l=50, r=50)
    )
    fig_revenue_forecast = format_date_axis(fig_revenue_forecast, selected_period)
    st.plotly_chart(fig_revenue_forecast, use_container_width=True)
    
    # New Users Forecast
    st.markdown("### New Users Forecast")
    # Create users forecast
    users_forecast, users_model = create_prophet_forecast(filtered_df, 'New Users', forecast_periods)
    
    # Plot users forecast
    fig_users_forecast = go.Figure()
    
    # Add actual values
    fig_users_forecast.add_trace(go.Scatter(
        x=filtered_df['Created at'],
        y=filtered_df['New Users'],
        name='Actual Users',
        line=dict(color='green', width=2)
    ))
    
    # Add forecasted values
    fig_users_forecast.add_trace(go.Scatter(
        x=users_forecast['ds'],
        y=users_forecast['yhat'],
        name='Forecast',
        line=dict(color='red', dash='dash', width=2)
    ))
    
    # Add confidence interval
    fig_users_forecast.add_trace(go.Scatter(
        x=users_forecast['ds'].tolist() + users_forecast['ds'].tolist()[::-1],
        y=users_forecast['yhat_upper'].tolist() + users_forecast['yhat_lower'].tolist()[::-1],
        fill='toself',
        fillcolor='rgba(255,0,0,0.1)',
        line=dict(color='rgba(255,0,0,0)'),
        name='Confidence Interval'
    ))
    
    fig_users_forecast.update_layout(
        title='New Users Forecast',
        xaxis_title='Date',
        yaxis_title='Number of Users',
        hovermode='x unified',
        height=500,  # Increased height
        margin=dict(t=50, b=50, l=50, r=50)
    )
    fig_users_forecast = format_date_axis(fig_users_forecast, selected_period)
    st.plotly_chart(fig_users_forecast, use_container_width=True)
    
    # # Add forecast components visualization
    # st.markdown("### 📊 Forecast Components Analysis")
    # st.markdown("""
    # Understanding what drives our metrics across different time periods:
    # - **Trend**: The overall direction of growth
    # - **Yearly Seasonality**: Annual patterns and cycles
    # - **Monthly Seasonality**: Monthly variations
    # - **Weekly Seasonality**: Weekly patterns
    # - **Daily Seasonality**: Daily fluctuations
    # """)
    
    # col3, col4 = st.columns(2)
    
    # with col3:
    #     st.markdown("#### Revenue Components")
    #     # Create components forecast for revenue
    #     revenue_comp_forecast, revenue_comp_model = create_components_forecast(filtered_df_revenue, 'Total Revenue')
    #     fig_revenue_components = plot_components(revenue_comp_model, revenue_comp_forecast)
    #     st.pyplot(fig_revenue_components)
    #     plt.close()
    
    # with col4:
    #     st.markdown("#### Users Components")
    #     # Create components forecast for users
    #     users_comp_forecast, users_comp_model = create_components_forecast(filtered_df, 'New Users')
    #     fig_users_components = plot_components(users_comp_model, users_comp_forecast)
    #     st.pyplot(fig_users_components)
    #     plt.close()

    
    # Create forecasts for additional metrics
    conversion_df = pd.DataFrame({
        'Created at': filtered_df['Created at'],
        'Conversion Rate': (filtered_df['New Paying Users'] / filtered_df['New Users'] * 100)
    }).dropna()

    repeat_rate_df = pd.DataFrame({
        'Created at': filtered_df['Created at'],
        'Repeat Rate': (filtered_df['Returning, Repeat'].cumsum() / filtered_df['All New Paying'].cumsum() * 100)
    }).dropna()

    # Generate forecasts for each metric
    users_forecast, _ = create_prophet_forecast(filtered_df, 'New Users', forecast_periods)
    conv_forecast, _ = create_prophet_forecast(conversion_df, 'Conversion Rate', forecast_periods)
    repeat_forecast, _ = create_prophet_forecast(repeat_rate_df, 'Repeat Rate', forecast_periods)

    # Create combined forecast table
    forecast_table = pd.DataFrame({
        'Period': users_forecast['ds'],
        'Forecasted Users': users_forecast['yhat'].round(0),
        'Users Lower Bound': users_forecast['yhat_lower'].round(0),
        'Users Upper Bound': users_forecast['yhat_upper'].round(0),
        'Forecasted Conversion Rate (%)': conv_forecast['yhat'].round(2),
        'Conversion Rate Lower Bound (%)': conv_forecast['yhat_lower'].round(2),
        'Conversion Rate Upper Bound (%)': conv_forecast['yhat_upper'].round(2),
        'Forecasted Repeat Rate (%)': repeat_forecast['yhat'].round(2),
        'Repeat Rate Lower Bound (%)': repeat_forecast['yhat_lower'].round(2),
        'Repeat Rate Upper Bound (%)': repeat_forecast['yhat_upper'].round(2)
    })

    # Format the period column based on selected granularity
    if selected_period == 'Y':
        forecast_table['Period'] = forecast_table['Period'].dt.strftime('%Y')
    elif selected_period == 'Q':
        forecast_table['Period'] = forecast_table['Period'].dt.quarter.map(lambda x: f'Q{x}') + ' ' + forecast_table['Period'].dt.year.astype(str)
    elif selected_period == 'D':
        forecast_table['Period'] = forecast_table['Period'].dt.strftime('%Y-%m-%d')
    else:  # 'M'
        forecast_table['Period'] = forecast_table['Period'].dt.strftime('%B %Y')

    # Display the forecast table
    st.markdown("### 📊 Forecast Summary Table")
    st.markdown("""
    This table shows the forecasted values and their confidence intervals for:
    - Registered Users
    - Conversion Rate
    - Repeat Purchase Rate
    """)

    st.dataframe(
        forecast_table.set_index('Period').style.format({
            'Forecasted Users': '{:,.0f}',
            'Users Lower Bound': '{:,.0f}',
            'Users Upper Bound': '{:,.0f}',
            'Forecasted Conversion Rate (%)': '{:.2f}%',
            'Conversion Rate Lower Bound (%)': '{:.2f}%',
            'Conversion Rate Upper Bound (%)': '{:.2f}%',
            'Forecasted Repeat Rate (%)': '{:.2f}%',
            'Repeat Rate Lower Bound (%)': '{:.2f}%',
            'Repeat Rate Upper Bound (%)': '{:.2f}%'
        }),
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