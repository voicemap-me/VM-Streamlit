import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np

# Set page config
st.set_page_config(page_title="User Metrics Dashboard", layout="wide")

# Title and description
st.title("User Metrics Dashboard")
st.markdown("Analysis of user acquisition and purchase patterns")

# Define all payment types
ALL_PAYMENT_TYPES = ['InAppPurchase', 'AndroidPayment', 'StripePayment', 'CouponRedemptionCreditReseller', 'FreePayment', 'CouponRedemptionReseller', 'CouponRedemptionResellerReply', 'CouponRedemptionPaid', 'CouponRedemption', 'SwfRedemption', 'BraintreePayment', 'PaypalPayment']
DEFAULT_PAYMENT_TYPES = ['InAppPurchase', 'AndroidPayment', 'StripePayment']

def aggregate_by_period(df, period='M'):
    """
    Aggregate data by specified time period (Monthly, Quarterly, or Yearly)
    """
    temp_df = df.copy()
    
    if not pd.api.types.is_datetime64_any_dtype(temp_df['Created at']):
        temp_df['Created at'] = pd.to_datetime(temp_df['Created at'])
    
    if period == 'M':
        temp_df['Period'] = temp_df['Created at'].dt.to_period('M')
    elif period == 'Q':
        temp_df['Period'] = temp_df['Created at'].dt.to_period('Q')
    else:  # 'Y'
        temp_df['Period'] = temp_df['Created at'].dt.to_period('Y')
    
    # Aggregate numeric columns
    numeric_columns = temp_df.select_dtypes(include=[np.number]).columns
    agg_df = temp_df.groupby('Period').agg({col: 'sum' for col in numeric_columns}).reset_index()
    
    # Convert period back to timestamp for plotting
    agg_df['Created at'] = agg_df['Period'].apply(lambda x: x.to_timestamp())
    
    # Recalculate derived metrics if they exist
    if 'Percentage Returning, Repeat' in df.columns:
        agg_df['Percentage Returning, Repeat'] = (agg_df['Returning, Repeat'] / agg_df['Total Paying'] * 100).round(2)
    if 'Cumulative All New Paying' in df.columns:
        agg_df['Cumulative All New Paying'] = agg_df['All New Paying'].cumsum()
    
    return agg_df

def format_date_axis(fig, period):
    """Update date axis format based on selected period"""
    if period == 'Y':
        date_format = '%Y'
    elif period == 'Q':
        date_format = 'Q%q %Y'
    else:  # 'M'
        date_format = '%b %Y'
    
    fig.update_xaxes(
        tickformat=date_format,
        dtick='M2' if period == 'M' else 'M3' if period == 'Q' else 'M12'
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
    
    # Apply filters early
    users = users[(users['Created at'] >= start_date) & (users['Created at'] <= end_date)]
    purchases = purchases[
        (purchases['Created at [Route Purchase]'] >= start_date) & 
        (purchases['Created at [Route Purchase]'] <= end_date) &
        (purchases['Type [Payment]'].isin(payment_types_key))
    ]
    
    # Process metrics
    user_metrics = process_user_data(users, purchases, payment_types_key)
    revenue_metrics = process_revenue_data(users, purchases, payment_types_key)
    
    # Apply period aggregation
    if period != 'M':
        user_metrics = aggregate_by_period(user_metrics, period)
        revenue_metrics = aggregate_by_period(revenue_metrics, period)
    
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

def process_user_data(users, purchases, selected_payment_types):
    # Filter purchases based on payment types
    filtered_purchases = purchases[purchases['Type [Payment]'].isin(selected_payment_types)]
    
    # Calculate New Users for each month
    new_users = users.groupby(users['Created at'].dt.to_period('M')).size()
    
    # Get user ID ranges
    id_ranges = [get_user_id_range(users, month) for month in new_users.index]
    id_range_df = pd.DataFrame(id_ranges, columns=['Lowest ID', 'Highest ID'], index=new_users.index)
    
    def calculate_monthly_metrics(month):
        month_start = month.to_timestamp()
        month_end = month.to_timestamp(how='end')
        
        # New users for this month
        new_users_ids = users[users['Created at'].dt.to_period('M') == month]['Id']
        
        # New paying users (first month)
        new_paying = filtered_purchases[
            (filtered_purchases['Id [User]'].isin(new_users_ids)) & 
            (filtered_purchases['Created at [Route Purchase]'] >= month_start) & 
            (filtered_purchases['Created at [Route Purchase]'] <= month_end)
        ]['Id [User]'].nunique()
        
        # First-time purchasers this month
        first_time_purchasers = filtered_purchases[
            (filtered_purchases['Created at [Route Purchase]'] >= month_start) & 
            (filtered_purchases['Created at [Route Purchase]'] <= month_end) &
            (~filtered_purchases['Id [User]'].isin(
                filtered_purchases[filtered_purchases['Created at [Route Purchase]'] < month_start]['Id [User]']
            ))
        ]['Id [User]'].unique()
        
        # Returning first purchase
        highest_id_before_month = id_range_df.loc[:month - 1, 'Highest ID'].max()
        returning_first = sum(id <= highest_id_before_month for id in first_time_purchasers if pd.notnull(id))
        
        # All purchasers this month
        all_purchasers = filtered_purchases[
            (filtered_purchases['Created at [Route Purchase]'] >= month_start) & 
            (filtered_purchases['Created at [Route Purchase]'] <= month_end)
        ]['Id [User]'].unique()
        
        all_returning = len(set(all_purchasers) - set(new_users_ids))
        total_paying = len(all_purchasers)
        
        return pd.Series({
            'New Paying Users': new_paying,
            'Returning, First Purchase': returning_first,
            'All Returning': all_returning,
            'Total Paying': total_paying
        })
    
    # Calculate metrics for each month
    monthly_metrics = pd.DataFrame([calculate_monthly_metrics(month) for month in new_users.index],
                                 index=new_users.index)
    
    # Add new users to the dataframe
    monthly_metrics['New Users'] = new_users
    
    # Calculate derived metrics
    monthly_metrics['All New Paying'] = monthly_metrics['New Paying Users'] + monthly_metrics['Returning, First Purchase']
    monthly_metrics['Returning, Repeat'] = monthly_metrics['All Returning'] - monthly_metrics['Returning, First Purchase']
    monthly_metrics['Percentage Returning, Repeat'] = (monthly_metrics['Returning, Repeat'] / monthly_metrics['Total Paying'] * 100).round(2)
    monthly_metrics['Cumulative All New Paying'] = monthly_metrics['All New Paying'].cumsum()
    
    # Add ID ranges
    monthly_metrics['Lowest ID'] = id_range_df['Lowest ID']
    monthly_metrics['Highest ID'] = id_range_df['Highest ID']
    
    # Save the period index before resetting
    period_index = monthly_metrics.index
    
    # Convert to timestamps
    monthly_metrics = monthly_metrics.reset_index(drop=True)  # Drop the old index
    monthly_metrics['Created at'] = [period.to_timestamp() for period in period_index]
    
    return monthly_metrics

def process_revenue_data(users, purchases, selected_payment_types):
    filtered_purchases = purchases[purchases['Type [Payment]'].isin(selected_payment_types)]
    
    def calculate_monthly_revenue(month):
        month_start = month.to_timestamp()
        month_end = month.to_timestamp(how='end')
        
        # New users for this month
        new_users_ids = users[users['Created at'].dt.to_period('M') == month]['Id']
        
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
        
        # First-time purchasers this month
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
        
        # All revenue this month
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
st.sidebar.header("Data Filtering")

st.sidebar.caption("Payment filtering only works for Payment and Revenue related metrics")

# Add payment type filter
selected_payment_types = st.sidebar.multiselect(
    "Select Payment Types",
    options=ALL_PAYMENT_TYPES,
    default=DEFAULT_PAYMENT_TYPES
)

# Add time period selector to sidebar
time_period = st.sidebar.selectbox(
    "Select Date Granularity",
    options=['Monthly', 'Quarterly', 'Yearly'],
    index=0
)

# Convert selection to period code
period_map = {
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

# Get initial date range from raw data (we already have users loaded)
min_date = users['Created at'].min().date()
max_date = users['Created at'].max().date()

# Add date range filter
date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# Then process the filtered data
df, df_revenue = process_filtered_data(
    payment_types_key=tuple(sorted(selected_payment_types)),
    start_date=date_range[0],
    end_date=date_range[1],
    period=selected_period
)

filtered_df = df
filtered_df_revenue = df_revenue

# Top level metrics
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


# Main visualizations
st.subheader("User Growth Trends")
# tab1, tab2, tab3, tab4 = st.tabs(["User Acquisition", "Payment Metrics", "Conversion Analysis", "Revenue Metrics"])

tab1, tab2, tab3, tab4, tab5 = st.tabs(["User Acquisition", "Payment Metrics", "Conversion Analysis", "Revenue Metrics", "Unit Economics"])


with tab1:
    # User acquisition trends
    fig1 = go.Figure()
    fig1.add_trace(go.Bar(x=filtered_df['Created at'], y=filtered_df['New Users'],
                         name='New Users', marker_color='#2E86C1'))
    fig1.add_trace(go.Bar(x=filtered_df['Created at'], y=filtered_df['New Paying Users'],
                         name='New Paying Users', marker_color='#28B463'))
    fig1.update_layout(
        title=f'{time_period} New Users vs New Paying Users',
        xaxis_title='Date',
        yaxis_title='Number of Users',
        barmode='group'
    )
    fig1 = format_date_axis(fig1, selected_period)
    st.plotly_chart(fig1, use_container_width=True)

with tab2:
    # Payment metrics visualization
    fig2 = go.Figure()
    
    fig2.add_trace(go.Scatter(
        x=filtered_df['Created at'], 
        y=filtered_df['Cumulative All New Paying'],
        name='Cumulative New Paying Users',
        mode='lines',
        line=dict(width=3)
    ))
    
    fig2.add_trace(go.Bar(
        x=filtered_df['Created at'],
        y=filtered_df['All New Paying'],
        name='New Paying Users (Per Period)',
        marker_color='#AF7AC5',
        yaxis='y2'
    ))
    
    fig2.update_layout(
        title=f'{time_period} Payment Growth Trends',
        xaxis_title='Date',
        yaxis_title='Cumulative Users',
        yaxis2=dict(
            title=f'New Paying Users per {time_period}',
            overlaying='y',
            side='right'
        ),
        legend=dict(x=0.02, y=1.15, orientation='h')
    )
    fig2 = format_date_axis(fig2, selected_period)
    st.plotly_chart(fig2, use_container_width=True)
    
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
    # Add description
    st.markdown("""
    The conversion rate shows the percentage of new users who become paying users. It is calculated as:
    ```
    Conversion Rate = (All New Paying Users / Total New Users) × 100
    ```
    - **All New Paying Users** includes both users who paid in their first month and those who converted later
    - A higher conversion rate indicates more effective user monetization
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
        yaxis_range=[0, max(monthly_conversion) * 1.1]
    )
    fig5 = format_date_axis(fig5, selected_period)
    st.plotly_chart(fig5, use_container_width=True)

with tab4:
    # Revenue Growth
    fig_revenue = go.Figure()
    fig_revenue.add_trace(go.Bar(
        x=filtered_df_revenue['Created at'],
        y=filtered_df_revenue['Total Revenue'],
        name=f'{time_period} Revenue'
    ))
    fig_revenue.add_trace(go.Scatter(
        x=filtered_df_revenue['Created at'],
        y=filtered_df_revenue['Total Revenue'].cumsum(),
        name='Cumulative Revenue',
        yaxis='y2'
    ))
    fig_revenue.update_layout(
        title=f'{time_period} Revenue Growth',
        yaxis_title='Revenue per Period ($)',
        yaxis2=dict(title='Cumulative Revenue ($)', overlaying='y', side='right'),
        legend=dict(x=0.02, y=1.15, orientation='h')
    )
    fig_revenue = format_date_axis(fig_revenue, selected_period)
    st.plotly_chart(fig_revenue, use_container_width=True)

    # Revenue Breakdown
    fig_rev_breakdown = px.bar(filtered_df_revenue, x='Created at',
        y=['New Paying Users Revenue', 'Returning First Purchase Revenue', 'Returning Repeat Revenue'],
        title=f'{time_period} Revenue Breakdown by User Type',
        labels={'value': 'Revenue ($)', 'variable': 'Revenue Type'}
    )
    fig_rev_breakdown = format_date_axis(fig_rev_breakdown, selected_period)
    st.plotly_chart(fig_rev_breakdown, use_container_width=True)

    # Revenue per User Trend
    monthly_arpu = filtered_df_revenue['Total Revenue'] / filtered_df['New Users']
    monthly_arppu = filtered_df_revenue['Total Revenue'] / filtered_df['Total Paying']
    
    st.markdown("""
    ARPU and ARPPU is calculated as:
    ```
    ARPU = (Total Revenue / New Users) × 100
    ```
    ```
    ARPPU = (Total Revenue / All Paying Users) × 100
    ```
    """)
    fig_arpu = go.Figure()
    fig_arpu.add_trace(go.Scatter(x=filtered_df['Created at'], y=monthly_arpu, name='ARPU'))
    fig_arpu.add_trace(go.Scatter(x=filtered_df['Created at'], y=monthly_arppu, name='ARPPU'))
    fig_arpu.update_layout(
        title=f'{time_period} Revenue per User Trends',
        yaxis_title='Revenue per User ($)'
    )
    fig_arpu = format_date_axis(fig_arpu, selected_period)
    st.plotly_chart(fig_arpu, use_container_width=True)


with tab5:
    # st.header("Unit Economics Analysis")
    
    # 1. Overview Section
    st.subheader("1. Price Trends")
    # Calculate key metrics first
    filtered_df_revenue['Average Price'] = filtered_df_revenue['Total Revenue'] / filtered_df_revenue['Total Downloads']
    first_price = filtered_df_revenue['Average Price'].iloc[0]
    last_price = filtered_df_revenue['Average Price'].iloc[-1]
    price_change = ((last_price - first_price) / first_price * 100)
    
    st.markdown(f"""
    **Current Status:**
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
        showlegend=True
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
    
    # 3. Detailed Metrics Table
    st.subheader("2. Unit Metrics")
    st.markdown("""
    **Key Metrics Explained:**
    - **Downloads:** Number of purchases per period
    - **Revenue:** Total money received
    - **Avg Price:** Revenue ÷ Downloads
    - **Growth:** Percentage change from previous period
    """)

    # Format and display metrics table
    if selected_period == 'Y':
        period_col = filtered_df_revenue['Created at'].dt.strftime('%Y')
    elif selected_period == 'Q':
        period_col = filtered_df_revenue['Created at'].dt.quarter.map(lambda x: f'Q{x}') + ' ' + filtered_df_revenue['Created at'].dt.year.astype(str)
    else:
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

# Detailed metrics table
st.subheader("Detailed Metrics")

# Create a copy for display
display_df = filtered_df.copy()
display_revenue_df = filtered_df_revenue.copy()

# Format the 'Created at' column based on the selected period
if selected_period == 'Y':
    date_format = '%Y'
    display_df['Created at'] = display_df['Created at'].dt.strftime(date_format)
    display_revenue_df['Created at'] = display_revenue_df['Created at'].dt.strftime(date_format)
elif selected_period == 'Q':
    date_format = 'Q%q %Y'
    # Special handling for table display
    display_df['Created at'] = filtered_df['Created at'].dt.quarter.map(lambda x: f'Q{x}') + ' ' + filtered_df['Created at'].dt.year.astype(str)
    display_revenue_df['Created at'] = filtered_df_revenue['Created at'].dt.quarter.map(lambda x: f'Q{x}') + ' ' + filtered_df_revenue['Created at'].dt.year.astype(str)
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
display_df = display_df[['New Users', 'New Paying Users', 'Returning, First Purchase', 
                        'All New Paying', 'Cumulative All New Paying', 'All Returning', 
                        'Returning, Repeat', 'Total Paying', 'Percentage Returning, Repeat',
                        'Total Revenue', 'Total Downloads', 'ARPU', 'ARPPU']]

# Display the dataframe with index (which will be frozen)
st.dataframe(
    data=display_df.style.format({
        'New Users': '{:,.0f}',
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
                    
