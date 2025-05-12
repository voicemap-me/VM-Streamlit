# User & Revenue Analytics Dashboard (VM-Streamlit)

This repository contains a Streamlit data application designed for analyzing user acquisition, payment metrics, conversion rates, revenue trends, unit economics, forecasting, and Lifetime Value (LTV) based on user and purchase data.

The application relies on two core Python scripts:
1.  `datapipe.py`: A script to merge new monthly data with historical data.
2.  `dashboard.py`: The main Streamlit application script for visualization and analysis.

## Features

*   **Tabbed Interface:** Analysis is organized into logical sections:
    *   User Acquisition
    *   Payment Metrics
    *   Conversion Analysis
    *   Revenue Metrics
    *   Unit Economics
    *   Forecasting
    *   LTV (Lifetime Value)
*   **Interactive Filtering:**
    *   Filter data by **Payment Type** (multi-select with defaults and 'Select All').
    *   Adjust **Date Granularity** (Daily, Monthly, Quarterly, Yearly).
    *   Select specific **Date Ranges**.
*   **Key Metrics Display:** Top-level KPIs for a quick overview.
*   **Rich Visualizations:** Interactive charts powered by Plotly for exploring trends and distributions (bar charts, stacked bars, line charts, pie charts, LTV curves, forecast plots).
*   **Dynamic Aggregation:** Data is automatically aggregated based on the selected time granularity.
*   **Forecasting:** Uses Facebook Prophet to forecast Revenue and New Users (Note: This tab uses a fixed configuration, see [Forecasting Details](#forecasting-details)).
*   **LTV Analysis:** Calculates and visualizes average cumulative LTV based on user cohorts and time since signup (Note: This tab respects sidebar filters, see [LTV Details](#ltv-details)).
*   **Data Tables:** Displays underlying aggregated data for detailed inspection.

## Technologies Used

*   Python 3.x
*   Streamlit
*   Pandas
*   Plotly (Express & Graph Objects)
*   NumPy
*   Prophet

## Setup & Installation

**Method 1: Using the Setup Script (Recommended)**

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-directory>
    ```
2.  **Make the script executable (if needed):**
    ```bash
    chmod +x setup.sh
    ```
3.  **Run the setup script:**
    ```bash
    ./setup.sh
    ```
    This script will:
    *   Create a Python virtual environment named `venv`.
    *   Activate the environment (temporarily, for the script's execution).
    *   Upgrade pip.
    *   Install all dependencies listed in `requirements.txt`.
4.  **Activate the Virtual Environment for Use:** After the script finishes, you need to activate the environment in your current terminal session:
    ```bash
    source venv/bin/activate
    ```
    *(On Windows, use: `.\venv\Scripts\activate`)*

**Method 2: Manual Setup**

1.  **Clone the repository:** (As above)
2.  **Create a Virtual Environment:**
    ```bash
    python3 -m venv venv
    ```
3.  **Activate the Environment:**
    ```bash
    # Linux/macOS
    source venv/bin/activate
    # Windows
    .\venv\Scripts\activate
    ```
4.  **Install Dependencies:** Ensure you have a `requirements.txt` file listing the necessary libraries:
    ```
    streamlit
    pandas
    plotly
    numpy
    prophet
    matplotlib # Often a dependency for Prophet plots
    ```
    Then, install them:
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: Prophet installation can sometimes be tricky. Refer to the official [Prophet installation guide](https://facebook.github.io/prophet/docs/installation.html) if you encounter issues.)*

**After Setup (Both Methods):**

5.  **Place Initial Data Files:** Ensure the initial historical data files (e.g., `User Data 12 May 2025.csv` and `Purchase Data 12 May 2025.csv` as referenced initially in `dashboard.py`) are present in the root directory of the project.

6.  **Run the Dashboard:** Make sure your virtual environment (`venv`) is activated first.
    ```bash
    streamlit run dashboard.py
    ```

## Usage

Once the application is running (with the virtual environment activated):

1.  Use the **Sidebar** on the left to:
    *   Select the desired **Date Range**.
    *   Choose the **Date Granularity** (Daily, Monthly, Quarterly, Yearly).
    *   Filter by **Payment Types**.
2.  Navigate through the **Tabs** at the top to explore different aspects of the data analysis.
3.  **Hover** over charts to see specific data points and values.
4.  Scroll down to the bottom to view the **Detailed Metrics** table summarizing the filtered and aggregated data.

## Data Schema Expectations

The application expects the following minimal columns in the CSV files used by `dashboard.py`:

*   **`User Data <Date>.csv`:**
    *   `Id`: Unique identifier for the user.
    *   `Created at`: Timestamp of user signup/creation (datetime format).
*   **`Purchase Data <Date>.csv`:**
    *   `Id [User]`: Identifier linking the purchase to a user.
    *   `Created at [Route Purchase]`: Timestamp of the purchase (datetime format).
    *   `Type [Payment]`: Category/method of payment (string).
    *   `Price [Route Purchase]`: The value of the purchase (numeric).

*Important Note:* The `datapipe.py` script only keeps columns common to *both* the historical and new monthly purchase files when merging. Ensure your `route_purchase_<month>.csv` files contain at least these essential columns.

## Monthly Maintenance Workflow

This dashboard requires a **manual data update process** each month using the `datapipe.py` script.

**Prerequisites:** Obtain the new raw user and purchase data CSV files for the month that just ended (e.g., `user_may25.csv` and `route_purchase_may25.csv`).

**Steps:**

1.  **Update `datapipe.py` Input/Output Filenames:**
    *   **Line 6:** Change `pd.read_csv('Purchase Data <Previous Month Date>.csv')` to point to the *output* purchase file from the *previous* month's run.
    *   **Line 7:** Change `pd.read_csv('User Data <Previous Month Date>.csv')` to point to the *output* user file from the *previous* month's run.
    *   **Line 10:** Change `pd.read_csv('route_purchase_<previous_month_short>.csv')` to point to the *new* monthly purchase CSV file you just obtained (e.g., `route_purchase_may25.csv`).
    *   **Line 11:** Change `pd.read_csv('user_<previous_month_short>.csv')` to point to the *new* monthly user CSV file (e.g., `user_may25.csv`).
    *   **Line 34:** Update the output filename to reflect the *current* update date (e.g., `bigger_purchase.to_csv('Purchase Data <Current Month Date>.csv', index=False)`).
    *   **Line 35:** Update the output filename similarly (e.g., `bigger_user.to_csv('User Data <Current Month Date>.csv', index=False)`).

    *Example (Updating in June 2025 for May data):*
    ```python
    # Previous output files
    big_purchase = pd.read_csv('Purchase Data 12 May 2025.csv') # From last month's run
    big_user = pd.read_csv('User Data 12 May 2025.csv')     # From last month's run

    # New input files for May
    small_purchase = pd.read_csv('route_purchase_may25.csv') # New May data
    small_user = pd.read_csv('user_may25.csv')             # New May data

    # ... processing ...

    # New output files for June update
    bigger_purchase.to_csv('Purchase Data 12 June 2025.csv', index=False) # Updated filename
    bigger_user.to_csv('User Data 12 June 2025.csv', index=False)     # Updated filename
    ```

2.  **Run `datapipe.py`:**
    ```bash
    python datapipe.py
    ```
    Verify that the new merged CSV files (e.g., `Purchase Data <Current Month Date>.csv` and `User Data <Current Month Date>.csv`) are created in the project directory. Check for any errors during execution.

3.  **Update `dashboard.py` Data Loading:**
    *   Go to the `load_data` function (around line 230).
    *   Modify the `pd.read_csv()` calls to use the **newly generated** filenames from step 2.

    *Example (After running datapipe in June):*
    ```python
    @st.cache_data
    def load_data():
        users = pd.read_csv("User Data 12 June 2025.csv") # Use new user file
        purchases = pd.read_csv("Purchase Data 12 June 2025.csv") # Use new purchase file
        # ... rest of the function ...
    ```

4.  **Update `dashboard.py` Default Start Date (Optional but Recommended):**
    *   To prevent the default date range from becoming excessively large over time, consider updating the default `value` for the `start_date` input.
    *   Locate the sidebar date input section (around line 333).
    *   Modify the `value` parameter within `st.date_input("Start Date", ...)` to a more recent date (e.g., 12-24 months prior to the current `max_date`).

    *Example (Adjusting the default start date):*
    ```python
    # Around line 333 in dashboard.py
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input(
                "Start Date",
                # Update this value periodically
                value=datetime(2023, 6, 1).date(), # e.g., Changed default to June 1, 2023
                min_value=min_date,
                max_value=max_date
        )
    # ... rest of the sidebar code ...
    ```

5.  **Test the Dashboard Thoroughly:**
    *   Activate the virtual environment: `source venv/bin/activate`
    *   Run `streamlit run dashboard.py`.
    *   Ensure the application loads without errors.
    *   Verify the date range filters now include the latest month's data *and* that the default start date (if changed) is reflected.
    *   Check the most recent data points on key charts (e.g., revenue, new users) to ensure they look correct.
    *   Test various filter combinations (date granularity, payment types) to check for stability.

6.  **Commit Changes (if using Git):**
    *   Commit the changes made to `datapipe.py` and `dashboard.py`.
    *   Add the *new* data files (`User Data <Current Month Date>.csv`, `Purchase Data <Current Month Date>.csv`) to your commit if you intend to track the merged data in Git (be mindful of file size limits).

7.  **Deploy (If Applicable):** Follow your standard deployment procedure to update the live application.

## Forecasting Details (Tab 6)

*   The **Forecasting** tab uses a fixed configuration and **ignores** any filters selected in the sidebar (Date Range, Granularity, Payment Types).
*   It always trains the Prophet models on **monthly aggregated data starting from January 1st, 2020**.
*   It specifically uses only the following payment types for the **Revenue** forecast: `'InAppPurchase'`, `'AndroidPayment'`, `'StripePayment'`.
*   The **New Users** forecast uses *all* users from 2020 onward.
*   The forecast plots show the **lower bound** (`yhat_lower`) of the prediction interval as the main dashed forecast line, along with the confidence interval shading.

## LTV Details (Tab 7)

*   The **LTV (Lifetime Value)** tab **respects** the filters selected in the sidebar (Date Range, Payment Types).
*   The LTV is calculated based *only* on the users and paid purchases that fall within the selected date range and match the selected payment types.
*   It analyzes cohorts based on when users made their first *paid* purchase relative to their signup month.

## Potential Improvements

*   Automate the `datapipe.py` execution (e.g., using cron or a workflow tool).
*   Add more robust error handling and logging to `datapipe.py`.
*   Implement parameter tuning or alternative models for the Forecasting tab.
*   Add unit or integration tests for data processing functions.
