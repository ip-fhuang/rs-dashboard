import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ------------------------
# Streamlit setup
# ------------------------
st.set_page_config(page_title="SKU-Level Sales Analytics", layout="wide")
st.title("📊 Analytics Dashboard")

# Add analytics functions
# @st.cache_data
# def calculate_growth_rates(df, metric, group_col):
#     """Calculate month-over-month and year-over-year growth rates"""
#     growth_data = []
    
#     for group in df[group_col].unique():
#         group_data = df[df[group_col] == group].sort_values('Month')
#         group_data['MoM_Growth'] = group_data[metric].pct_change() * 100
        
#         # YoY Growth (if we have data from previous year)
#         group_data['Month_Key'] = group_data['Month'].dt.strftime('%m')
#         group_data['YoY_Growth'] = group_data.groupby('Month_Key')[metric].pct_change() * 100
        
#         growth_data.append(group_data)
    
#     return pd.concat(growth_data, ignore_index=True) if growth_data else pd.DataFrame()

@st.cache_data
def calculate_growth_rates(df, metric, group_col):
    """Calculate month-over-month and year-over-year growth rates
       based on aggregated SKU totals across all doors per month.
    """
    growth_data = []

    # # 1. Ensure Month is a datetime (if not already)
    # if not pd.api.types.is_datetime64_any_dtype(df['Month']):
    #     df['Month'] = pd.to_datetime(df['Month'], format='%Y%m')

    # 2. Aggregate sales by Month + SKU
    df_agg = (
        df.groupby(['Month', group_col])[metric]
        .sum()
        .reset_index()
        .sort_values(['Month'])
    )

    # 3. Calculate growth for each SKU
    for group in df_agg[group_col].unique():
        group_data = df_agg[df_agg[group_col] == group].copy()

        # MoM growth
        group_data['MoM_Growth'] = group_data[metric].pct_change() * 100

        # Add Month_Key for YoY growth (same month across years)
        group_data['Month_Key'] = group_data['Month'].dt.strftime('%m')
        group_data['YoY_Growth'] = (
            group_data.groupby('Month_Key')[metric].pct_change() * 100
        )

        growth_data.append(group_data)

    return pd.concat(growth_data, ignore_index=True) if growth_data else pd.DataFrame()

@st.cache_data
def perform_abc_analysis(df, metric, group_col):
    """Perform ABC analysis on sales data"""
    grouped = df.groupby(group_col)[metric].sum().sort_values(ascending=False)
    total = grouped.sum()
    
    if total == 0:
        return pd.DataFrame()
    
    cumulative_pct = (grouped.cumsum() / total) * 100
    
    abc_categories = []
    for pct in cumulative_pct:
        if pct <= 80:
            abc_categories.append('A')
        elif pct <= 95:
            abc_categories.append('B')
        else:
            abc_categories.append('C')
    
    abc_df = pd.DataFrame({
        group_col: grouped.index,
        metric: grouped.values,
        'Cumulative_Pct': cumulative_pct.values,
        'ABC_Category': abc_categories
    })
    
    return abc_df

@st.cache_data
def calculate_seasonality_index(df, metric, group_col):
    """Calculate seasonality index for each group"""
    seasonality_data = []
    
    for group in df[group_col].unique():
        group_data = df[df[group_col] == group].copy()
        group_data['Month_Num'] = group_data['Month'].dt.month
        
        monthly_avg = group_data.groupby('Month_Num')[metric].mean()
        overall_avg = group_data[metric].mean()
        
        if overall_avg > 0:
            seasonality_index = (monthly_avg / overall_avg) * 100
            
            for month, index in seasonality_index.items():
                seasonality_data.append({
                    group_col: group,
                    'Month_Num': month,
                    'Month_Name': pd.to_datetime(f'2024-{month:02d}-01').strftime('%B'),
                    'Seasonality_Index': index
                })
    
    return pd.DataFrame(seasonality_data)

@st.cache_data
def detect_outliers(df, metric, group_col):
    """Detect outliers using IQR method"""
    outliers = []
    
    for group in df[group_col].unique():
        group_data = df[df[group_col] == group][metric]
        if len(group_data) > 3:  # Need at least 4 data points
            Q1 = group_data.quantile(0.25)
            Q3 = group_data.quantile(0.75)
            IQR = Q3 - Q1
            
            if IQR > 0:
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                group_outliers = df[(df[group_col] == group) & 
                                   ((df[metric] < lower_bound) | (df[metric] > upper_bound))]
                
                if not group_outliers.empty:
                    outliers.append(group_outliers)
    
    return pd.concat(outliers, ignore_index=True) if outliers else pd.DataFrame()

@st.cache_data
def calculate_sku_performance_metrics(df):
    """Calculate comprehensive SKU performance metrics"""
    # First calculate active months correctly by counting unique months per SKU
    active_months_per_sku = df.groupby('SKU')['Month'].nunique().reset_index()
    active_months_per_sku.columns = ['SKU', 'Active_Months']
    
    sku_metrics = df.groupby('SKU').agg({
        'AYP Gross Sales (USD)': ['sum', 'mean'],
        'AYP Units': ['sum', 'mean'],
        'Revenue_per_Unit': ['mean', 'std'],
        'Month': ['min', 'max'],
        'Country': 'nunique',
        'Region': 'nunique',
        'Store_Column': 'nunique',  # Will be mapped to appropriate store column
        'Brand': 'first' if 'Brand' in df.columns else lambda x: 'N/A',
        'Brand Line': 'first' if 'Brand Line' in df.columns else lambda x: 'N/A',
        'Family': 'first' if 'Family' in df.columns else lambda x: 'N/A'
    }).round(2)
    
    # Flatten column names
    sku_metrics.columns = [f'{col[0]}_{col[1]}' if col[1] != '' else col[0] for col in sku_metrics.columns]
    sku_metrics = sku_metrics.reset_index()
    
    # Merge with active months
    sku_metrics = sku_metrics.merge(active_months_per_sku, on='SKU', how='left')
    
    # Calculate additional metrics
    sku_metrics['Total_Sales'] = sku_metrics['AYP Gross Sales (USD)_sum']
    sku_metrics['Total_Units'] = sku_metrics['AYP Units_sum']
    sku_metrics['Avg_Monthly_Sales'] = sku_metrics['AYP Gross Sales (USD)_mean']
    sku_metrics['Sales_Consistency'] = 1 / (sku_metrics['Revenue_per_Unit_std'].fillna(1) + 0.01)  # Higher = more consistent
    sku_metrics['Geographic_Reach'] = sku_metrics['Country_nunique']
    sku_metrics['Store_Penetration'] = sku_metrics['Store_Column_nunique']
    
    # Calculate date range
    sku_metrics['Date_Range_Days'] = (pd.to_datetime(sku_metrics['Month_max']) - pd.to_datetime(sku_metrics['Month_min'])).dt.days + 1
    
    return sku_metrics

def apply_dataframe_filters(df, filter_config):
    """Apply multiple filters to dataframe"""
    filtered_df = df.copy()
    
    for column, filter_values in filter_config.items():
        if filter_values and column in filtered_df.columns:
            if isinstance(filter_values, (list, tuple)):
                if len(filter_values) > 0:
                    filtered_df = filtered_df[filtered_df[column].isin(filter_values)]
            else:
                filtered_df = filtered_df[filtered_df[column] == filter_values]
    
    return filtered_df

def detect_column_structure(df):
    """Detect and map column structure to handle different data formats"""
    column_mapping = {}
    
    # Store column mapping - handle different store column names
    store_columns = ['Global Store Desc', 'Door/Store', 'Store', 'Retailer']
    brand_columns = ['Global Brand Desc', 'Brand']
    
    for col in store_columns:
        if col in df.columns:
            column_mapping['store_column'] = col
            break
    
    for col in brand_columns:
        if col in df.columns:
            column_mapping['brand_column'] = col
            break
    
    # Set defaults if not found
    if 'store_column' not in column_mapping:
        column_mapping['store_column'] = None
    if 'brand_column' not in column_mapping:
        column_mapping['brand_column'] = None
    
    return column_mapping

# Advanced filtering function
def create_advanced_filters(df, column_mapping):
    """Create advanced filtering interface"""
    st.sidebar.markdown("### 🎯 Advanced Filters")
    
    filter_config = {}
    
    # Multi-select filters
    if 'Region' in df.columns:
        regions = st.sidebar.multiselect("Regions:", sorted(df["Region"].unique()), 
                                        default=sorted(df["Region"].unique())[:3])
        filter_config['Region'] = regions
    
    if 'Country' in df.columns and filter_config.get('Region'):
        available_countries = df[df["Region"].isin(filter_config['Region'])]["Country"].unique()
        countries = st.sidebar.multiselect("Countries:", sorted(available_countries), 
                                          default=sorted(available_countries)[:5])
        filter_config['Country'] = countries
    
    # Brand filtering - use detected brand column
    brand_col = column_mapping.get('brand_column')
    if brand_col and brand_col in df.columns:
        brands = st.sidebar.multiselect(f"{brand_col}:", sorted(df[brand_col].unique()),
                                       default=sorted(df[brand_col].unique())[:5])
        filter_config[brand_col] = brands
    
    # Brand Line filtering (distinct from Family)
    if 'Brand Line' in df.columns:
        brand_lines = st.sidebar.multiselect("Brand Lines:", sorted(df["Brand Line"].unique()),
                                            default=sorted(df["Brand Line"].unique())[:5])
        filter_config['Brand Line'] = brand_lines
    
    # Family filtering (distinct from Brand Line)
    if 'Family' in df.columns:
        families = st.sidebar.multiselect("Families:", sorted(df["Family"].unique()),
                                         default=sorted(df["Family"].unique())[:5])
        filter_config['Family'] = families
    
    # Retailer filtering
    if 'Retailer' in df.columns:
        retailers = st.sidebar.multiselect("Retailers:", sorted(df["Retailer"].unique()),
                                          default=sorted(df["Retailer"].unique())[:3])
        filter_config['Retailer'] = retailers
    
    # SKU filtering with search
    if 'SKU' in df.columns:
        sku_search = st.sidebar.text_input("Search SKUs (comma-separated):")
        if sku_search:
            sku_list = [sku.strip() for sku in sku_search.split(',')]
            matching_skus = [sku for sku in df['SKU'].unique() if any(search_term.lower() in str(sku).lower() for search_term in sku_list)]
            filter_config['SKU'] = matching_skus
        else:
            # Show top SKUs by sales for selection
            top_skus_by_sales = df.groupby('SKU')['AYP Gross Sales (USD)'].sum().nlargest(20).index.tolist()
            selected_skus = st.sidebar.multiselect("Select SKUs:", sorted(df['SKU'].unique()),
                                                  default=top_skus_by_sales[:10])
            filter_config['SKU'] = selected_skus
    
    # Numeric range filters
    if 'Revenue_per_Unit' in df.columns:
        rev_range = st.sidebar.slider("Revenue per Unit Range:", 
                                     float(df['Revenue_per_Unit'].min()),
                                     float(df['Revenue_per_Unit'].max()),
                                     (float(df['Revenue_per_Unit'].min()), 
                                      float(df['Revenue_per_Unit'].max())),key="revPerUnit")
        # Apply numeric filter directly to dataframe in main function
    
    # Sales range filter
    if 'AYP Gross Sales (USD)' in df.columns:
        sales_range = st.sidebar.slider("Sales Range (USD):",
                                       float(df['AYP Gross Sales (USD)'].min()),
                                       float(df['AYP Gross Sales (USD)'].max()),
                                       (float(df['AYP Gross Sales (USD)'].min()),
                                        float(df['AYP Gross Sales (USD)'].max())),key="GrossSalesinUSD")
        # Apply numeric filter directly to dataframe in main function
    
    return filter_config

# File upload
uploaded_file = st.file_uploader("Upload merged sales file with SKU data (CSV or Excel)", type=["csv", "xlsx"])

if uploaded_file is not None:
    # ------------------------
    # Load and prepare data
    # ------------------------
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file, engine="openpyxl")

    # Check for required columns
    required_columns = ["Month", "AYP Gross Sales (USD)", "AYP Units"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.error(f"Missing required columns: {missing_columns}")
        st.stop()
    
    # Detect column structure
    column_mapping = detect_column_structure(df)
    
    # Create SKU column if it doesn't exist
    if 'SKU' not in df.columns:
        if 'Product Code' in df.columns:
            df['SKU'] = df['Product Code']
        elif column_mapping.get('brand_column'):
            df['SKU'] = df[column_mapping['brand_column']] + "_" + df.index.astype(str)
            st.warning("SKU column not found. Created SKU based on Brand + Index. Please ensure your data has a proper SKU column.")
        else:
            df['SKU'] = "SKU_" + df.index.astype(str)
            st.warning("SKU column not found. Created generic SKU identifiers. Please ensure your data has a proper SKU column.")

    # Handle different Month formats
    try:
        # First try YYYYMM format
        df["Month"] = pd.to_datetime(df["Month"].astype(str), format="%Y%m")
    except:
        try:
            # Try YYYY-MM format
            df["Month"] = pd.to_datetime(df["Month"].astype(str), format="%Y-%m")
        except:
            try:
                # Try general date parsing
                df["Month"] = pd.to_datetime(df["Month"])
            except:
                st.error("Could not parse Month column. Please ensure it's in YYYYMM, YYYY-MM, or standard date format.")
                st.stop()

    df["Month_Year"] = df["Month"].dt.strftime("%b %Y")
    df["Quarter"] = df["Month"].dt.quarter
    df["Year"] = df["Month"].dt.year
    df["Month_Name"] = df["Month"].dt.strftime("%B")

    # Calculate derived metrics
    df["Revenue_per_Unit"] = df["AYP Gross Sales (USD)"] / df["AYP Units"].replace(0, np.nan)
    df["Revenue_per_Unit"] = df["Revenue_per_Unit"].fillna(0)
    
    # Create Store_Column mapping for metrics calculation
    store_col = column_mapping.get('store_column')
    if store_col:
        df['Store_Column'] = df[store_col]
    else:
        df['Store_Column'] = 'Default_Store'

    # Enhanced sidebar with data overview
    st.sidebar.markdown("### 📈 Data Overview")
    st.sidebar.write(f"**Total Records:** {len(df):,}")
    st.sidebar.write(f"**Date Range:** {df['Month'].min().strftime('%b %Y')} - {df['Month'].max().strftime('%b %Y')}")
    st.sidebar.write(f"**Unique SKUs:** {df['SKU'].nunique():,}")
    
    # Dynamic column display based on what's available
    if 'Country' in df.columns:
        st.sidebar.write(f"**Countries:** {df['Country'].nunique()}")
    if 'Region' in df.columns:
        st.sidebar.write(f"**Regions:** {df['Region'].nunique()}")
    if column_mapping.get('brand_column'):
        st.sidebar.write(f"**Brands:** {df[column_mapping['brand_column']].nunique()}")
    if 'Brand Line' in df.columns:
        st.sidebar.write(f"**Brand Lines:** {df['Brand Line'].nunique()}")
    if 'Family' in df.columns:
        st.sidebar.write(f"**Families:** {df['Family'].nunique()}")
    if 'Retailer' in df.columns:
        st.sidebar.write(f"**Retailers:** {df['Retailer'].nunique()}")
    
    # Quick insights
    total_sales = df["AYP Gross Sales (USD)"].sum()
    total_units = df["AYP Units"].sum()
    st.sidebar.metric("Total Sales (USD)", f"${total_sales:,.0f}")
    st.sidebar.metric("Total Units", f"{total_units:,.0f}")
    st.sidebar.metric("Avg Revenue/Unit", f"${df['Revenue_per_Unit'].mean():.2f}")

    # Date range filter (always available)
    st.sidebar.markdown("### 📅 Date Range")
    min_date = df["Month"].min()
    max_date = df["Month"].max()
    date_range = st.sidebar.date_input(
        "Select Date Range:",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # Apply date filter
    if len(date_range) == 2:
        start_date, end_date = date_range
        df_filtered = df[(df["Month"] >= pd.to_datetime(start_date)) & 
                        (df["Month"] <= pd.to_datetime(end_date))]
    else:
        df_filtered = df

    # Create advanced filters
    filter_config = create_advanced_filters(df_filtered, column_mapping)
    
    # Apply filters
    df_filtered = apply_dataframe_filters(df_filtered, filter_config)
    
    # Apply numeric range filters
    if 'Revenue_per_Unit' in df_filtered.columns:
        rev_min, rev_max = st.sidebar.slider("Revenue per Unit Range:", 
                                            float(df['Revenue_per_Unit'].min()),
                                            float(df['Revenue_per_Unit'].max()),
                                            (float(df['Revenue_per_Unit'].min()), 
                                             float(df['Revenue_per_Unit'].max())),key="revPerUnitFiltered")
        df_filtered = df_filtered[(df_filtered['Revenue_per_Unit'] >= rev_min) & 
                                 (df_filtered['Revenue_per_Unit'] <= rev_max)]
    
    if 'AYP Gross Sales (USD)' in df_filtered.columns:
        sales_min, sales_max = st.sidebar.slider("Sales Range (USD):",
                                                float(df['AYP Gross Sales (USD)'].min()),
                                                float(df['AYP Gross Sales (USD)'].max()),
                                                (float(df['AYP Gross Sales (USD)'].min()),
                                                 float(df['AYP Gross Sales (USD)'].max())),key="AYPGrossSalesFiltered")
        df_filtered = df_filtered[(df_filtered['AYP Gross Sales (USD)'] >= sales_min) & 
                                 (df_filtered['AYP Gross Sales (USD)'] <= sales_max)]

    # Update Month_Year after filtering
    df_filtered["Month_Year"] = df_filtered["Month"].dt.strftime("%b %Y")
    
    # Show filtering results
    if len(df_filtered) == 0:
        st.warning("No data matches the current filters. Please adjust your selection.")
        st.stop()
    
    st.sidebar.markdown("### 📊 Filtered Data Summary")
    st.sidebar.write(f"**Filtered Records:** {len(df_filtered):,} ({len(df_filtered)/len(df)*100:.1f}% of total)")
    st.sidebar.write(f"**Filtered SKUs:** {df_filtered['SKU'].nunique():,}")
    
    # Primary metric selection
    metric = st.sidebar.radio("Select primary metric:", ["AYP Gross Sales (USD)", "AYP Units", "Revenue_per_Unit"])
    
    # Analysis type selection with SKU-focused options
    analysis_type = st.sidebar.selectbox(
        "Select Analysis Type:",
        ["SKU Performance Dashboard", "SKU Growth Analysis", "SKU ABC Analysis", 
         "SKU Seasonality Analysis", "SKU Comparison", "SKU Outlier Detection", 
         "SKU Portfolio Analysis", "Executive Summary", "Regional Deep Dive",
         "Brand Analysis", "Brand Line Analysis", "Family Analysis", "Retailer Analysis",
         "Hierarchical Analysis (Brand → Line → Family → SKU)"]
    )

    # ------------------------
    # SKU Performance Dashboard
    # ------------------------
    if analysis_type == "SKU Performance Dashboard":
        st.header("📦 SKU Performance Dashboard")
        
        # Calculate SKU performance metrics
        sku_metrics = calculate_sku_performance_metrics(df_filtered)
        
        # Top-level KPIs
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Active SKUs", f"{len(sku_metrics):,}")
        col2.metric("Top SKU Sales", f"${sku_metrics['Total_Sales'].max():,.0f}")
        col3.metric("Avg SKU Sales", f"${sku_metrics['Total_Sales'].mean():,.0f}")
        col4.metric("SKUs >$10K", f"{len(sku_metrics[sku_metrics['Total_Sales'] > 10000]):,}")
        col5.metric("Multi-Region SKUs", f"{len(sku_metrics[sku_metrics['Geographic_Reach'] > 1]):,}")
        
        # Top performing SKUs
        st.subheader("🏆 Top 20 SKUs by Sales")
        
        # Prepare columns to include Brand, Line, Family
        display_columns = ['SKU', 'Total_Sales', 'Total_Units', 'Revenue_per_Unit_mean', 
                          'Geographic_Reach', 'Store_Penetration', 'Active_Months']
        
        # Add Brand, Line, Family columns if they exist
        if 'Brand' in sku_metrics.columns:
            display_columns.insert(1, 'Brand')
        if 'Brand Line' in sku_metrics.columns:
            display_columns.insert(2 if 'Brand' in display_columns else 1, 'Brand Line')
        if 'Family' in sku_metrics.columns:
            display_columns.insert(3 if 'Brand Line' in display_columns else (2 if 'Brand' in display_columns else 1), 'Family')
        
        top_skus = sku_metrics.nlargest(20, 'Total_Sales')[display_columns].round(2)
        
        # Rename columns for display
        column_rename = {
            'SKU': 'SKU', 'Total_Sales': 'Total Sales', 'Total_Units': 'Total Units', 
            'Revenue_per_Unit_mean': 'Avg Rev/Unit', 'Geographic_Reach': 'Countries', 
            'Store_Penetration': 'Stores', 'Active_Months': 'Active Months',
            'Brand': 'Brand', 'Brand Line': 'Brand Line', 'Family': 'Family'
        }
        
        top_skus = top_skus.rename(columns=column_rename)
        st.dataframe(top_skus.reset_index(drop=True), use_container_width=True, hide_index=True)
        
        # SKU performance distribution
        col1, col2 = st.columns(2)
        
        with col1:
            fig_dist = px.histogram(sku_metrics, x='Total_Sales', nbins=50,
                                  title="SKU Sales Distribution")
            fig_dist.update_layout(xaxis_title="Sales (USD)", yaxis_title="Number of SKUs")
            st.plotly_chart(fig_dist, use_container_width=True)
        
        with col2:
            fig_scatter = px.scatter(sku_metrics, x='Total_Units', y='Total_Sales',
                                   hover_data=['SKU'], title="Units vs Sales by SKU",
                                   trendline="ols")
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        # SKU lifecycle analysis
        st.subheader("📅 SKU Lifecycle Analysis")
        lifecycle_data = df_filtered.groupby(['SKU', 'Month_Year']).agg({
            'AYP Gross Sales (USD)': 'sum'
        }).reset_index()
        
        # Select top 10 SKUs for lifecycle visualization
        top_10_skus = sku_metrics.nlargest(10, 'Total_Sales')['SKU'].tolist()
        lifecycle_filtered = lifecycle_data[lifecycle_data['SKU'].isin(top_10_skus)]
        
        fig_lifecycle = px.line(lifecycle_filtered, x='Month_Year', y='AYP Gross Sales (USD)',
                              color='SKU', title="Top 10 SKUs Sales Trend Over Time")
        fig_lifecycle.update_xaxes(tickangle=45)
        st.plotly_chart(fig_lifecycle, use_container_width=True)

    # ------------------------
    # Brand Analysis (New)
    # ------------------------
    # ------------------------
    # Brand Line Analysis (New)
    # ------------------------
    elif analysis_type == "Brand Line Analysis":
        st.header("📏 Brand Line Performance Analysis")
        
        if 'Brand Line' not in df_filtered.columns:
            st.error("No Brand Line column found in the data. This analysis requires brand line information.")
        else:
            # Brand Line performance metrics
            line_metrics = df_filtered.groupby('Brand Line').agg({
                'AYP Gross Sales (USD)': ['sum', 'mean'],
                'AYP Units': 'sum',
                'SKU': 'nunique',
                'Revenue_per_Unit': 'mean',
                'Country': 'nunique' if 'Country' in df_filtered.columns else lambda x: 1,
                'Month': 'nunique',
                'Brand': 'first' if 'Brand' in df_filtered.columns else lambda x: 'N/A',
                'Family': 'nunique' if 'Family' in df_filtered.columns else lambda x: 1
            }).round(2)
            
            line_metrics.columns = ['Total_Sales', 'Avg_Sales', 'Total_Units', 
                                   'SKU_Count', 'Avg_RevPerUnit', 'Countries', 
                                   'Active_Months', 'Brand', 'Family_Count']
            line_metrics = line_metrics.sort_values('Total_Sales', ascending=False)
            
            # Brand Line KPIs
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Brand Lines", line_metrics.shape[0])
            col2.metric("Top Line Sales", f"${line_metrics['Total_Sales'].max():,.0f}")
            col3.metric("Avg Line Sales", f"${line_metrics['Total_Sales'].mean():,.0f}")
            col4.metric("Total SKUs", line_metrics['SKU_Count'].sum())
            
            # Brand Line performance table
            st.subheader("📊 Brand Line Performance Summary")
            display_metrics = line_metrics.reset_index()
            st.dataframe(display_metrics, use_container_width=True, hide_index=True)
            
            # Brand Line visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Brand Line sales bar chart
                fig_line_sales = px.bar(line_metrics.reset_index().head(15), 
                                       x='Brand Line', y='Total_Sales',
                                       color='Brand',
                                       title="Top 15 Brand Lines by Sales")
                fig_line_sales.update_xaxes(tickangle=45)
                st.plotly_chart(fig_line_sales, use_container_width=True)
            
            with col2:
                # SKU count vs Sales scatter
                fig_sku_sales = px.scatter(line_metrics.reset_index(), 
                                         x='SKU_Count', y='Total_Sales',
                                         color='Brand',
                                         hover_data=['Brand Line'],
                                         title="SKU Count vs Brand Line Sales")
                st.plotly_chart(fig_sku_sales, use_container_width=True)
            
            # Brand Line trend analysis
            st.subheader("📈 Brand Line Trends Over Time")
            line_trends = df_filtered.groupby(['Brand Line', 'Month_Year'])['AYP Gross Sales (USD)'].sum().reset_index()
            top_lines = line_metrics.head(5).index.tolist()
            line_trends_filtered = line_trends[line_trends['Brand Line'].isin(top_lines)]
            
            fig_line_trends = px.line(line_trends_filtered, x='Month_Year', y='AYP Gross Sales (USD)',
                                     color='Brand Line', title="Top 5 Brand Line Sales Trends")
            fig_line_trends.update_xaxes(tickangle=45)
            st.plotly_chart(fig_line_trends, use_container_width=True)

    # ------------------------
    # Family Analysis (New)
    # ------------------------
    elif analysis_type == "Family Analysis":
        st.header("👨‍👩‍👧‍👦 Family Performance Analysis")
        
        if 'Family' not in df_filtered.columns:
            st.error("No Family column found in the data. This analysis requires family information.")
        else:
            # Family performance metrics
            family_metrics = df_filtered.groupby('Family').agg({
                'AYP Gross Sales (USD)': ['sum', 'mean'],
                'AYP Units': 'sum',
                'SKU': 'nunique',
                'Revenue_per_Unit': 'mean',
                'Country': 'nunique' if 'Country' in df_filtered.columns else lambda x: 1,
                'Month': 'nunique',
                'Brand': 'nunique' if 'Brand' in df_filtered.columns else lambda x: 1,
                'Brand Line': 'nunique' if 'Brand Line' in df_filtered.columns else lambda x: 1
            }).round(2)
            
            family_metrics.columns = ['Total_Sales', 'Avg_Sales', 'Total_Units', 
                                     'SKU_Count', 'Avg_RevPerUnit', 'Countries', 
                                     'Active_Months', 'Brand_Count', 'Line_Count']
            family_metrics = family_metrics.sort_values('Total_Sales', ascending=False)
            
            # Family KPIs
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Families", family_metrics.shape[0])
            col2.metric("Top Family Sales", f"${family_metrics['Total_Sales'].max():,.0f}")
            col3.metric("Avg Family Sales", f"${family_metrics['Total_Sales'].mean():,.0f}")
            col4.metric("Total SKUs", family_metrics['SKU_Count'].sum())
            
            # Family performance table
            st.subheader("📊 Family Performance Summary")
            display_metrics = family_metrics.reset_index()
            st.dataframe(display_metrics, use_container_width=True, hide_index=True)
            
            # Family visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Family sales treemap
                fig_family_treemap = px.treemap(family_metrics.reset_index().head(20), 
                                               path=['Family'], values='Total_Sales',
                                               title="Top 20 Families by Sales")
                st.plotly_chart(fig_family_treemap, use_container_width=True)
            
            with col2:
                # Family efficiency (Sales per SKU)
                family_metrics['Sales_per_SKU'] = family_metrics['Total_Sales'] / family_metrics['SKU_Count']
                fig_efficiency = px.bar(family_metrics.reset_index().head(10), 
                                      x='Family', y='Sales_per_SKU',
                                      title="Sales Efficiency per SKU by Family")
                fig_efficiency.update_xaxes(tickangle=45)
                st.plotly_chart(fig_efficiency, use_container_width=True)
            
            # Family trend analysis
            st.subheader("📈 Family Trends Over Time")
            family_trends = df_filtered.groupby(['Family', 'Month_Year'])['AYP Gross Sales (USD)'].sum().reset_index()
            top_families = family_metrics.head(5).index.tolist()
            family_trends_filtered = family_trends[family_trends['Family'].isin(top_families)]
            
            fig_family_trends = px.line(family_trends_filtered, x='Month_Year', y='AYP Gross Sales (USD)',
                                       color='Family', title="Top 5 Family Sales Trends")
            fig_family_trends.update_xaxes(tickangle=45)
            st.plotly_chart(fig_family_trends, use_container_width=True)

    # ------------------------
    # Hierarchical Analysis (New)
    # ------------------------
    elif analysis_type == "Hierarchical Analysis (Brand → Line → Family → SKU)":
        st.header("🏗️ Hierarchical Analysis: Brand → Line → Family → SKU")
        
        required_cols = ['Brand', 'Brand Line', 'Family', 'SKU']
        missing_cols = [col for col in required_cols if col not in df_filtered.columns]
        
        if missing_cols:
            st.error(f"Missing required columns for hierarchical analysis: {missing_cols}")
        else:
            # Hierarchical performance metrics
            st.subheader("📊 Hierarchy Overview")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Brands", df_filtered['Brand'].nunique())
            col2.metric("Brand Lines", df_filtered['Brand Line'].nunique()) 
            col3.metric("Families", df_filtered['Family'].nunique())
            col4.metric("SKUs", df_filtered['SKU'].nunique())
            
            # Interactive drill-down
            st.subheader("🔍 Interactive Drill-Down Analysis")
            
            # Level 1: Brand selection
            selected_brand = st.selectbox("Select Brand:", ['All'] + sorted(df_filtered['Brand'].unique()))
            
            if selected_brand != 'All':
                brand_data = df_filtered[df_filtered['Brand'] == selected_brand]
                
                # Level 2: Brand Line selection
                available_lines = sorted(brand_data['Brand Line'].unique())
                selected_line = st.selectbox("Select Brand Line:", ['All'] + available_lines)
                
                if selected_line != 'All':
                    line_data = brand_data[brand_data['Brand Line'] == selected_line]
                    
                    # Level 3: Family selection
                    available_families = sorted(line_data['Family'].unique())
                    selected_family = st.selectbox("Select Family:", ['All'] + available_families)
                    
                    if selected_family != 'All':
                        family_data = line_data[line_data['Family'] == selected_family]
                        analysis_data = family_data
                        analysis_level = 'SKU'
                        st.write(f"**Analysis Level:** SKU within {selected_brand} → {selected_line} → {selected_family}")
                    else:
                        analysis_data = line_data
                        analysis_level = 'Family'
                        st.write(f"**Analysis Level:** Family within {selected_brand} → {selected_line}")
                else:
                    analysis_data = brand_data
                    analysis_level = 'Brand Line'
                    st.write(f"**Analysis Level:** Brand Line within {selected_brand}")
            else:
                analysis_data = df_filtered
                analysis_level = 'Brand'
                st.write("**Analysis Level:** Brand")
            
            # Performance analysis at selected level
            if analysis_level == 'Brand':
                group_col = 'Brand'
            elif analysis_level == 'Brand Line':
                group_col = 'Brand Line'
            elif analysis_level == 'Family':
                group_col = 'Family'
            else:  # SKU
                group_col = 'SKU'
            
            # Calculate performance metrics for current level
            level_performance = analysis_data.groupby(group_col).agg({
                'AYP Gross Sales (USD)': 'sum',
                'AYP Units': 'sum',
                'Revenue_per_Unit': 'mean'
            }).round(2)
            
            level_performance = level_performance.sort_values('AYP Gross Sales (USD)', ascending=False)
            level_performance.columns = ['Total Sales', 'Total Units', 'Avg Rev/Unit']
            
            # Display performance table
            st.subheader(f"📈 {analysis_level} Performance")
            st.dataframe(level_performance.reset_index(), use_container_width=True, hide_index=True)
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Performance bar chart
                fig_perf = px.bar(level_performance.reset_index().head(15), 
                                 x=group_col, y='Total Sales',
                                 title=f"Top 15 {analysis_level} by Sales")
                fig_perf.update_xaxes(tickangle=45)
                st.plotly_chart(fig_perf, use_container_width=True)
            
            with col2:
                # Performance pie chart
                top_10_data = level_performance.head(10)
                fig_pie = px.pie(values=top_10_data['Total Sales'], 
                               names=top_10_data.index,
                               title=f"Top 10 {analysis_level} Sales Distribution")
                st.plotly_chart(fig_pie, use_container_width=True)
            
            # Hierarchical sunburst chart
            st.subheader("☀️ Hierarchical Sunburst Visualization")
            
            # Hierarchical sunburst chart
            st.subheader("☀️ Hierarchical Sunburst Visualization")
            
            sunburst_data = df_filtered.groupby(['Brand', 'Brand Line', 'Family'])['AYP Gross Sales (USD)'].sum().reset_index()
            
            fig_sunburst = px.sunburst(sunburst_data, 
                                     path=['Brand', 'Brand Line', 'Family'], 
                                     values='AYP Gross Sales (USD)',
                                     title="Hierarchical Sales Distribution (Brand → Line → Family)")
            fig_sunburst.update_layout(height=600)
            st.plotly_chart(fig_sunburst, use_container_width=True)
            
            # Cross-level insights
            st.subheader("💡 Cross-Level Insights")
            
            insights = []
            
            # Brand concentration
            brand_concentration = df_filtered.groupby('Brand')['AYP Gross Sales (USD)'].sum()
            top_brand = brand_concentration.idxmax()
            brand_share = (brand_concentration.max() / brand_concentration.sum()) * 100
            insights.append(f"**Top Brand:** {top_brand} accounts for {brand_share:.1f}% of total sales")
            
            # SKU diversity by brand
            sku_by_brand = df_filtered.groupby('Brand')['SKU'].nunique().sort_values(ascending=False)
            most_diverse_brand = sku_by_brand.index[0]
            insights.append(f"**Most Diverse Brand:** {most_diverse_brand} with {sku_by_brand.iloc[0]} SKUs")
            
            # Best performing family
            family_performance = df_filtered.groupby('Family')['AYP Gross Sales (USD)'].sum()
            top_family = family_performance.idxmax()
            insights.append(f"**Top Family:** {top_family} with ${family_performance.max():,.0f} in sales")
            
            for insight in insights:
                st.write(insight)

    
    
    elif analysis_type == "Brand Analysis":
        st.header("🏷️ Brand Performance Analysis")
        
        brand_col = column_mapping.get('brand_column')
        if not brand_col:
            st.error("No brand column found in the data. This analysis requires brand information.")
        else:
            # Brand performance metrics
            brand_metrics = df_filtered.groupby(brand_col).agg({
                'AYP Gross Sales (USD)': ['sum', 'mean'],
                'AYP Units': 'sum',
                'SKU': 'nunique',
                'Revenue_per_Unit': 'mean',
                'Country': 'nunique' if 'Country' in df_filtered.columns else lambda x: 1,
                'Month': 'nunique'
            }).round(2)
            
            brand_metrics.columns = ['Total_Sales', 'Avg_Sales', 'Total_Units', 
                                   'SKU_Count', 'Avg_RevPerUnit', 'Countries', 'Active_Months']
            brand_metrics = brand_metrics.sort_values('Total_Sales', ascending=False)
            
            # Brand KPIs
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Brands", brand_metrics.shape[0])
            col2.metric("Top Brand Sales", f"${brand_metrics['Total_Sales'].max():,.0f}")
            col3.metric("Avg Brand Sales", f"${brand_metrics['Total_Sales'].mean():,.0f}")
            col4.metric("Total SKUs", brand_metrics['SKU_Count'].sum())
            
            # Brand performance table
            st.subheader("📊 Brand Performance Summary")
            st.dataframe(brand_metrics.reset_index(), use_container_width=True, hide_index=True)
            
            # Brand visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Brand sales bar chart
                fig_brand_sales = px.bar(brand_metrics.reset_index().head(15), 
                                       x=brand_col, y='Total_Sales',
                                       title="Top 15 Brands by Sales")
                fig_brand_sales.update_xaxes(tickangle=45)
                st.plotly_chart(fig_brand_sales, use_container_width=True)
            
            with col2:
                # SKU count vs Sales scatter
                fig_sku_sales = px.scatter(brand_metrics.reset_index(), 
                                         x='SKU_Count', y='Total_Sales',
                                         hover_data=[brand_col],
                                         title="SKU Count vs Brand Sales")
                st.plotly_chart(fig_sku_sales, use_container_width=True)
            
            # Brand trend analysis
            st.subheader("📈 Brand Trends Over Time")
            brand_trends = df_filtered.groupby([brand_col, 'Month_Year'])['AYP Gross Sales (USD)'].sum().reset_index()
            top_brands = brand_metrics.head(5).index.tolist()
            brand_trends_filtered = brand_trends[brand_trends[brand_col].isin(top_brands)]
            
            fig_brand_trends = px.line(brand_trends_filtered, x='Month_Year', y='AYP Gross Sales (USD)',
                                     color=brand_col, title="Top 5 Brand Sales Trends")
            fig_brand_trends.update_xaxes(tickangle=45)
            st.plotly_chart(fig_brand_trends, use_container_width=True)

    # ------------------------
    # Retailer Analysis (New)
    # ------------------------
    elif analysis_type == "Retailer Analysis":
        st.header("🏪 Retailer Performance Analysis")
        
        if 'Retailer' not in df_filtered.columns:
            st.error("No Retailer column found in the data. This analysis requires retailer information.")
        else:
            # Retailer performance metrics
            retailer_metrics = df_filtered.groupby('Retailer').agg({
                'AYP Gross Sales (USD)': ['sum', 'mean'],
                'AYP Units': 'sum',
                'SKU': 'nunique',
                'Revenue_per_Unit': 'mean',
                'Country': 'nunique' if 'Country' in df_filtered.columns else lambda x: 1,
                'Month': 'nunique'
            }).round(2)
            
            retailer_metrics.columns = ['Total_Sales', 'Avg_Sales', 'Total_Units', 
                                      'SKU_Count', 'Avg_RevPerUnit', 'Countries', 'Active_Months']
            retailer_metrics = retailer_metrics.sort_values('Total_Sales', ascending=False)
            
            # Retailer KPIs
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Retailers", retailer_metrics.shape[0])
            col2.metric("Top Retailer Sales", f"${retailer_metrics['Total_Sales'].max():,.0f}")
            col3.metric("Avg Retailer Sales", f"${retailer_metrics['Total_Sales'].mean():,.0f}")
            col4.metric("Total SKUs", retailer_metrics['SKU_Count'].sum())
            
            # Retailer performance table
            st.subheader("📊 Retailer Performance Summary")
            st.dataframe(retailer_metrics.reset_index(), use_container_width=True, hide_index=True)
            
            # Retailer visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Retailer sales pie chart
                fig_retailer_pie = px.pie(retailer_metrics.reset_index(), 
                                        values='Total_Sales', names='Retailer',
                                        title="Sales Distribution by Retailer")
                st.plotly_chart(fig_retailer_pie, use_container_width=True)
            
            with col2:
                # Retailer efficiency (Sales per SKU)
                retailer_metrics['Sales_per_SKU'] = retailer_metrics['Total_Sales'] / retailer_metrics['SKU_Count']
                fig_efficiency = px.bar(retailer_metrics.reset_index().head(10), 
                                      x='Retailer', y='Sales_per_SKU',
                                      title="Sales Efficiency per SKU by Retailer")
                fig_efficiency.update_xaxes(tickangle=45)
                st.plotly_chart(fig_efficiency, use_container_width=True)

    # ------------------------
    # SKU Growth Analysis
    # ------------------------
    elif analysis_type == "SKU Growth Analysis":
        st.header("📈 SKU Growth Analysis")
        
        # Calculate growth rates for SKUs
        df_filtered = df_filtered.groupby(['Month', 'SKU'])[metric].sum().reset_index()
        growth_df = calculate_growth_rates(df_filtered, metric, 'SKU')
        #growth_df = calculate_growth_rates(df_filtered, metric, 'SKU')
        
        if not growth_df.empty:
            # Current period growth
            latest_month = growth_df['Month'].max()
            latest_growth = growth_df[growth_df['Month'] == latest_month]
            
            col1, col2, col3 = st.columns(3)
            positive_growth = len(latest_growth[latest_growth['MoM_Growth'] > 0])
            negative_growth = len(latest_growth[latest_growth['MoM_Growth'] < 0])
            avg_growth = latest_growth['MoM_Growth'].mean()
            
            col1.metric("SKUs with Positive Growth", positive_growth)
            col2.metric("SKUs with Negative Growth", negative_growth)
            col3.metric("Average Growth Rate", f"{avg_growth:.1f}%")
            
            # Top growth performers
            st.subheader("🚀 Top Growth Performers (Latest Month)")
            top_growth = latest_growth.nlargest(20, 'MoM_Growth')[
                ['SKU', 'MoM_Growth', metric]
            ].round(2)
            top_growth.columns = ['SKU', 'MoM Growth %', 'Current Sales']
            st.dataframe(top_growth.reset_index(drop=True), use_container_width=True, hide_index=True)
            
            # Growth distribution
            col1, col2 = st.columns(2)
            
            with col1:
                fig_growth_dist = px.histogram(latest_growth, x='MoM_Growth', nbins=30,
                                             title="Growth Rate Distribution (Latest Month)")
                fig_growth_dist.add_vline(x=0, line_dash="dash", line_color="red")
                st.plotly_chart(fig_growth_dist, use_container_width=True)
            
            with col2:
                # Growth vs Size scatter
                fig_growth_scatter = px.scatter(latest_growth, x=metric, y='MoM_Growth',
                                              hover_data=['SKU'], 
                                              title="Growth Rate vs Current Performance")
                fig_growth_scatter.add_hline(y=0, line_dash="dash", line_color="red")
                st.plotly_chart(fig_growth_scatter, use_container_width=True)
            
            # Growth trend over time for selected SKUs
            st.subheader("📊 Growth Trend Analysis")
            selected_skus_growth = st.multiselect("Select SKUs for growth trend:", 
                                                 growth_df['SKU'].unique(),
                                                 default=list(growth_df['SKU'].unique())[:5])
            
            if selected_skus_growth:
                growth_trend = growth_df[growth_df['SKU'].isin(selected_skus_growth)]
                fig_trend = px.line(growth_trend, x='Month', y='MoM_Growth', 
                                  color='SKU', title="MoM Growth Trend by SKU")
                fig_trend.add_hline(y=0, line_dash="dash", line_color="red")
                fig_trend.update_xaxes(tickangle=45)
                st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.warning("Insufficient data for growth analysis. Need at least 2 time periods per SKU.")

    # ------------------------
    # SKU ABC Analysis
    # ------------------------
    elif analysis_type == "SKU ABC Analysis":
        st.header("🎯 SKU ABC Analysis")
        
        abc_results = perform_abc_analysis(df_filtered, metric, 'SKU')
        
        if not abc_results.empty:
            # ABC Summary
            abc_summary = abc_results.groupby('ABC_Category').agg({
                'SKU': 'count',
                metric: 'sum'
            }).rename(columns={'SKU': 'SKU Count'})
            abc_summary['Sales %'] = (abc_summary[metric] / abc_summary[metric].sum() * 100).round(1)
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.write("**ABC Category Summary**")
                st.dataframe(abc_summary)
            
            with col2:
                # ABC pie chart
                fig_pie = px.pie(abc_summary.reset_index(), values=metric, names='ABC_Category',
                               title=f"Sales Distribution by ABC Category")
                st.plotly_chart(fig_pie, use_container_width=True)
            
            # Pareto chart
            fig_pareto = go.Figure()
            
            # Limit to top 100 SKUs for readability
            abc_top = abc_results.head(100)
            
            fig_pareto.add_trace(go.Bar(x=abc_top['SKU'], y=abc_top[metric],
                                      name=metric, yaxis='y'))
            
            fig_pareto.add_trace(go.Scatter(x=abc_top['SKU'], y=abc_top['Cumulative_Pct'],
                                          mode='lines+markers', name='Cumulative %', 
                                          yaxis='y2', line=dict(color='red')))
            
            fig_pareto.update_layout(
                title=f'ABC Analysis - Top 100 SKUs by {metric}',
                yaxis=dict(title=metric, side='left'),
                yaxis2=dict(title='Cumulative %', side='right', overlaying='y'),
                xaxis_tickangle=45,
                height=500
            )
            
            st.plotly_chart(fig_pareto, use_container_width=True)
            
            # Category details
            st.subheader("📋 ABC Category Details")
            category_filter = st.selectbox("Select ABC Category:", ['All'] + list(abc_results['ABC_Category'].unique()))
            
            if category_filter == 'All':
                display_results = abc_results
            else:
                display_results = abc_results[abc_results['ABC_Category'] == category_filter]
            
            st.dataframe(display_results.sort_values(metric, ascending=False).reset_index(drop=True), use_container_width=True, hide_index=True)
        else:
            st.warning("Insufficient data for ABC analysis.")

    # ------------------------
    # SKU Seasonality Analysis
    # ------------------------
    elif analysis_type == "SKU Seasonality Analysis":
        st.header("🌊 SKU Seasonality Analysis")
        
        seasonality_results = calculate_seasonality_index(df_filtered, metric, 'SKU')
        
        if not seasonality_results.empty:
            # Select SKUs for seasonality analysis
            top_skus_for_seasonality = df_filtered.groupby('SKU')[metric].sum().nlargest(20).index.tolist()
            selected_skus_seasonality = st.multiselect("Select SKUs for seasonality analysis:",
                                                      seasonality_results['SKU'].unique(),
                                                      default=top_skus_for_seasonality[:10])
            
            if selected_skus_seasonality:
                filtered_seasonality = seasonality_results[seasonality_results['SKU'].isin(selected_skus_seasonality)]
                
                # Seasonality heatmap
                pivot_seasonality = filtered_seasonality.pivot(index='SKU', 
                                                             columns='Month_Name', 
                                                             values='Seasonality_Index')
                
                # Reorder columns to show months in chronological order
                month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                              'July', 'August', 'September', 'October', 'November', 'December']
                pivot_seasonality = pivot_seasonality.reindex(columns=[m for m in month_order if m in pivot_seasonality.columns])
                
                fig_heatmap = px.imshow(pivot_seasonality, 
                                      title=f'SKU Seasonality Heatmap - {metric}',
                                      color_continuous_scale='RdBu_r',
                                      aspect='auto')
                fig_heatmap.update_layout(height=400)
                st.plotly_chart(fig_heatmap, use_container_width=True)
                
                # Peak and low seasons
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Peak Seasons by SKU**")
                    peak_seasons = filtered_seasonality.loc[filtered_seasonality.groupby('SKU')['Seasonality_Index'].idxmax()]
                    peak_display = peak_seasons[['SKU', 'Month_Name', 'Seasonality_Index']].sort_values('Seasonality_Index', ascending=False)
                    st.dataframe(peak_display.head(10).reset_index(drop=True), hide_index=True)
                
                with col2:
                    st.write("**Low Seasons by SKU**")
                    low_seasons = filtered_seasonality.loc[filtered_seasonality.groupby('SKU')['Seasonality_Index'].idxmin()]
                    low_display = low_seasons[['SKU', 'Month_Name', 'Seasonality_Index']].sort_values('Seasonality_Index')
                    st.dataframe(low_display.head(10).reset_index(drop=True), hide_index=True)
                
                # Monthly performance trends
                st.subheader("📊 Monthly Performance Trends")
                monthly_avg = filtered_seasonality.groupby('Month_Name')['Seasonality_Index'].mean().reset_index()
                monthly_avg = monthly_avg.set_index('Month_Name').reindex(month_order).reset_index()
                
                fig_monthly = px.bar(monthly_avg, x='Month_Name', y='Seasonality_Index',
                                   title="Average Seasonality Index by Month")
                fig_monthly.add_hline(y=100, line_dash="dash", line_color="red", 
                                    annotation_text="Average Performance")
                st.plotly_chart(fig_monthly, use_container_width=True)
            else:
                st.warning("Please select at least one SKU for seasonality analysis.")
        else:
            st.warning("Insufficient data for seasonality analysis.")

    # ------------------------
    # SKU Comparison
    # ------------------------
    elif analysis_type == "SKU Comparison":
        st.header("⚖️ SKU Performance Comparison")
        
        # SKU selection for comparison
        comparison_skus = st.multiselect("Select SKUs to compare (2-10 SKUs):",
                                       df_filtered['SKU'].unique(),
                                       default=df_filtered.groupby('SKU')['AYP Gross Sales (USD)'].sum().nlargest(5).index.tolist(),
                                       max_selections=10)
        
        if len(comparison_skus) >= 2:
            comparison_data = df_filtered[df_filtered['SKU'].isin(comparison_skus)]
            
            # Multi-metric comparison
            comparison_metrics = ["AYP Gross Sales (USD)", "AYP Units", "Revenue_per_Unit"]
            sku_performance = comparison_data.groupby('SKU')[comparison_metrics].agg(['sum', 'mean']).round(2)
            
            # Flatten column names
            sku_performance.columns = [f'{col[0]}_{col[1]}' for col in sku_performance.columns]
            sku_performance = sku_performance.reset_index()
            
            # Normalize data for radar chart
            numeric_cols = [col for col in sku_performance.columns if col != 'SKU']
            scaler = StandardScaler()
            normalized_data = scaler.fit_transform(sku_performance[numeric_cols])
            
            # Create radar chart
            fig_radar = go.Figure()
            
            categories = ['Total Sales', 'Total Units', 'Avg Rev/Unit', 'Avg Sales', 'Avg Units', 'Avg Rev/Unit']
            
            for i, sku in enumerate(comparison_skus):
                sku_data = sku_performance[sku_performance['SKU'] == sku]
                if not sku_data.empty:
                    fig_radar.add_trace(go.Scatterpolar(
                        r=normalized_data[i],
                        theta=categories,
                        fill='toself',
                        name=sku,
                        opacity=0.6
                    ))
            
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True)),
                title="Multi-Metric SKU Performance Comparison (Normalized)",
                height=500
            )
            
            st.plotly_chart(fig_radar, use_container_width=True)
            
            # Side-by-side metrics comparison
            st.subheader("📊 Detailed Metrics Comparison")
            
            # Create comparison table
            comparison_table = comparison_data.groupby('SKU').agg({
                'AYP Gross Sales (USD)': ['sum', 'mean'],
                'AYP Units': ['sum', 'mean'],
                'Revenue_per_Unit': 'mean',
                'Month': 'nunique',
                'Country': 'nunique' if 'Country' in comparison_data.columns else lambda x: 0,
                'Store_Column': 'nunique'
            }).round(2)
            
            comparison_table.columns = ['Total Sales', 'Avg Monthly Sales', 'Total Units', 
                                      'Avg Monthly Units', 'Avg Rev/Unit', 'Active Months',
                                      'Countries', 'Stores']
            
            st.dataframe(comparison_table.sort_values('Total Sales', ascending=False), 
                        use_container_width=True)
            
            # Time series comparison
            st.subheader("📈 Sales Trend Comparison")
            time_comparison = comparison_data.groupby(['SKU', 'Month_Year'])['AYP Gross Sales (USD)'].sum().reset_index()
            
            fig_time = px.line(time_comparison, x='Month_Year', y='AYP Gross Sales (USD)',
                             color='SKU', title="Sales Trend Comparison Over Time",
                             markers=True)
            fig_time.update_xaxes(tickangle=45)
            st.plotly_chart(fig_time, use_container_width=True)
            
            # Performance ranking
            st.subheader("🏆 Performance Ranking")
            ranking_data = comparison_table.copy()
            
            # Create composite score
            ranking_data['Sales_Rank'] = ranking_data['Total Sales'].rank(ascending=False)
            ranking_data['Units_Rank'] = ranking_data['Total Units'].rank(ascending=False)
            ranking_data['Revenue_Rank'] = ranking_data['Avg Rev/Unit'].rank(ascending=False)
            ranking_data['Composite_Score'] = (ranking_data['Sales_Rank'] + 
                                             ranking_data['Units_Rank'] + 
                                             ranking_data['Revenue_Rank']) / 3
            
            ranking_display = ranking_data[['Total Sales', 'Total Units', 'Avg Rev/Unit', 
                                          'Composite_Score']].sort_values('Composite_Score')
            ranking_display.index.name = 'SKU'
            st.dataframe(ranking_display, use_container_width=True)
            
        else:
            st.warning("Please select at least 2 SKUs for comparison.")

    # ------------------------
    # SKU Outlier Detection
    # ------------------------
    elif analysis_type == "SKU Outlier Detection":
        st.header("🔍 SKU Outlier Detection")
        
        outliers = detect_outliers(df_filtered, metric, 'SKU')
        
        if not outliers.empty:
            col1, col2, col3 = st.columns(3)
            
            outlier_skus = outliers['SKU'].nunique()
            total_skus = df_filtered['SKU'].nunique()
            outlier_value = outliers[metric].sum()
            total_value = df_filtered[metric].sum()
            
            col1.metric("Outlier SKUs", f"{outlier_skus}/{total_skus}")
            col2.metric("Outlier Percentage", f"{outlier_skus/total_skus*100:.1f}%")
            col3.metric("Value Impact", f"{outlier_value/total_value*100:.1f}% of total")
            
            # Box plot showing outliers
            fig_box = px.box(df_filtered, y=metric, title=f"SKU {metric} Distribution with Outliers")
            st.plotly_chart(fig_box, use_container_width=True)
            
            # Outlier details by type
            st.subheader("📊 Outlier Analysis")
            
            # Separate positive and negative outliers
            median_value = df_filtered[metric].median()
            high_outliers = outliers[outliers[metric] > median_value]
            low_outliers = outliers[outliers[metric] <= median_value]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**High Performers (Positive Outliers)**")
                if not high_outliers.empty:
                    high_summary = high_outliers.groupby('SKU')[metric].agg(['count', 'mean', 'max']).round(2)
                    high_summary.columns = ['Outlier Count', 'Avg Value', 'Max Value']
                    st.dataframe(high_summary.sort_values('Max Value', ascending=False))
                else:
                    st.info("No positive outliers detected.")
            
            with col2:
                st.write("**Under Performers (Negative Outliers)**")
                if not low_outliers.empty:
                    low_summary = low_outliers.groupby('SKU')[metric].agg(['count', 'mean', 'min']).round(2)
                    low_summary.columns = ['Outlier Count', 'Avg Value', 'Min Value']
                    st.dataframe(low_summary.sort_values('Min Value'))
                else:
                    st.info("No negative outliers detected.")
            
            # Detailed outlier table
            st.subheader("🔍 Detailed Outlier Records")
            outlier_columns = ['SKU', metric, 'Month_Year']
            if 'Country' in outliers.columns:
                outlier_columns.append('Country')
            if store_col:
                outlier_columns.append(store_col)
            
            outlier_details = outliers[outlier_columns]
            st.dataframe(outlier_details.sort_values(metric, ascending=False), use_container_width=True, hide_index=True)
            
            # Outlier patterns
            if len(outliers) > 10:
                st.subheader("📈 Outlier Patterns")
                outlier_patterns = outliers.groupby(['Month_Year', 'SKU']).size().reset_index(name='Outlier_Count')
                
                fig_patterns = px.bar(outlier_patterns, x='Month_Year', y='Outlier_Count',
                                    color='SKU', title="Outlier Occurrences Over Time")
                fig_patterns.update_xaxes(tickangle=45)
                st.plotly_chart(fig_patterns, use_container_width=True)
        else:
            st.info("No outliers detected in the current dataset and filters.")

        # ------------------------
        # Export full SKU portfolio with BCG categories
        # ------------------------
        st.subheader("📤 Export Full Outlier Records")

        if 'outliers' in locals() and not outliers.empty:
            # Prepare CSV without index
            csv_data = outliers.to_csv(index=False).encode('utf-8')

            # Download button
            st.download_button(
                label="Download Full Outliers List as CSV",
                data=csv_data,
                file_name="outliers.csv",
                mime="text/csv"
            )
        else:
            st.info("⚠️ Outliers data not available yet. Please run SKU Portfolio Analysis with sufficient time periods.")
    # ------------------------
    # SKU Portfolio Analysis
    # ------------------------
    elif analysis_type == "SKU Portfolio Analysis":
        st.header("📋 SKU Portfolio Analysis")
        
        # Calculate comprehensive portfolio metrics
        portfolio_metrics = calculate_sku_performance_metrics(df_filtered)
        
        # Portfolio overview
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total SKUs", f"{len(portfolio_metrics):,}")
        col2.metric("Active SKUs (>$1K)", f"{len(portfolio_metrics[portfolio_metrics['Total_Sales'] > 1000]):,}")
        col3.metric("Star SKUs (>$50K)", f"{len(portfolio_metrics[portfolio_metrics['Total_Sales'] > 50000]):,}")
        col4.metric("Portfolio Concentration", f"{portfolio_metrics['Total_Sales'].head(10).sum() / portfolio_metrics['Total_Sales'].sum() * 100:.1f}%")
        
        # BCG-style matrix (Sales vs Growth)
        if len(df_filtered) > df_filtered['SKU'].nunique():  # Need multiple time periods
            growth_data = calculate_growth_rates(df_filtered, 'AYP Gross Sales (USD)', 'SKU')
            if not growth_data.empty:
                latest_growth = growth_data.groupby('SKU')['MoM_Growth'].mean().reset_index()
                bcg_data = portfolio_metrics.merge(latest_growth, on='SKU', how='left')
                bcg_data['MoM_Growth'] = bcg_data['MoM_Growth'].fillna(0)
                
                # Create BCG matrix
                sales_median = bcg_data['Total_Sales'].median()
                growth_median = bcg_data['MoM_Growth'].median()
                
                def classify_sku(row):
                    if row['Total_Sales'] >= sales_median and row['MoM_Growth'] >= growth_median:
                        return 'Stars'
                    elif row['Total_Sales'] >= sales_median and row['MoM_Growth'] < growth_median:
                        return 'Cash Cows'
                    elif row['Total_Sales'] < sales_median and row['MoM_Growth'] >= growth_median:
                        return 'Question Marks'
                    else:
                        return 'Dogs'
                
                bcg_data['BCG_Category'] = bcg_data.apply(classify_sku, axis=1)
                
                # BCG Matrix visualization
                fig_bcg = px.scatter(bcg_data, x='Total_Sales', y='MoM_Growth', 
                                color='BCG_Category', hover_data=['SKU'],
                                title="SKU Portfolio Matrix (BCG-Style Analysis)",
                                size='Total_Units', size_max=20)
                
                fig_bcg.add_hline(y=growth_median, line_dash="dash", line_color="gray")
                fig_bcg.add_vline(x=sales_median, line_dash="dash", line_color="gray")
                fig_bcg.update_layout(xaxis_title="Total Sales (USD)", yaxis_title="Average Growth %")
                
                st.plotly_chart(fig_bcg, use_container_width=True)
                
                # BCG category summary
                st.subheader("📊 Portfolio Category Analysis")
                bcg_summary = bcg_data.groupby('BCG_Category').agg({
                    'SKU': 'count',
                    'Total_Sales': 'sum',
                    'MoM_Growth': 'mean'
                }).round(2)
                bcg_summary.columns = ['SKU Count', 'Total Sales', 'Avg Growth %']
                bcg_summary['Sales %'] = (bcg_summary['Total Sales'] / bcg_summary['Total Sales'].sum() * 100).round(1)
                
                st.dataframe(bcg_summary, use_container_width=True)

                # 🔥 NEW: Top 10 SKUs in each BCG Category
                st.subheader("🏆 Top 10 SKUs per BCG Category")
                categories = ['Stars', 'Cash Cows', 'Question Marks', 'Dogs']
                
                for cat in categories:
                    cat_df = bcg_data[bcg_data['BCG_Category'] == cat].nlargest(10, 'Total_Sales')
                    if not cat_df.empty:
                        st.write(f"**Top 10 {cat}**")
                        st.dataframe(cat_df[['SKU', 'Total_Sales', 'MoM_Growth', 'Geographic_Reach', 'Store_Penetration']], hide_index=True)
                # ------------------------
                # Export full SKU portfolio with BCG categories
                # ------------------------
                st.subheader("📤 Export Full SKU Portfolio with BCG Categories")

                if 'bcg_data' in locals() and not bcg_data.empty:
                    # Prepare CSV without index
                    csv_data = bcg_data.to_csv(index=False).encode('utf-8')

                    # Download button
                    st.download_button(
                        label="Download Full SKU Portfolio as CSV",
                        data=csv_data,
                        file_name="sku_portfolio_analysis_with_bcg.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("⚠️ BCG data not available yet. Please run SKU Portfolio Analysis with sufficient time periods.")
    # # ------------------------
    # # SKU Portfolio Analysis
    # # ------------------------
    # elif analysis_type == "SKU Portfolio Analysis":
    #     st.header("📋 SKU Portfolio Analysis")
        
    #     # Calculate comprehensive portfolio metrics
    #     portfolio_metrics = calculate_sku_performance_metrics(df_filtered)
        
    #     # Portfolio overview
    #     col1, col2, col3, col4 = st.columns(4)
    #     col1.metric("Total SKUs", f"{len(portfolio_metrics):,}")
    #     col2.metric("Active SKUs (>$1K)", f"{len(portfolio_metrics[portfolio_metrics['Total_Sales'] > 1000]):,}")
    #     col3.metric("Star SKUs (>$50K)", f"{len(portfolio_metrics[portfolio_metrics['Total_Sales'] > 50000]):,}")
    #     col4.metric("Portfolio Concentration", f"{portfolio_metrics['Total_Sales'].head(10).sum() / portfolio_metrics['Total_Sales'].sum() * 100:.1f}%")
        
    #     # BCG-style matrix (Sales vs Growth)
    #     if len(df_filtered) > df_filtered['SKU'].nunique():  # Need multiple time periods
    #         growth_data = calculate_growth_rates(df_filtered, 'AYP Gross Sales (USD)', 'SKU')
    #         if not growth_data.empty:
    #             latest_growth = growth_data.groupby('SKU')['MoM_Growth'].mean().reset_index()
    #             bcg_data = portfolio_metrics.merge(latest_growth, on='SKU', how='left')
    #             bcg_data['MoM_Growth'] = bcg_data['MoM_Growth'].fillna(0)
                
    #             # Create BCG matrix
    #             sales_median = bcg_data['Total_Sales'].median()
    #             growth_median = bcg_data['MoM_Growth'].median()
                
    #             def classify_sku(row):
    #                 if row['Total_Sales'] >= sales_median and row['MoM_Growth'] >= growth_median:
    #                     return 'Stars'
    #                 elif row['Total_Sales'] >= sales_median and row['MoM_Growth'] < growth_median:
    #                     return 'Cash Cows'
    #                 elif row['Total_Sales'] < sales_median and row['MoM_Growth'] >= growth_median:
    #                     return 'Question Marks'
    #                 else:
    #                     return 'Dogs'
                
    #             bcg_data['BCG_Category'] = bcg_data.apply(classify_sku, axis=1)
                
    #             # BCG Matrix visualization
    #             fig_bcg = px.scatter(bcg_data, x='Total_Sales', y='MoM_Growth', 
    #                                color='BCG_Category', hover_data=['SKU'],
    #                                title="SKU Portfolio Matrix (BCG-Style Analysis)",
    #                                size='Total_Units', size_max=20)
                
    #             fig_bcg.add_hline(y=growth_median, line_dash="dash", line_color="gray")
    #             fig_bcg.add_vline(x=sales_median, line_dash="dash", line_color="gray")
    #             fig_bcg.update_layout(xaxis_title="Total Sales (USD)", yaxis_title="Average Growth %")
                
    #             st.plotly_chart(fig_bcg, use_container_width=True)
                
    #             # BCG category summary
    #             st.subheader("📊 Portfolio Category Analysis")
    #             bcg_summary = bcg_data.groupby('BCG_Category').agg({
    #                 'SKU': 'count',
    #                 'Total_Sales': 'sum',
    #                 'MoM_Growth': 'mean'
    #             }).round(2)
    #             bcg_summary.columns = ['SKU Count', 'Total Sales', 'Avg Growth %']
    #             bcg_summary['Sales %'] = (bcg_summary['Total Sales'] / bcg_summary['Total Sales'].sum() * 100).round(1)
                
    #             st.dataframe(bcg_summary, use_container_width=True)
        
    #     # Portfolio performance distribution
    #     st.subheader("📈 Portfolio Performance Distribution")
        
    #     col1, col2 = st.columns(2)
        
    #     with col1:
    #         # Sales distribution
    #         fig_sales_dist = px.histogram(portfolio_metrics, x='Total_Sales', nbins=30,
    #                                     title="SKU Sales Distribution")
    #         fig_sales_dist.update_layout(xaxis_title="Sales (USD)", yaxis_title="Number of SKUs")
    #         st.plotly_chart(fig_sales_dist, use_container_width=True)
        
    #     with col2:
    #         # Geographic reach vs sales
    #         fig_geo_sales = px.scatter(portfolio_metrics, x='Geographic_Reach', y='Total_Sales',
    #                                  hover_data=['SKU'], title="Geographic Reach vs Sales Performance")
    #         fig_geo_sales.update_layout(xaxis_title="Number of Countries", yaxis_title="Sales (USD)")
    #         st.plotly_chart(fig_geo_sales, use_container_width=True)
        
    #     # Portfolio efficiency metrics
    #     st.subheader("⚡ Portfolio Efficiency Metrics")
        
    #     efficiency_metrics = pd.DataFrame({
    #         'Metric': ['Sales per SKU', 'Units per SKU', 'Revenue per Unit', 'Average Geographic Reach',
    #                   'Average Store Penetration', 'Portfolio Diversity (Coefficient of Variation)'],
    #         'Value': [
    #             f"${portfolio_metrics['Total_Sales'].mean():,.0f}",
    #             f"{portfolio_metrics['Total_Units'].mean():,.0f}",
    #             f"${portfolio_metrics['Revenue_per_Unit_mean'].mean():.2f}",
    #             f"{portfolio_metrics['Geographic_Reach'].mean():.1f}",
    #             f"{portfolio_metrics['Store_Penetration'].mean():.1f}",
    #             f"{(portfolio_metrics['Total_Sales'].std() / portfolio_metrics['Total_Sales'].mean()):.2f}"
    #         ]
    #     })
        
    #     st.dataframe(efficiency_metrics, use_container_width=True, hide_index=True)
        
    #     # Top and bottom performers
    #     col1, col2 = st.columns(2)
        
    #     with col1:
    #         st.write("**Top 10 SKUs by Sales**")
    #         top_performers = portfolio_metrics.nlargest(10, 'Total_Sales')[
    #             ['SKU', 'Total_Sales', 'Geographic_Reach', 'Store_Penetration']
    #         ]
    #         top_performers.columns = ['SKU', 'Sales', 'Countries', 'Stores']
    #         st.dataframe(top_performers)
        
    #     with col2:
    #         st.write("**Bottom 10 SKUs by Sales**")
    #         bottom_performers = portfolio_metrics.nsmallest(10, 'Total_Sales')[
    #             ['SKU', 'Total_Sales', 'Geographic_Reach', 'Store_Penetration']
    #         ]
    #         bottom_performers.columns = ['SKU', 'Sales', 'Countries', 'Stores']
    #         st.dataframe(bottom_performers)

    # ------------------------
    # Executive Summary
    # ------------------------
    elif analysis_type == "Executive Summary":
        st.header("🎯 Executive Summary")
        
        # High-level KPIs
        col1, col2, col3, col4, col5 = st.columns(5)
        
        current_month_data = df_filtered[df_filtered["Month"] == df_filtered["Month"].max()]
        previous_month_data = df_filtered[df_filtered["Month"] == df_filtered["Month"].max() - pd.DateOffset(months=1)]
        
        current_sales = current_month_data["AYP Gross Sales (USD)"].sum()
        previous_sales = previous_month_data["AYP Gross Sales (USD)"].sum()
        mom_change = ((current_sales - previous_sales) / previous_sales * 100) if previous_sales > 0 else 0
        
        col1.metric("Current Month Sales", f"${current_sales:,.0f}", f"{mom_change:+.1f}%")
        col2.metric("Active SKUs", df_filtered["SKU"].nunique())
        col3.metric("Top SKU Sales", f"${df_filtered.groupby('SKU')['AYP Gross Sales (USD)'].sum().max():,.0f}")
        col4.metric("Avg SKU Performance", f"${df_filtered.groupby('SKU')['AYP Gross Sales (USD)'].sum().mean():,.0f}")
        
        if 'Country' in df_filtered.columns:
            col5.metric("Geographic Coverage", f"{df_filtered['Country'].nunique()} countries")
        else:
            col5.metric("Total Units", f"{df_filtered['AYP Units'].sum():,.0f}")
        
        # Performance trends dashboard
        st.subheader("📊 Performance Overview")
        
        # Create comprehensive dashboard
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=("Monthly Sales Trend", "SKU Performance Distribution", "Top SKUs",
                           "Geographic Performance", "Brand Performance", "Growth Trend"),
            specs=[[{"secondary_y": False}, {"secondary_y": False}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}, {"secondary_y": False}]]
        )
        
        # Monthly sales trend
        monthly_sales = df_filtered.groupby("Month_Year")["AYP Gross Sales (USD)"].sum().reset_index()
        month_order = sorted(monthly_sales["Month_Year"].unique(), 
                           key=lambda x: pd.to_datetime(x, format="%b %Y"))
        monthly_sales["Month_Year"] = pd.Categorical(monthly_sales["Month_Year"], 
                                                   categories=month_order, ordered=True)
        monthly_sales = monthly_sales.sort_values("Month_Year")
        
        fig.add_trace(go.Scatter(x=monthly_sales["Month_Year"], 
                               y=monthly_sales["AYP Gross Sales (USD)"],
                               mode='lines+markers', name='Sales', line=dict(color='blue')), 
                     row=1, col=1)
        
        # SKU performance distribution
        sku_sales = df_filtered.groupby('SKU')['AYP Gross Sales (USD)'].sum()
        fig.add_trace(go.Histogram(x=sku_sales, nbinsx=20, name='SKU Distribution',
                                 marker_color='lightblue'), row=1, col=2)
        
        # Top SKUs
        top_skus = sku_sales.nlargest(10)
        fig.add_trace(go.Bar(x=top_skus.values, y=top_skus.index, 
                           orientation='h', name='Top SKUs', marker_color='green'), row=1, col=3)
        
        # Geographic performance
        if 'Country' in df_filtered.columns:
            country_sales = df_filtered.groupby("Country")["AYP Gross Sales (USD)"].sum().nlargest(10)
            fig.add_trace(go.Bar(x=country_sales.index, y=country_sales.values, 
                               name='Countries', marker_color='orange'), row=2, col=1)
        
        # Brand performance
        brand_col = column_mapping.get('brand_column')
        if brand_col and brand_col in df_filtered.columns:
            brand_sales = df_filtered.groupby(brand_col)["AYP Gross Sales (USD)"].sum().nlargest(10)
            fig.add_trace(go.Bar(x=brand_sales.index, y=brand_sales.values, 
                               name='Brands', marker_color='purple'), row=2, col=2)
        
        # Growth trend (if possible)
        if len(monthly_sales) > 1:
            monthly_sales['Growth'] = monthly_sales['AYP Gross Sales (USD)'].pct_change() * 100
            fig.add_trace(go.Scatter(x=monthly_sales["Month_Year"], 
                                   y=monthly_sales['Growth'],
                                   mode='lines+markers', name='MoM Growth %',
                                   line=dict(color='red')), row=2, col=3)
        
        fig.update_layout(height=800, showlegend=False, 
                         title_text="Executive Dashboard Overview")
        st.plotly_chart(fig, use_container_width=True)
        
        # Key insights
        st.subheader("💡 Key Insights")
        
        insights = []
        
        # Top performing SKU
        top_sku = df_filtered.groupby('SKU')['AYP Gross Sales (USD)'].sum().idxmax()
        top_sku_sales = df_filtered.groupby('SKU')['AYP Gross Sales (USD)'].sum().max()
        insights.append(f"🏆 **Top Performing SKU:** {top_sku} with ${top_sku_sales:,.0f} in sales")
        
        # Portfolio concentration
        total_portfolio_sales = df_filtered.groupby('SKU')['AYP Gross Sales (USD)'].sum()
        top_10_concentration = total_portfolio_sales.nlargest(10).sum() / total_portfolio_sales.sum() * 100
        insights.append(f"📊 **Portfolio Concentration:** Top 10 SKUs account for {top_10_concentration:.1f}% of total sales")
        
        # Average performance
        avg_sku_sales = total_portfolio_sales.mean()
        insights.append(f"💰 **Average SKU Performance:** ${avg_sku_sales:,.0f} per SKU")
        
        # Growth insight
        if len(monthly_sales) > 1:
            latest_growth = monthly_sales['Growth'].iloc[-1]
            if not pd.isna(latest_growth):
                growth_direction = "📈 positive" if latest_growth > 0 else "📉 negative"
                insights.append(f"**Latest Month Growth:** {growth_direction} growth of {abs(latest_growth):.1f}%")
        
        # Geographic insight
        if 'Country' in df_filtered.columns:
            country_count = df_filtered['Country'].nunique()
            insights.append(f"🌐 **Geographic Reach:** Portfolio spans across {country_count} countries")
        
        # Brand insight
        brand_col = column_mapping.get('brand_column')
        if brand_col and brand_col in df_filtered.columns:
            brand_count = df_filtered[brand_col].nunique()
            insights.append(f"🏷️ **Brand Portfolio:** {brand_count} active brands")
        
        # Retailer insight
        if 'Retailer' in df_filtered.columns:
            retailer_count = df_filtered['Retailer'].nunique()
            insights.append(f"🏪 **Retailer Network:** {retailer_count} active retailers")
        
        for insight in insights:
            st.write(insight)

    # ------------------------
    # Regional Deep Dive
    # ------------------------
    else:  # Regional Deep Dive
        st.header("🌍 Regional Deep Dive with SKU Analysis")
        
        if 'Region' not in df_filtered.columns:
            st.error("Region column not found in the dataset. This analysis requires regional data.")
        else:
            selected_region = st.selectbox("Select Region for Deep Dive:", df_filtered["Region"].unique())
            region_data = df_filtered[df_filtered["Region"] == selected_region]
            
            # Regional KPIs
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("SKUs in Region", region_data["SKU"].nunique())
            col2.metric("Countries", region_data["Country"].nunique() if 'Country' in region_data.columns else 'N/A')
            
            store_col = column_mapping.get('store_column')
            if store_col and store_col in region_data.columns:
                col3.metric("Stores", region_data[store_col].nunique())
            else:
                col3.metric("Stores", 'N/A')
            
            col4.metric("Total Sales", f"${region_data['AYP Gross Sales (USD)'].sum():,.0f}")
            col5.metric("Avg SKU Sales", f"${region_data.groupby('SKU')['AYP Gross Sales (USD)'].sum().mean():,.0f}")
            
            # Regional SKU performance
            st.subheader(f"🏆 Top SKUs in {selected_region}")
            regional_sku_performance = region_data.groupby('SKU').agg({
                'AYP Gross Sales (USD)': 'sum',
                'AYP Units': 'sum',
                'Revenue_per_Unit': 'mean',
                'Country': 'nunique' if 'Country' in region_data.columns else lambda x: 1
            }).round(2)
            
            regional_sku_performance.columns = ['Total Sales', 'Total Units', 'Avg Rev/Unit', 'Countries']
            regional_sku_performance = regional_sku_performance.sort_values('Total Sales', ascending=False)
            
            st.dataframe(regional_sku_performance.head(20), use_container_width=True)
            
            # Regional analysis charts
            col1, col2 = st.columns(2)
            
            with col1:
                # SKU sales distribution in region
                fig_regional_dist = px.histogram(regional_sku_performance, x='Total Sales', nbins=20,
                                               title=f"SKU Sales Distribution in {selected_region}")
                st.plotly_chart(fig_regional_dist, use_container_width=True)
            
            with col2:
                # Top SKUs treemap
                top_regional_skus = regional_sku_performance.head(15)
                if not top_regional_skus.empty:
                    fig_treemap = px.treemap(values=top_regional_skus['Total Sales'],
                                           names=top_regional_skus.index,
                                           title=f"Top SKUs in {selected_region}")
                    st.plotly_chart(fig_treemap, use_container_width=True)
            
            # Time series for regional SKUs
            st.subheader(f"📈 SKU Trends in {selected_region}")
            
            top_5_regional_skus = regional_sku_performance.head(5).index.tolist()
            regional_time_series = region_data[region_data['SKU'].isin(top_5_regional_skus)]
            regional_trends = regional_time_series.groupby(['SKU', 'Month_Year'])['AYP Gross Sales (USD)'].sum().reset_index()
            
            fig_regional_trends = px.line(regional_trends, x='Month_Year', y='AYP Gross Sales (USD)',
                                        color='SKU', title=f"Top 5 SKU Trends in {selected_region}",
                                        markers=True)
            fig_regional_trends.update_xaxes(tickangle=45)
            st.plotly_chart(fig_regional_trends, use_container_width=True)
            
            # Country-level analysis within region
            if 'Country' in region_data.columns and region_data['Country'].nunique() > 1:
                st.subheader(f"🗺️ Country Performance within {selected_region}")
                
                country_sku_analysis = region_data.groupby(['Country', 'SKU'])['AYP Gross Sales (USD)'].sum().reset_index()
                country_totals = region_data.groupby('Country')['AYP Gross Sales (USD)'].sum().sort_values(ascending=False)
                
                # Country performance bar chart
                fig_country_performance = px.bar(x=country_totals.index, y=country_totals.values,
                                               title=f"Sales by Country in {selected_region}")
                fig_country_performance.update_layout(xaxis_title="Country", yaxis_title="Sales (USD)")
                st.plotly_chart(fig_country_performance, use_container_width=True)
                
                # SKU performance by country heatmap
                country_sku_pivot = country_sku_analysis.pivot(index='Country', columns='SKU', values='AYP Gross Sales (USD)').fillna(0)
                
                # Limit to top SKUs for readability
                top_skus_for_heatmap = regional_sku_performance.head(10).index.tolist()
                available_skus = [sku for sku in top_skus_for_heatmap if sku in country_sku_pivot.columns]
                if available_skus:
                    country_sku_pivot = country_sku_pivot[available_skus]
                    
                    if not country_sku_pivot.empty:
                        fig_heatmap = px.imshow(country_sku_pivot, 
                                              title=f"SKU Performance Heatmap by Country in {selected_region}",
                                              aspect='auto', color_continuous_scale='Blues')
                        fig_heatmap.update_layout(height=400)
                        st.plotly_chart(fig_heatmap, use_container_width=True)

    # ------------------------
    # Data Export Options
    # ------------------------
    st.sidebar.markdown("### 💾 Export Options")
    
    if st.sidebar.button("📊 Export Filtered Data"):
        csv = df_filtered.to_csv(index=False)
        st.sidebar.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"_filtered_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    if st.sidebar.button("📦 Export SKU Summary"):
        sku_summary = calculate_sku_performance_metrics(df_filtered)
        csv_summary = sku_summary.to_csv(index=False)
        st.sidebar.download_button(
            label="Download SKU Summary CSV",
            data=csv_summary,
            file_name=f"_sku_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

else:
    st.info("👆 Please upload a CSV or Excel file with SKU data to begin the analysis.")
    
    # Enhanced sample data structure guide
    st.markdown("""
    ### Expected Data Structure for SKU Analysis
    Your file should contain the following columns:
    
    **Required Columns:**
    - `Month`: YYYYMM format (e.g., 202501) or YYYY-MM format
    - `SKU`: Unique product identifier
    - `AYP Gross Sales (USD)`: Sales values in USD
    - `AYP Units`: Unit quantities
    
    **Optional Columns (for enhanced analysis):**
    - `Country`: Country name
    - `Region`: Regional grouping
    - `Brand` or `Global Brand Desc`: Brand names
    - `Door/Store` or `Global Store Desc`: Store identifiers
    - `Retailer`: Retailer information
    - `Brand Line`: Product brand line
    - `Family`: Product family grouping
    
    ### New Features for Enhanced Data Format:
    
    🆕 **Enhanced Column Detection:**
    - Automatically detects different column naming conventions
    - Supports both `Brand` and `Global Brand Desc` columns
    - Handles `Door/Store`, `Global Store Desc`, or `Store` columns
    - Flexible date format parsing (YYYYMM, YYYY-MM, standard dates)
    
    🎯 **New Filter Options:**
    - **Retailer Filtering**: Filter by specific retailers
    - **Brand Line Filtering**: Focus on specific brand lines
    - **Family Filtering**: Analyze by product families
    - **Enhanced Brand Filtering**: Works with various brand column names
    
    📊 **New Analysis Types:**
    - **Brand Analysis**: Complete brand performance breakdown
    - **Retailer Analysis**: Retailer-specific performance metrics
    - All existing SKU-level analyses remain fully functional
    
    🔧 **Improved Functionality:**
    - **Backward Compatibility**: Works with both old and new data formats
    - **Dynamic Column Mapping**: Automatically adapts to your data structure
    - **Enhanced Insights**: Includes retailer and brand line insights in executive summary
    - **Flexible Store Column Support**: Handles different store naming conventions
    
    💡 **Key Features Maintained:**
    
    📍 **Advanced Filtering System:**
    - Multi-dimensional filtering (Region, Country, Brand, SKU, Retailer, Brand Line, Family)
    - Text-based SKU search functionality
    - Numeric range filters for sales and revenue
    - Real-time filter impact display
    
    📦 **SKU-Level Analytics:**
    - Individual SKU performance tracking
    - SKU lifecycle analysis
    - Portfolio concentration metrics
    - BCG-style matrix analysis
    - SKU comparison capabilities
    
    📈 **Enhanced Analysis Types:**
    - **SKU Performance Dashboard**: Comprehensive SKU metrics and trends
    - **SKU Growth Analysis**: Month-over-month growth patterns per SKU
    - **SKU ABC Analysis**: Pareto analysis at SKU level
    - **SKU Seasonality**: Seasonal patterns for individual products
    - **SKU Comparison**: Multi-metric comparison tool
    - **SKU Outlier Detection**: Statistical anomaly identification
    - **SKU Portfolio Analysis**: BCG matrix and portfolio optimization
    - **Brand Analysis**: Brand-level performance insights (NEW)
    - **Retailer Analysis**: Retailer performance breakdown (NEW)
    
    ⚡ **Advanced Metrics:**
    - Revenue per unit calculations
    - Geographic reach per SKU
    - Store penetration metrics
    - Sales consistency indicators
    - Composite performance scores
    
    🎯 **Business Intelligence Features:**
    - Automated insights generation
    - Performance ranking systems
    - Portfolio concentration analysis
    - Exception reporting
    - Executive summary dashboards with new insights
    """)
    
    # Add sample data generator with new format
    if st.button("Generate Sample Data Structure (New Format)"):
        sample_data = pd.DataFrame({
            'Month': ['202501', '202501', '202502', '202502'],
            'SKU': ['CC001A01', 'CC002B01', 'CC001A01', 'CC002B01'],
            'Brand': ['Coach', 'Coach', 'Coach', 'Coach'],
            'AYP Gross Sales (USD)': [1136, 2300, 852, 2100],
            'AYP Units': [8, 15, 6, 14],
            'Retailer': ['Shilla', 'DFS', 'Shilla', 'DFS'],
            'Country': ['Singapore', 'Singapore', 'Singapore', 'Singapore'],
            'Region': ['Singapore', 'Singapore', 'Singapore', 'Singapore'],
            'Door/Store': ['Singapore Terminal 1', 'Changi T3', 'Singapore Terminal 1', 'Changi T3'],
            'Brand Line': ['CCH', 'CCH', 'CCH', 'CCH'],
            'Family': ['WOMA WEDP', 'MENS ACCS', 'WOMA WEDP', 'MENS ACCS']
        })
        
        st.subheader("Sample Data Structure (New Format)")
        st.dataframe(sample_data, use_container_width=True)
        
        csv_sample = sample_data.to_csv(index=False)
        st.download_button(
            label="Download Sample Data Template (New Format)",
            data=csv_sample,
            file_name="_sample_data_template_new.csv",
            mime="text/csv"
        )
    
    if st.button("Generate Sample Data Structure (Legacy Format)"):
        legacy_sample_data = pd.DataFrame({
            'Month': ['202401', '202401', '202402', '202402'],
            'SKU': ['SKU001', 'SKU002', 'SKU001', 'SKU002'],
            'AYP Gross Sales (USD)': [1500.50, 2300.75, 1650.25, 2100.00],
            'AYP Units': [50, 75, 55, 70],
            'Country': ['USA', 'Canada', 'USA', 'Canada'],
            'Region': ['North America', 'North America', 'North America', 'North America'],
            'Global Brand Desc': ['Brand A', 'Brand B', 'Brand A', 'Brand B'],
            'Global Store Desc': ['Store 001', 'Store 002', 'Store 001', 'Store 002']
        })
        
        st.subheader("Sample Data Structure (Legacy Format)")
        st.dataframe(legacy_sample_data, use_container_width=True)
        
        csv_legacy = legacy_sample_data.to_csv(index=False)
        st.download_button(
            label="Download Sample Data Template (Legacy Format)",
            data=csv_legacy,
            file_name="_sample_data_template_legacy.csv",
            mime="text/csv"
        )