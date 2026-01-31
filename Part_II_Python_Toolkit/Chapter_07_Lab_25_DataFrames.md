# Lab 25: Pandas DataFrames - 2D Tabular Data

## Chapter 7: Pandas for Data Analysis

### Learning Objectives
- Create and manipulate DataFrames
- Perform data selection and filtering
- Handle missing data

### Duration: 75 minutes

---

## Part 1: Creating DataFrames

```python
import pandas as pd
import numpy as np

# From dictionary
data = {
    'product': ['Laptop', 'Mouse', 'Keyboard', 'Monitor'],
    'price': [999.99, 29.99, 79.99, 299.99],
    'quantity': [10, 50, 30, 15]
}
df = pd.DataFrame(data)
print(df)

# From lists of lists
transactions = [
    ['2024-01-01', 'A', 100, 29.99],
    ['2024-01-01', 'B', 50, 39.99],
    ['2024-01-02', 'A', 75, 29.99],
]
df_trans = pd.DataFrame(transactions, 
                       columns=['date', 'product', 'quantity', 'price'])

# From CSV (simulated)
# df = pd.read_csv('sales.csv')

# YOUR CODE: Create DataFrame
# - Customer database with name, email, join_date, purchases
# - Sales data with multiple columns
# - Time series data with datetime index
```

---

## Part 2: Inspecting Data

```python
# Generate sample data
np.random.seed(42)
df = pd.DataFrame({
    'customer_id': range(1, 101),
    'age': np.random.randint(18, 70, 100),
    'purchases': np.random.randint(1, 50, 100),
    'revenue': np.random.uniform(100, 5000, 100),
    'region': np.random.choice(['North', 'South', 'East', 'West'], 100)
})

# Basic info
print(df.head())  # First 5 rows
print(df.tail(3))  # Last 3 rows
print(df.info())  # Data types, non-null counts
print(df.describe())  # Statistical summary
print(df.shape)  # (rows, columns)
print(df.columns)  # Column names
print(df.dtypes)  # Data types

# Quick stats
print(f"Mean age: {df['age'].mean():.1f}")
print(f"Total revenue: ${df['revenue'].sum():,.2f}")
print(f"Unique regions: {df['region'].nunique()}")

# YOUR CODE: Explore
# - Find data quality issues
# - Identify outliers
# - Check for duplicates
```

---

## Part 3: Selection & Filtering

```python
# Select columns
ages = df['age']  # Series
subset = df[['age', 'purchases']]  # DataFrame

# Select rows by position
first_10 = df.head(10)
rows_5_to_15 = df.iloc[5:15]

# Select by label
# df.loc[row_label, col_label]

# Boolean indexing
young_customers = df[df['age'] < 30]
high_value = df[df['revenue'] > 1000]

# Multiple conditions
target_segment = df[
    (df['age'] >= 25) & 
    (df['age'] <= 45) & 
    (df['purchases'] > 10)
]

print(f"Target segment: {len(target_segment)} customers")

# isin for multiple values
coastal = df[df['region'].isin(['East', 'West'])]

# String matching
# names_with_a = df[df['name'].str.contains('a', case=False)]

# YOUR CODE: Filter
# - Customers who purchased in last month
# - Top 10% by revenue
# - Customers in multiple regions
```

---

## Part 4: Adding & Modifying Columns

```python
# Add calculated column
df['revenue_per_purchase'] = df['revenue'] / df['purchases']

# Conditional column
df['customer_tier'] = np.where(
    df['revenue'] > 2000, 'Premium',
    np.where(df['revenue'] > 1000, 'Standard', 'Basic')
)

# Apply function
def categorize_age(age):
    if age < 25:
        return 'Young'
    elif age < 50:
        return 'Middle'
    return 'Senior'

df['age_group'] = df['age'].apply(categorize_age)

# Rename columns
df_renamed = df.rename(columns={
    'customer_id': 'id',
    'purchases': 'total_purchases'
})

# Drop columns
df_slim = df.drop(['revenue_per_purchase'], axis=1)

# YOUR CODE: Add columns
# - Customer lifetime value
# - Discount tier
# - Risk score
# - Next purchase prediction
```

---

## Part 5: Handling Missing Data

```python
# Create data with missing values
df_missing = pd.DataFrame({
    'A': [1, 2, np.nan, 4, 5],
    'B': [np.nan, 2, 3, np.nan, 5],
    'C': [1, 2, 3, 4, 5]
})

# Detect missing
print(df_missing.isnull())
print(df_missing.isnull().sum())

# Drop missing
df_dropped = df_missing.dropna()  # Drop any row with NaN
df_dropped_cols = df_missing.dropna(axis=1)  # Drop columns with NaN

# Fill missing
df_filled = df_missing.fillna(0)
df_forward = df_missing.fillna(method='ffill')  # Forward fill
df_mean = df_missing.fillna(df_missing.mean())

# Interpolate
df_interp = df_missing.interpolate()

# YOUR CODE: Handle missing
# - Fill categorical with mode
# - Fill numerical with median
# - Use ML to impute
```

---

## Part 6: Sorting & Ranking

```python
# Sort by column
df_sorted = df.sort_values('revenue', ascending=False)
print(df_sorted.head())

# Sort by multiple columns
df_multi = df.sort_values(['region', 'revenue'], ascending=[True, False])

# Rank
df['revenue_rank'] = df['revenue'].rank(ascending=False)
df['percentile'] = df['revenue'].rank(pct=True)

# Top N per group
top_per_region = df.groupby('region').apply(
    lambda x: x.nlargest(3, 'revenue')
).reset_index(drop=True)

# YOUR CODE: Find
# - Top 10 customers globally
# - Top 5 per region
# - Customers in bottom 25%
```

---

## Part 7: GroupBy Operations

```python
# Group by region
region_stats = df.groupby('region').agg({
    'revenue': ['sum', 'mean', 'count'],
    'purchases': 'sum',
    'age': 'mean'
})

print(region_stats)

# Multiple groupings
age_region_stats = df.groupby(['age_group', 'region'])['revenue'].mean()

# Transform (keep original shape)
df['region_avg_revenue'] = df.groupby('region')['revenue'].transform('mean')
df['vs_regional_avg'] = df['revenue'] - df['region_avg_revenue']

# Filter groups
large_regions = df.groupby('region').filter(lambda x: len(x) > 20)

# YOUR CODE: Analyze
# - Revenue by customer tier
# - Purchase patterns by age group
# - Regional performance rankings
```

---

## Deliverables

1. ✅ Customer segmentation analysis
2. ✅ Sales performance dashboard
3. ✅ Data quality report
4. ✅ Regional comparison

---

## Challenge: E-commerce Analytics

Build complete analytics system:
- Customer lifetime value
- Churn prediction features
- Cohort analysis
- RFM segmentation

**Next Lab:** Lab 26 - Data Cleaning
