# Lab 26-30: Data Wrangling & Cleaning - Complete Guide

## Chapter 7: Pandas for Data Analysis

### 5 Labs in 1: Data Cleaning Mastery

---

# Lab 26: Data Cleaning Fundamentals

## Duration: 60 minutes

### Exercise 1: Handling Messy Data
```python
import pandas as pd
import numpy as np

# Messy customer data
messy_data = pd.DataFrame({
    'Name': ['  alice johnson  ', 'BOB SMITH', 'charlie_brown', 'Diana Lee', None],
    'Email': ['ALICE@TEST.COM', 'bob@test', 'charlie@test.com', 'diana@test.com', ''],
    'Phone': ['555-1234', '(555) 2345', '555.3456', '5554567', '555-0000'],
    'Age': ['25', '30', '-5', '150', '35'],
    'Revenue': ['$1,234.56', '$2,345', 'N/A', '$3,456.78', '']
})

# Clean names
messy_data['Name'] = messy_data['Name'].str.strip().str.title().str.replace('_', ' ')

# Clean emails
messy_data['Email'] = messy_data['Email'].str.lower().str.strip()
valid_emails = messy_data['Email'].str.contains('@', na=False) & \
               messy_data['Email'].str.contains('.', na=False)
messy_data.loc[~valid_emails, 'Email'] = np.nan

# Clean phone numbers
messy_data['Phone'] = messy_data['Phone'].str.replace(r'[^\d]', '', regex=True)

# Clean ages
messy_data['Age'] = pd.to_numeric(messy_data['Age'], errors='coerce')
messy_data.loc[(messy_data['Age'] < 18) | (messy_data['Age'] > 100), 'Age'] = np.nan

# Clean revenue
messy_data['Revenue'] = messy_data['Revenue'].str.replace('[\$,]', '', regex=True)
messy_data['Revenue'] = pd.to_numeric(messy_data['Revenue'], errors='coerce')

print(messy_data)
```

---

# Lab 27: Merging & Joining DataFrames

## Duration: 60 minutes

### Exercise 1: Combining Datasets
```python
# Customers
customers = pd.DataFrame({
    'customer_id': [1, 2, 3, 4],
    'name': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'segment': ['Premium', 'Standard', 'Premium', 'Basic']
})

# Orders
orders = pd.DataFrame({
    'order_id': [101, 102, 103, 104, 105],
    'customer_id': [1, 2, 1, 3, 5],  # Note: customer 5 doesn't exist
    'amount': [100, 150, 200, 175, 125]
})

# Inner join (only matching)
inner = pd.merge(customers, orders, on='customer_id', how='inner')
print(f"Inner join:\n{inner}")

# Left join (all customers)
left = pd.merge(customers, orders, on='customer_id', how='left')
print(f"Left join:\n{left}")

# Outer join (all records)
outer = pd.merge(customers, orders, on='customer_id', how='outer')

# Aggregate after join
customer_totals = left.groupby(['customer_id', 'name', 'segment'])['amount'].sum().reset_index()
print(customer_totals)
```

---

# Lab 28: Reshaping Data (Pivot & Melt)

## Duration: 60 minutes

### Exercise 1: Wide to Long
```python
# Wide format
sales_wide = pd.DataFrame({
    'Product': ['A', 'B', 'C'],
    'Q1': [100, 150, 200],
    'Q2': [120, 160, 210],
    'Q3': [110, 155, 205],
    'Q4': [130, 165, 215]
})

# Melt to long format
sales_long = pd.melt(sales_wide, 
                     id_vars=['Product'],
                     var_name='Quarter',
                     value_name='Sales')
print(sales_long)

# Pivot back to wide
sales_pivot = sales_long.pivot(index='Product', columns='Quarter', values='Sales')
print(sales_pivot)

# Pivot table with aggregation
transactions = pd.DataFrame({
    'Date': pd.date_range('2024-01-01', periods=20),
    'Product': np.random.choice(['A', 'B', 'C'], 20),
    'Region': np.random.choice(['North', 'South'], 20),
    'Sales': np.random.randint(100, 300, 20)
})

pivot_table = transactions.pivot_table(
    values='Sales',
    index='Product',
    columns='Region',
    aggfunc='sum',
    fill_value=0
)
print(pivot_table)
```

---

# Lab 29: Advanced Aggregation

## Duration: 60 minutes

### Exercise 1: Custom Aggregations
```python
# Sales data
df = pd.DataFrame({
    'date': pd.date_range('2024-01-01', periods=100),
    'product': np.random.choice(['A', 'B', 'C'], 100),
    'region': np.random.choice(['North', 'South', 'East', 'West'], 100),
    'quantity': np.random.randint(1, 50, 100),
    'price': np.random.uniform(10, 100, 100)
})

df['revenue'] = df['quantity'] * df['price']

# Multiple aggregations
summary = df.groupby(['product', 'region']).agg({
    'revenue': ['sum', 'mean', 'count'],
    'quantity': ['sum', 'mean'],
    'price': ['min', 'max']
})

# Custom aggregation function
def revenue_per_unit(x):
    return x['revenue'].sum() / x['quantity'].sum()

custom_agg = df.groupby('product').apply(revenue_per_unit)

# Rolling aggregations
df = df.sort_values('date')
df['rolling_avg_7d'] = df.groupby('product')['revenue'].transform(
    lambda x: x.rolling(window=7, min_periods=1).mean()
)

# Cumulative
df['cumulative_revenue'] = df.groupby('product')['revenue'].cumsum()

print(df.head(20))
```

---

# Lab 30: Time Series Operations

## Duration: 60 minutes

### Exercise 1: Date Operations
```python
# Time series data
dates = pd.date_range('2024-01-01', '2024-12-31', freq='D')
df = pd.DataFrame({
    'date': dates,
    'sales': np.random.normal(1000, 200, len(dates))
})

# Set datetime index
df = df.set_index('date')

# Extract date components
df['year'] = df.index.year
df['month'] = df.index.month
df['day'] = df.index.day
df['dayofweek'] = df.index.dayofweek
df['quarter'] = df.index.quarter

# Resampling
daily = df['sales']
weekly = daily.resample('W').sum()
monthly = daily.resample('M').mean()

# Rolling statistics
df['ma7'] = daily.rolling(window=7).mean()
df['ma30'] = daily.rolling(window=30).mean()
df['std7'] = daily.rolling(window=7).std()

# Lag features
df['sales_lag1'] = df['sales'].shift(1)
df['sales_lag7'] = df['sales'].shift(7)
df['growth'] = (df['sales'] - df['sales_lag1']) / df['sales_lag1']

# Time-based selections
jan_sales = df['2024-01']
q1_sales = df['2024-01':'2024-03']

print(df.head())
```

---

## Deliverables (Labs 26-30)

1. ✅ Data cleaning pipeline
2. ✅ Multi-table joins
3. ✅ Pivot tables
4. ✅ Custom aggregations
5. ✅ Time series features

**Next:** Chapter 8 - SQL
