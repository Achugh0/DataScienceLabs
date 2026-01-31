# Lab 24: Pandas Series - 1D Labeled Data

## Chapter 7: Pandas for Data Analysis

### Learning Objectives
- Work with Pandas Series
- Handle labeled data efficiently
- Perform time series operations

### Duration: 60 minutes

---

## Part 1: Creating Series

```python
import pandas as pd
import numpy as np

# From list
sales = pd.Series([100, 150, 200, 175, 125])
print(sales)

# With custom index
days = pd.Series(
    [100, 150, 200, 175, 125],
    index=['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
)
print(days)

# From dictionary
product_prices = pd.Series({
    'Laptop': 999.99,
    'Mouse': 29.99,
    'Keyboard': 79.99,
    'Monitor': 299.99
})
print(product_prices)

# From NumPy array with date index
dates = pd.date_range('2024-01-01', periods=7)
daily_sales = pd.Series(np.random.randint(1000, 5000, 7), index=dates)
print(daily_sales)

# YOUR CODE: Create Series for
# - Monthly temperatures
# - Customer satisfaction scores by region
# - Hourly website traffic
```

---

## Part 2: Indexing & Selection

```python
# Positional indexing
print(product_prices[0])  # First item
print(product_prices[1:3])  # Slice

# Label indexing
print(product_prices['Laptop'])
print(product_prices[['Laptop', 'Mouse']])

# Boolean indexing
expensive = product_prices > 100
print(product_prices[expensive])

# loc and iloc
print(product_prices.loc['Laptop'])  # Label
print(product_prices.iloc[0])  # Position

# YOUR CODE: Extract
# - Products between $50-$500
# - Top 3 most expensive
# - Products starting with 'M'
```

---

## Part 3: Operations

```python
# Arithmetic
prices = pd.Series([29.99, 39.99, 49.99], index=['A', 'B', 'C'])
with_tax = prices * 1.10
discount = prices * 0.80

# Alignment by index
sales_east = pd.Series([100, 200, 300], index=['A', 'B', 'C'])
sales_west = pd.Series([150, 250], index=['A', 'B'])

total = sales_east + sales_west  # Automatic alignment
print(total)  # C becomes NaN

# Fill missing values
total_filled = sales_east.add(sales_west, fill_value=0)
print(total_filled)

# Statistical methods
print(f"Mean: {prices.mean()}")
print(f"Median: {prices.median()}")
print(f"Std: {prices.std()}")

# YOUR CODE: Calculate
# - Revenue = quantity * price
# - Year-over-year growth
# - Z-scores
```

---

## Part 4: Time Series

```python
# Date range
dates = pd.date_range('2024-01-01', '2024-01-31', freq='D')
sales = pd.Series(np.random.randint(1000, 5000, len(dates)), index=dates)

# Resampling
weekly = sales.resample('W').sum()
print(f"Weekly totals:\n{weekly}")

# Rolling window
ma7 = sales.rolling(window=7).mean()
print(f"7-day moving average:\n{ma7.tail()}")

# Shift (for comparisons)
prev_day = sales.shift(1)
growth = (sales - prev_day) / prev_day
print(f"Daily growth:\n{growth.tail()}")

# YOUR CODE: Calculate
# - Month-end values
# - Expanding mean
# - Day-of-week patterns
```

---

## Part 5: String Operations

```python
# String methods
products = pd.Series(['Laptop', 'MOUSE', 'keyboard', 'Monitor'])

# Clean
clean = products.str.lower().str.title()
print(clean)

# Filter
starts_m = products.str.startswith('M', na=False)
print(products[starts_m])

# Extract patterns
codes = pd.Series(['PROD-100', 'PROD-200', 'ITEM-300'])
numbers = codes.str.extract(r'(\d+)')
print(numbers)

# YOUR CODE: Clean
# - Extract email domains
# - Standardize phone numbers
# - Parse SKU codes
```

---

## Deliverables

Build sales analysis functions using Series.

**Next Lab:** Lab 25 - DataFrames
