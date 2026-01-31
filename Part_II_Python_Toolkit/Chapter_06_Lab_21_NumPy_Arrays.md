# Lab 21: NumPy Arrays - Fast Numerical Computing

## Chapter 6: NumPy for Data Science

### Learning Objectives
- Create and manipulate NumPy arrays
- Understand vectorization benefits
- Perform array operations 100x faster than Python lists

### Duration: 60 minutes

---

## Part 1: Array Creation

### Exercise 1.1: From Lists to Arrays
```python
import numpy as np

# Basic creation
sales_list = [100, 150, 200, 175, 125]
sales_array = np.array(sales_list)

print(f"Type: {type(sales_array)}")
print(f"Shape: {sales_array.shape}")
print(f"Data type: {sales_array.dtype}")

# 2D arrays (matrix)
sales_by_region = [
    [100, 150, 200],  # Region A
    [120, 160, 180],  # Region B
    [90, 140, 190],   # Region C
]

sales_matrix = np.array(sales_by_region)
print(f"Shape: {sales_matrix.shape}")  # (3, 4)
print(f"Dimensions: {sales_matrix.ndim}")  # 2

# YOUR CODE: Create 3D array
# Shape: (4 products, 3 regions, 7 days)
```

### Exercise 1.2: Array Generation Functions
```python
# Zeros - initialize data structures
empty_sales = np.zeros((5, 3))  # 5 days, 3 products
print("Zeros:\n", empty_sales)

# Ones - default values
default_prices = np.ones((10,)) * 29.99
print("Default prices:", default_prices)

# Range - sequences
days = np.arange(1, 32)  # Days of month
print("Days:", days)

# Linspace - evenly spaced values
price_points = np.linspace(0, 100, 11)  # 0 to 100 in 11 steps
print("Price points:", price_points)

# Random data - simulations
random_sales = np.random.randint(50, 200, size=30)  # 30 days
print("Random sales:", random_sales[:10])

# YOUR CODE: Create:
# 1. 7x24 array of hourly temperatures (use random.normal)
# 2. Identity matrix 5x5 (np.eye)
# 3. Array from 0 to 1000 in steps of 50
```

---

## Part 2: Array Operations (Vectorization)

### Exercise 2.1: Speed Comparison
```python
import time

# Python list approach
sales_list = list(range(1000000))
start = time.time()
with_tax_list = [x * 1.10 for x in sales_list]
list_time = time.time() - start

# NumPy approach
sales_array = np.array(sales_list)
start = time.time()
with_tax_array = sales_array * 1.10
numpy_time = time.time() - start

print(f"List time: {list_time:.4f}s")
print(f"NumPy time: {numpy_time:.4f}s")
print(f"Speedup: {list_time / numpy_time:.1f}x faster!")

# YOUR CODE: Compare performance for:
# - Calculating discounts
# - Filtering values > threshold
# - Calculating running total
```

### Exercise 2.2: Vectorized Calculations
```python
# Sales data for 30 days
np.random.seed(42)
daily_sales = np.random.randint(1000, 5000, 30)
prices = np.random.uniform(20, 100, 30)

# Calculations (all vectorized!)
revenue = daily_sales * prices
tax = revenue * 0.08
profit_margin = 0.3
profit = revenue * profit_margin
net_revenue = revenue - tax

# Statistical operations
print(f"Average daily revenue: ${revenue.mean():,.2f}")
print(f"Total month revenue: ${revenue.sum():,.2f}")
print(f"Best day: ${revenue.max():,.2f}")
print(f"Worst day: ${revenue.min():,.2f}")
print(f"Std deviation: ${revenue.std():,.2f}")

# YOUR CODE: Calculate
# 1. Week-over-week growth rate
# 2. Days above/below average
# 3. Cumulative revenue (np.cumsum)
```

---

## Part 3: Array Indexing & Slicing

### Exercise 3.1: Basic Indexing
```python
sales = np.array([100, 150, 200, 175, 125, 180, 160])

# Single element
first_day = sales[0]
last_day = sales[-1]
print(f"First: {first_day}, Last: {last_day}")

# Slicing
first_three = sales[:3]
last_three = sales[-3:]
middle = sales[2:5]
every_other = sales[::2]

print(f"First 3 days: {first_three}")
print(f"Every other day: {every_other}")

# 2D indexing
sales_matrix = np.array([
    [100, 150, 200],  # Week 1
    [120, 160, 180],  # Week 2
    [90, 140, 190],   # Week 3
])

# Row and column
week1 = sales_matrix[0, :]  # First row
product1 = sales_matrix[:, 0]  # First column
specific = sales_matrix[1, 2]  # Week 2, Product 3

print(f"Week 1: {week1}")
print(f"Product 1 across weeks: {product1}")

# YOUR CODE: Extract:
# 1. Last 2 weeks
# 2. Products 2 and 3
# 3. Bottom-right 2x2 submatrix
```

### Exercise 3.2: Boolean Indexing
```python
daily_sales = np.array([100, 250, 180, 320, 150, 290, 210])

# Boolean mask
high_sales = daily_sales > 200
print(f"High sales days: {high_sales}")

# Filter array
high_sales_values = daily_sales[high_sales]
print(f"High sales amounts: {high_sales_values}")

# Multiple conditions
medium_sales = (daily_sales >= 150) & (daily_sales < 300)
medium_values = daily_sales[medium_sales]
print(f"Medium sales: {medium_values}")

# Practical: Find outliers
mean = daily_sales.mean()
std = daily_sales.std()
outliers = np.abs(daily_sales - mean) > 2 * std
print(f"Outlier days: {np.where(outliers)[0]}")

# YOUR CODE: Find
# 1. Weekend sales (indices 5, 6)
# 2. Days within 10% of average
# 3. Bottom 25% of days
```

---

## Part 4: Array Reshaping

### Exercise 4.1: Reshape Operations
```python
# Flatten data for processing
sales_2d = np.array([
    [100, 150, 200],
    [120, 160, 180],
])

flattened = sales_2d.flatten()
print(f"Flattened: {flattened}")

# Reshape back
reshaped = flattened.reshape(2, 3)
print(f"Reshaped:\n{reshaped}")

# Auto-calculate dimension with -1
auto_reshape = flattened.reshape(3, -1)  # 3 rows, auto cols
print(f"Auto reshape:\n{auto_reshape}")

# Practical: Hourly to daily aggregation
hourly_sales = np.random.randint(10, 50, 168)  # 7 days × 24 hours
daily = hourly_sales.reshape(7, 24)
daily_totals = daily.sum(axis=1)
print(f"Daily totals: {daily_totals}")

# YOUR CODE: Reshape
# 1. 12 months → 4 quarters × 3 months
# 2. 365 days → 52 weeks × 7 days (trim to 364)
# 3. Minute data → hourly averages
```

### Exercise 4.2: Transpose and Axes
```python
# Sales: 3 products × 4 regions
sales = np.array([
    [100, 120, 90, 110],   # Product A
    [150, 140, 160, 155],  # Product B
    [200, 180, 190, 210],  # Product C
])

print(f"Original shape: {sales.shape}")  # (3, 4)

# Transpose: flip axes
transposed = sales.T
print(f"Transposed shape: {transposed.shape}")  # (4, 3)

# Now: 4 regions × 3 products
print("Sales by region:\n", transposed)

# Sum along axes
product_totals = sales.sum(axis=1)  # Sum across regions
region_totals = sales.sum(axis=0)   # Sum across products

print(f"Product totals: {product_totals}")
print(f"Region totals: {region_totals}")

# YOUR CODE: Calculate
# 1. Average by product
# 2. Max by region
# 3. Region with highest total
```

---

## Part 5: Array Concatenation & Splitting

### Exercise 5.1: Combining Arrays
```python
# Q1 and Q2 sales
q1_sales = np.array([100, 120, 110])
q2_sales = np.array([150, 160, 155])

# Concatenate
h1_sales = np.concatenate([q1_sales, q2_sales])
print(f"H1 sales: {h1_sales}")

# Stack vertically (rows)
sales_by_quarter = np.vstack([q1_sales, q2_sales])
print(f"Stacked vertically:\n{sales_by_quarter}")

# Stack horizontally (columns)
sales_comparison = np.hstack([q1_sales.reshape(-1, 1), 
                              q2_sales.reshape(-1, 1)])
print(f"Side by side:\n{sales_comparison}")

# Practical: Combine multiple datasets
regions_a = np.random.randint(100, 200, (5, 3))
regions_b = np.random.randint(100, 200, (5, 3))
all_regions = np.vstack([regions_a, regions_b])
print(f"All regions shape: {all_regions.shape}")

# YOUR CODE: Combine
# 1. Add new product row
# 2. Add new region column
# 3. Merge multiple monthly reports
```

### Exercise 5.2: Splitting Arrays
```python
# 12 months of data
annual_sales = np.random.randint(1000, 5000, 12)

# Split into quarters
q1, q2, q3, q4 = np.split(annual_sales, 4)
print(f"Q1: {q1}")
print(f"Q4: {q4}")

# Unequal splits
first_half, second_half = np.split(annual_sales, [6])
print(f"H1: {first_half}")
print(f"H2: {second_half}")

# 2D splits
sales_matrix = np.random.randint(100, 200, (6, 4))
top, bottom = np.vsplit(sales_matrix, 2)
left, right = np.hsplit(sales_matrix, 2)

print(f"Top half:\n{top}")
print(f"Left half:\n{left}")

# YOUR CODE: Split
# 1. Weekly data into weekday/weekend
# 2. Train/test split (80/20)
# 3. Quarterly data from monthly
```

---

## Part 6: Real-World Application

### Exercise 6.1: Sales Analytics Dashboard
```python
import numpy as np

# Generate realistic sales data
np.random.seed(42)
num_days = 90
num_products = 5

# Sales (days × products)
sales = np.random.randint(50, 300, (num_days, num_products))
prices = np.array([29.99, 39.99, 19.99, 49.99, 34.99])

# Analytics
revenue = sales * prices  # Broadcasting!
daily_revenue = revenue.sum(axis=1)
product_revenue = revenue.sum(axis=0)

# Statistics
print("=== 90-Day Sales Report ===")
print(f"Total Revenue: ${daily_revenue.sum():,.2f}")
print(f"Average Daily: ${daily_revenue.mean():,.2f}")
print(f"Best Day: ${daily_revenue.max():,.2f}")
print(f"Worst Day: ${daily_revenue.min():,.2f}")

# Product performance
best_product = product_revenue.argmax()
print(f"\nBest Product: Product {best_product + 1}")
print(f"Revenue: ${product_revenue[best_product]:,.2f}")

# Trends
weekly_avg = daily_revenue.reshape(-1, 7).mean(axis=1)
trend = np.polyfit(range(len(weekly_avg)), weekly_avg, 1)[0]
print(f"\nWeekly Trend: ${trend:.2f} per week")

# YOUR CODE: Add
# 1. Month-over-month growth
# 2. Product correlation matrix
# 3. Forecast next week (simple average)
```

### Exercise 6.2: Customer Segmentation
```python
# Customer data: [age, income, purchases]
np.random.seed(42)
customers = np.column_stack([
    np.random.randint(18, 70, 1000),     # Age
    np.random.randint(20000, 150000, 1000),  # Income
    np.random.randint(1, 50, 1000),      # Purchases
])

print(f"Customer data shape: {customers.shape}")

# Basic segmentation
high_value = (customers[:, 1] > 80000) & (customers[:, 2] > 20)
young = customers[:, 0] < 30
seniors = customers[:, 0] > 60

print(f"High-value customers: {high_value.sum()}")
print(f"Young customers: {young.sum()}")
print(f"Senior customers: {seniors.sum()}")

# Segment statistics
high_value_customers = customers[high_value]
print(f"\nHigh-value segment:")
print(f"  Avg age: {high_value_customers[:, 0].mean():.1f}")
print(f"  Avg income: ${high_value_customers[:, 1].mean():,.0f}")
print(f"  Avg purchases: {high_value_customers[:, 2].mean():.1f}")

# YOUR CODE: Create segments
# 1. Budget-conscious (income < 40k, purchases < 10)
# 2. Millennials (age 25-40)
# 3. Power users (purchases > 30)
# 4. Calculate segment values
```

---

## Deliverables

1. ✅ Array creation and manipulation functions
2. ✅ Vectorized operations (100x faster)
3. ✅ Boolean indexing for filtering
4. ✅ Reshaping and aggregation
5. ✅ Sales analytics dashboard
6. ✅ Customer segmentation analysis

---

## Challenge: Financial Portfolio Analysis

```python
# Build a portfolio analyzer with:
# - 10 stocks, 252 trading days
# - Random daily returns
# - Calculate:
#   * Portfolio value over time
#   * Volatility (std dev)
#   * Sharpe ratio
#   * Maximum drawdown
#   * Correlation matrix
#   * Rebalancing strategy

# Hint: Use np.random.normal for returns
# Hint: Use np.cumprod for cumulative growth
```

**Next Lab:** Lab 22 - NumPy Broadcasting
