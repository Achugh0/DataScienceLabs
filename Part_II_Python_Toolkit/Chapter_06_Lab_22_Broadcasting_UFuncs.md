# Lab 22: Broadcasting & Universal Functions

## Chapter 6: NumPy for Data Science

### Learning Objectives
- Master NumPy broadcasting rules
- Use universal functions (ufuncs) for element-wise operations
- Optimize array computations

### Duration: 60 minutes

---

## Part 1: Broadcasting Basics

### Exercise 1.1: Scalar Broadcasting
```python
import numpy as np

# Add scalar to array
sales = np.array([100, 150, 200, 175, 125])
bonus = sales + 50  # Add $50 to each
print(f"With bonus: {bonus}")

# Multiply by scalar
with_tax = sales * 1.10  # 10% tax
print(f"With tax: {with_tax}")

# Practical: Apply discounts
original_prices = np.array([29.99, 39.99, 49.99, 19.99])
discount_20 = original_prices * 0.80
discount_amounts = original_prices - discount_20
print(f"Savings: {discount_amounts}")

# YOUR CODE: Apply tiered discounts
# - Products < $30: 10% off
# - Products $30-$50: 15% off
# - Products > $50: 20% off
```

### Exercise 1.2: Array Broadcasting
```python
# Different shaped arrays
daily_sales = np.array([[100], [150], [200]])  # 3 days (3, 1)
hourly_distribution = np.array([0.05, 0.15, 0.30, 0.25, 0.15, 0.10])  # 6 time slots (6,)

# Broadcasting: (3, 1) * (6,) → (3, 6)
hourly_sales = daily_sales * hourly_distribution
print(f"Hourly sales shape: {hourly_sales.shape}")
print(f"Hourly sales:\n{hourly_sales}")

# Verify: each row sums to daily total
print(f"Daily totals: {hourly_sales.sum(axis=1)}")

# YOUR CODE: Broadcast
# Product prices (5,) × quantity per region (3, 1)
# Calculate revenue per product per region
```

---

## Part 2: Broadcasting Rules

### Exercise 2.1: Dimension Compatibility
```python
# Rule: Dimensions are compatible if:
# 1. They are equal, OR
# 2. One of them is 1

# Example 1: Same shape
a = np.array([1, 2, 3])
b = np.array([10, 20, 30])
result = a + b  # (3,) + (3,) = (3,)
print(result)

# Example 2: Scalar (treated as shape ())
a = np.array([1, 2, 3])
b = 10
result = a + b  # (3,) + () = (3,)
print(result)

# Example 3: Column + Row
col = np.array([[1], [2], [3]])  # (3, 1)
row = np.array([10, 20, 30])      # (3,)
result = col + row                 # (3, 1) + (3,) = (3, 3)
print(f"Column + Row:\n{result}")

# Example 4: 3D Broadcasting
matrix_2d = np.ones((4, 5))      # (4, 5)
vector_1d = np.array([1, 2, 3, 4, 5])  # (5,)
result = matrix_2d + vector_1d   # (4, 5) + (5,) = (4, 5)

# YOUR CODE: Test these cases
# (5, 3) + (3,)
# (5, 3) + (5, 1)
# (5, 1, 3) + (3,)
```

### Exercise 2.2: Sales Analysis with Broadcasting
```python
# 4 products × 7 days
sales_quantity = np.random.randint(50, 200, (4, 7))

# Product prices (broadcast column)
prices = np.array([[29.99], [39.99], [49.99], [19.99]])  # (4, 1)

# Calculate daily revenue per product
revenue = sales_quantity * prices  # (4, 7) * (4, 1) = (4, 7)

print(f"Revenue:\n{revenue}")

# Daily costs (broadcast row)
daily_costs = np.array([100, 120, 110, 115, 125, 105, 100])  # (7,)

# Profit per product per day
profit = revenue - daily_costs  # (4, 7) - (7,) = (4, 7)

print(f"Profit:\n{profit}")

# Total profit by product
product_profit = profit.sum(axis=1)
print(f"Product profits: {product_profit}")

# YOUR CODE: Add
# - Profit margin percentage
# - Best day for each product
# - Most profitable product
```

---

## Part 3: Universal Functions (ufuncs)

### Exercise 3.1: Math Operations
```python
sales = np.array([100, 150, 200, 175, 125])

# Square root
print(f"Square root: {np.sqrt(sales)}")

# Exponential
growth_rate = 0.05
future_value = sales * np.exp(growth_rate)
print(f"After 5% growth: {future_value}")

# Logarithm
log_sales = np.log(sales)
print(f"Log sales: {log_sales}")

# Trigonometric (seasonal patterns)
days = np.arange(365)
seasonal = 100 + 50 * np.sin(2 * np.pi * days / 365)
print(f"First week: {seasonal[:7]}")

# YOUR CODE: Calculate
# 1. Compound growth over 12 months
# 2. Normalize data using log transform
# 3. Model seasonal demand
```

### Exercise 3.2: Statistical ufuncs
```python
sales = np.random.normal(1000, 200, 100)  # 100 days

# Basic stats
print(f"Mean: {np.mean(sales):.2f}")
print(f"Median: {np.median(sales):.2f}")
print(f"Std Dev: {np.std(sales):.2f}")
print(f"Variance: {np.var(sales):.2f}")

# Percentiles
p25 = np.percentile(sales, 25)
p75 = np.percentile(sales, 75)
print(f"25th percentile: {p25:.2f}")
print(f"75th percentile: {p75:.2f}")

# Min/Max
print(f"Range: {np.ptp(sales):.2f}")  # Peak-to-peak

# Cumulative
cumulative = np.cumsum(sales)
print(f"Cumulative sales: {cumulative[-1]:.2f}")

# Moving average
window = 7
moving_avg = np.convolve(sales, np.ones(window)/window, mode='valid')
print(f"7-day moving average: {moving_avg[:5]}")

# YOUR CODE: Calculate
# - Z-scores
# - Interquartile range
# - Running maximum
```

---

## Part 4: Comparison & Logic Operations

### Exercise 4.1: Boolean Operations
```python
sales = np.array([100, 250, 180, 320, 150, 290, 210])
target = 200

# Comparisons
above_target = sales > target
below_target = sales < target
at_target = sales == target

print(f"Above target: {above_target.sum()} days")
print(f"Success rate: {above_target.mean():.1%}")

# Logical AND, OR, NOT
high_sales = sales > 200
very_high = sales > 300
medium = high_sales & ~very_high  # High but not very high

print(f"Medium sales days: {medium.sum()}")

# Practical: Inventory alerts
inventory = np.array([45, 12, 8, 55, 3, 22])
low_stock = inventory < 10
critical = inventory < 5
reorder_needed = low_stock | critical

print(f"Reorder these products: {np.where(reorder_needed)[0]}")

# YOUR CODE: Create alerts for
# - Sales declining 3 days in a row
# - Inventory below reorder point
# - Prices outside acceptable range
```

### Exercise 4.2: Conditional Selection
```python
sales = np.array([100, 250, 180, 320, 150, 290, 210])

# np.where: vectorized if-else
bonus = np.where(sales > 200, 100, 50)
print(f"Bonuses: {bonus}")

# Multiple conditions
performance = np.where(
    sales > 300, 'Excellent',
    np.where(sales > 200, 'Good',
    np.where(sales > 100, 'Fair', 'Poor'))
)
print(f"Performance: {performance}")

# np.select: multiple conditions
conditions = [
    sales > 300,
    sales > 200,
    sales > 100
]
choices = ['Excellent', 'Good', 'Fair']
rating = np.select(conditions, choices, default='Poor')
print(f"Ratings: {rating}")

# YOUR CODE: Assign
# - Shipping method based on order value
# - Customer tier based on purchase history
# - Priority level based on inventory
```

---

## Part 5: Advanced Broadcasting

### Exercise 5.1: Outer Products
```python
# Multiplication table
a = np.arange(1, 6)
b = np.arange(1, 6)
table = a[:, np.newaxis] * b
print(f"Multiplication table:\n{table}")

# Practical: Price matrix
quantities = np.array([1, 2, 5, 10, 20])[:, np.newaxis]
unit_prices = np.array([29.99, 39.99, 49.99])
price_matrix = quantities * unit_prices
print(f"Price matrix:\n{price_matrix}")

# Distance matrix
cities = np.array([0, 50, 120, 200])  # Positions
distances = np.abs(cities[:, np.newaxis] - cities)
print(f"Distance matrix:\n{distances}")

# YOUR CODE: Create
# - Discount matrix (quantities × discount rates)
# - Shipping cost matrix (weights × zones)
# - Time difference matrix (timezones)
```

### Exercise 5.2: 3D Broadcasting
```python
# 3 products × 4 regions × 7 days
products = np.arange(3)
regions = np.arange(4)
days = np.arange(7)

# Reshape for broadcasting
p = products[:, np.newaxis, np.newaxis]      # (3, 1, 1)
r = regions[np.newaxis, :, np.newaxis]       # (1, 4, 1)
d = days[np.newaxis, np.newaxis, :]          # (1, 1, 7)

# Base sales influenced by all dimensions
base = 100
product_factor = 1 + p * 0.2      # Products 0, 1, 2
region_factor = 1 + r * 0.1       # Regions 0, 1, 2, 3
day_factor = 1 + np.sin(d) * 0.1  # Day of week effect

sales = base * product_factor * region_factor * day_factor
print(f"Sales shape: {sales.shape}")  # (3, 4, 7)

# Analyze
print(f"Total by product: {sales.sum(axis=(1, 2))}")
print(f"Total by region: {sales.sum(axis=(0, 2))}")
print(f"Total by day: {sales.sum(axis=(0, 1))}")

# YOUR CODE: Add seasonal factor (12 months)
```

---

## Part 6: Performance Optimization

### Exercise 6.1: Vectorization vs Loops
```python
import time

# Setup
size = 1000000
a = np.random.random(size)
b = np.random.random(size)

# Method 1: Python loop
start = time.time()
result_loop = []
for i in range(len(a)):
    result_loop.append(a[i] * b[i] + 10)
loop_time = time.time() - start

# Method 2: List comprehension
start = time.time()
result_comp = [a[i] * b[i] + 10 for i in range(len(a))]
comp_time = time.time() - start

# Method 3: NumPy vectorized
start = time.time()
result_numpy = a * b + 10
numpy_time = time.time() - start

print(f"Loop: {loop_time:.4f}s")
print(f"Comprehension: {comp_time:.4f}s")
print(f"NumPy: {numpy_time:.4f}s")
print(f"Speedup: {loop_time / numpy_time:.0f}x")

# YOUR CODE: Compare performance for:
# - Calculating distance formula
# - Applying sigmoid function
# - Matrix multiplication
```

### Exercise 6.2: Memory-Efficient Operations
```python
# In-place operations save memory
large_array = np.random.random((1000, 1000))

# Creates new array (uses 2x memory)
result = large_array * 2

# In-place (uses 1x memory)
large_array *= 2

# Out parameter
a = np.arange(1000000)
b = np.arange(1000000)
result = np.empty_like(a)
np.add(a, b, out=result)  # Writes directly to result

# Views vs copies
original = np.arange(12).reshape(3, 4)
view = original[0, :]      # View (shares memory)
copy = original[0, :].copy()  # Copy (new memory)

view[0] = 999
print(f"Original changed: {original[0, 0]}")  # 999

# YOUR CODE: Optimize
# - Calculate running statistics without storing all values
# - Process large files in chunks
# - Use views for data subsets
```

---

## Deliverables

1. ✅ Broadcasting operations for sales analysis
2. ✅ Universal functions for statistics
3. ✅ Boolean indexing for filtering
4. ✅ Optimized vectorized computations
5. ✅ 3D data analysis

---

## Challenge: Recommendation Engine

Build a product recommendation system using broadcasting:

```python
# Given:
# - User ratings matrix (1000 users × 100 products)
# - User similarity matrix (1000 × 1000)
# - Product features (100 × 10)

# Calculate:
# 1. Predicted ratings using collaborative filtering
# 2. Product similarities using cosine distance
# 3. Top 10 recommendations per user
# 4. Optimize for millions of users

# Use broadcasting to avoid loops!
```

**Next Lab:** Lab 23 - Advanced Array Operations
