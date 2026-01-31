# Lab 31-36: SQL Mastery - Complete Database Course

## Chapter 8: SQL for Data Science

### 6 Labs: From Basics to Advanced

---

# Lab 31: SQL Fundamentals - SELECT & WHERE

## Duration: 60 minutes

### Exercise 1: Basic Queries
```sql
-- Setup sample database
CREATE TABLE customers (
    customer_id INT PRIMARY KEY,
    name VARCHAR(100),
    email VARCHAR(100),
    city VARCHAR(50),
    signup_date DATE,
    total_purchases INT
);

-- SELECT basics
SELECT * FROM customers;

SELECT name, email FROM customers;

SELECT DISTINCT city FROM customers;

-- WHERE filtering
SELECT * FROM customers WHERE city = 'New York';

SELECT * FROM customers WHERE total_purchases > 10;

SELECT * FROM customers 
WHERE city = 'Boston' AND total_purchases >= 5;

SELECT * FROM customers
WHERE city IN ('New York', 'Boston', 'Chicago');

SELECT * FROM customers
WHERE name LIKE 'A%';  -- Starts with A

-- Date filtering
SELECT * FROM customers
WHERE signup_date >= '2024-01-01';

-- Practice with Python
import sqlite3
import pandas as pd

conn = sqlite3.connect(':memory:')

# Create sample data
customers_df = pd.DataFrame({
    'customer_id': range(1, 101),
    'name': [f'Customer {i}' for i in range(1, 101)],
    'city': np.random.choice(['New York', 'Boston', 'Chicago'], 100),
    'total_purchases': np.random.randint(1, 50, 100)
})

customers_df.to_sql('customers', conn, index=False)

# Query
result = pd.read_sql("SELECT * FROM customers WHERE total_purchases > 20", conn)
print(result)
```

---

# Lab 32: Aggregations & GROUP BY

## Duration: 60 minutes

### Exercise 1: Statistical Queries
```sql
-- Count
SELECT COUNT(*) AS total_customers FROM customers;

SELECT city, COUNT(*) AS customer_count
FROM customers
GROUP BY city;

-- SUM, AVG
SELECT city,
       COUNT(*) AS customers,
       SUM(total_purchases) AS total_sales,
       AVG(total_purchases) AS avg_per_customer,
       MAX(total_purchases) AS top_customer,
       MIN(total_purchases) AS smallest_purchase
FROM customers
GROUP BY city
ORDER BY total_sales DESC;

-- HAVING (filter after grouping)
SELECT city, AVG(total_purchases) AS avg_purchases
FROM customers
GROUP BY city
HAVING AVG(total_purchases) > 15;

-- Multiple grouping levels
SELECT city,
       CASE
           WHEN total_purchases < 10 THEN 'Low'
           WHEN total_purchases < 25 THEN 'Medium'
           ELSE 'High'
       END AS segment,
       COUNT(*) AS count
FROM customers
GROUP BY city, segment
ORDER BY city, count DESC;
```

### Python Integration
```python
# Run aggregation query
query = """
SELECT city,
       AVG(total_purchases) as avg_purchases,
       COUNT(*) as customer_count
FROM customers
GROUP BY city
ORDER BY avg_purchases DESC
"""

stats = pd.read_sql(query, conn)
print(stats)

# Visualize
import matplotlib.pyplot as plt
stats.plot(x='city', y='avg_purchases', kind='bar')
plt.title('Average Purchases by City')
plt.show()
```

---

# Lab 33: JOINs - Combining Tables

## Duration: 75 minutes

### Exercise 1: Multi-Table Queries
```sql
-- Create related tables
CREATE TABLE orders (
    order_id INT PRIMARY KEY,
    customer_id INT,
    order_date DATE,
    amount DECIMAL(10, 2)
);

CREATE TABLE products (
    product_id INT PRIMARY KEY,
    product_name VARCHAR(100),
    category VARCHAR(50),
    price DECIMAL(10, 2)
);

CREATE TABLE order_items (
    order_id INT,
    product_id INT,
    quantity INT
);

-- INNER JOIN
SELECT c.name, o.order_id, o.amount
FROM customers c
INNER JOIN orders o ON c.customer_id = o.customer_id
WHERE o.amount > 100;

-- LEFT JOIN (all customers, even without orders)
SELECT c.name, COUNT(o.order_id) AS order_count
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id
GROUP BY c.customer_id, c.name;

-- Multiple JOINs
SELECT c.name, p.product_name, oi.quantity, o.order_date
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
JOIN order_items oi ON o.order_id = oi.order_id
JOIN products p ON oi.product_id = p.product_id
WHERE o.order_date >= '2024-01-01';

-- Self JOIN (find customers in same city)
SELECT c1.name AS customer1, c2.name AS customer2, c1.city
FROM customers c1
JOIN customers c2 ON c1.city = c2.city AND c1.customer_id < c2.customer_id;
```

### Python Example
```python
# Create sample data
orders_df = pd.DataFrame({
    'order_id': range(1, 301),
    'customer_id': np.random.randint(1, 101, 300),
    'order_date': pd.date_range('2024-01-01', periods=300, freq='D')[:300],
    'amount': np.random.uniform(10, 500, 300)
})

orders_df.to_sql('orders', conn, index=False, if_exists='replace')

# Customer order summary
query = """
SELECT c.name, c.city,
       COUNT(o.order_id) AS orders,
       SUM(o.amount) AS total_spent,
       AVG(o.amount) AS avg_order
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id
GROUP BY c.customer_id
ORDER BY total_spent DESC
LIMIT 10
"""

top_customers = pd.read_sql(query, conn)
print(top_customers)
```

---

# Lab 34: Subqueries & CTEs

## Duration: 60 minutes

### Exercise 1: Nested Queries
```sql
-- Subquery in WHERE
SELECT name FROM customers
WHERE customer_id IN (
    SELECT customer_id FROM orders
    WHERE amount > 200
);

-- Subquery in SELECT
SELECT name,
       total_purchases,
       (SELECT AVG(total_purchases) FROM customers) AS avg_purchases
FROM customers;

-- Correlated subquery
SELECT name,
       (SELECT COUNT(*) FROM orders o
        WHERE o.customer_id = c.customer_id) AS order_count
FROM customers c;

-- Common Table Expression (CTE)
WITH high_value_customers AS (
    SELECT customer_id, SUM(amount) AS total_spent
    FROM orders
    GROUP BY customer_id
    HAVING SUM(amount) > 1000
)
SELECT c.name, hvc.total_spent
FROM customers c
JOIN high_value_customers hvc ON c.customer_id = hvc.customer_id
ORDER BY hvc.total_spent DESC;

-- Multiple CTEs
WITH monthly_sales AS (
    SELECT DATE_TRUNC('month', order_date) AS month,
           SUM(amount) AS revenue
    FROM orders
    GROUP BY month
),
sales_growth AS (
    SELECT month, revenue,
           LAG(revenue) OVER (ORDER BY month) AS prev_month
    FROM monthly_sales
)
SELECT month, revenue,
       ((revenue - prev_month) / prev_month * 100) AS growth_pct
FROM sales_growth;
```

---

# Lab 35: Window Functions

## Duration: 75 minutes

### Exercise 1: Advanced Analytics
```sql
-- Running total
SELECT order_date, amount,
       SUM(amount) OVER (ORDER BY order_date) AS running_total
FROM orders;

-- Ranking
SELECT customer_id, amount,
       ROW_NUMBER() OVER (ORDER BY amount DESC) AS row_num,
       RANK() OVER (ORDER BY amount DESC) AS rank,
       DENSE_RANK() OVER (ORDER BY amount DESC) AS dense_rank
FROM orders;

-- Partition by
SELECT customer_id, order_date, amount,
       ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY order_date) AS order_num,
       SUM(amount) OVER (PARTITION BY customer_id ORDER BY order_date) AS customer_running_total
FROM orders;

-- Moving average
SELECT order_date, amount,
       AVG(amount) OVER (ORDER BY order_date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS ma7
FROM orders;

-- Lead/Lag
SELECT order_date, amount,
       LAG(amount) OVER (ORDER BY order_date) AS prev_day,
       LEAD(amount) OVER (ORDER BY order_date) AS next_day,
       amount - LAG(amount) OVER (ORDER BY order_date) AS change
FROM orders;
```

### Python Example
```python
# Window function simulation in Pandas
df = pd.read_sql("SELECT * FROM orders ORDER BY order_date", conn)

# Running total
df['running_total'] = df['amount'].cumsum()

# Ranking
df['rank'] = df['amount'].rank(ascending=False, method='min')

# Moving average
df['ma7'] = df['amount'].rolling(window=7, min_periods=1).mean()

# Lag
df['prev_amount'] = df['amount'].shift(1)
df['growth'] = df['amount'] - df['prev_amount']

print(df.head(20))
```

---

# Lab 36: Performance & Optimization

## Duration: 60 minutes

### Exercise 1: Indexes & Query Optimization
```sql
-- Create indexes
CREATE INDEX idx_customer_city ON customers(city);
CREATE INDEX idx_order_customer ON orders(customer_id);
CREATE INDEX idx_order_date ON orders(order_date);

-- Composite index
CREATE INDEX idx_order_customer_date ON orders(customer_id, order_date);

-- Analyze query performance
EXPLAIN ANALYZE
SELECT c.name, SUM(o.amount) as total
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
WHERE o.order_date >= '2024-01-01'
GROUP BY c.customer_id;

-- Optimization tips
-- 1. Use WHERE before JOIN when possible
-- 2. Limit result sets early
-- 3. Avoid SELECT *
-- 4. Use EXIST instead of IN for large subqueries

-- Efficient query
SELECT c.name, order_stats.total
FROM customers c
JOIN (
    SELECT customer_id, SUM(amount) AS total
    FROM orders
    WHERE order_date >= '2024-01-01'
    GROUP BY customer_id
) order_stats ON c.customer_id = order_stats.customer_id
WHERE order_stats.total > 500;
```

### Python: Query Optimization
```python
import time

# Inefficient: Multiple queries
start = time.time()
customers = pd.read_sql("SELECT * FROM customers", conn)
orders = pd.read_sql("SELECT * FROM orders", conn)
result = customers.merge(orders, on='customer_id')
print(f"Separate queries: {time.time() - start:.4f}s")

# Efficient: Single JOIN query
start = time.time()
result = pd.read_sql("""
    SELECT c.*, o.order_id, o.amount, o.order_date
    FROM customers c
    JOIN orders o ON c.customer_id = o.customer_id
""", conn)
print(f"Single JOIN: {time.time() - start:.4f}s")

# Use query parameters
customer_id = 5
result = pd.read_sql(
    "SELECT * FROM orders WHERE customer_id = ?",
    conn,
    params=(customer_id,)
)
```

---

## Deliverables (Labs 31-36)

1. ✅ Customer analytics dashboard (SQL)
2. ✅ Sales forecasting queries
3. ✅ Product recommendation engine
4. ✅ Churn analysis with CTEs
5. ✅ Performance optimization report

**Next:** Chapter 9 - Data Visualization
