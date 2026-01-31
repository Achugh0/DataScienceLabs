# Lab 37-42: Matplotlib & Seaborn - Data Visualization Mastery

## Chapter 9-10: Data Visualization

### 6 Comprehensive Visualization Labs

---

# Lab 37: Matplotlib Fundamentals

## Duration: 60 minutes

### Exercise 1: Line Plots
```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Daily sales data
dates = pd.date_range('2024-01-01', periods=90)
sales = np.random.normal(1000, 200, 90) + np.linspace(0, 500, 90)

plt.figure(figsize=(12, 6))
plt.plot(dates, sales, linewidth=2, color='#2ecc71', label='Daily Sales')
plt.title('Sales Trend - Q1 2024', fontsize=16, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Sales ($)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Multiple lines
products = {
    'Product A': np.random.normal(1000, 100, 90),
    'Product B': np.random.normal(1500, 150, 90),
    'Product C': np.random.normal(800, 80, 90)
}

plt.figure(figsize=(12, 6))
for product, values in products.items():
    plt.plot(dates, values, label=product, linewidth=2)

plt.title('Product Sales Comparison')
plt.xlabel('Date')
plt.ylabel('Sales ($)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### Exercise 2: Bar Charts
```python
# Regional sales
regions = ['North', 'South', 'East', 'West']
sales = [12500, 15800, 14200, 13900]
colors = ['#3498db', '#e74c3c', '#f39c12', '#2ecc71']

plt.figure(figsize=(10, 6))
bars = plt.bar(regions, sales, color=colors, edgecolor='black', linewidth=1.5)

# Add value labels
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height,
             f'${height:,.0f}',
             ha='center', va='bottom', fontweight='bold')

plt.title('Sales by Region', fontsize=16)
plt.ylabel('Revenue ($)', fontsize=12)
plt.ylim(0, max(sales) * 1.1)
plt.show()

# Grouped bar chart
products = ['A', 'B', 'C']
q1 = [100, 150, 200]
q2 = [120, 160, 210]
q3 = [110, 155, 205]

x = np.arange(len(products))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x - width, q1, width, label='Q1', color='#3498db')
ax.bar(x, q2, width, label='Q2', color='#e74c3c')
ax.bar(x + width, q3, width, label='Q3', color='#f39c12')

ax.set_xticks(x)
ax.set_xticklabels(products)
ax.legend()
ax.set_title('Quarterly Sales by Product')
plt.show()
```

---

# Lab 38: Advanced Matplotlib

## Duration: 60 minutes

### Exercise 1: Subplots & Layouts
```python
# 2x2 grid
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Sales trend
axes[0, 0].plot(dates, sales, color='#2ecc71')
axes[0, 0].set_title('Sales Trend')
axes[0, 0].grid(True)

# Histogram
axes[0, 1].hist(sales, bins=20, color='#3498db', edgecolor='black')
axes[0, 1].set_title('Sales Distribution')
axes[0, 1].set_xlabel('Sales ($)')

# Scatter
advertising = np.random.uniform(1000, 5000, 90)
axes[1, 0].scatter(advertising, sales, alpha=0.6, s=100, c='#e74c3c')
axes[1, 0].set_title('Sales vs Advertising')
axes[1, 0].set_xlabel('Ad Spend ($)')
axes[1, 0].set_ylabel('Sales ($)')

# Pie chart
labels = ['Online', 'Store', 'Phone', 'Other']
sizes = [45, 30, 15, 10]
axes[1, 1].pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
axes[1, 1].set_title('Sales Channels')

plt.tight_layout()
plt.show()
```

### Exercise 2: Customization
```python
# Professional style
plt.style.use('seaborn-v0_8-darkgrid')

fig, ax = plt.subplots(figsize=(12, 6))

# Plot with custom styling
ax.plot(dates, sales, linewidth=3, color='#e74c3c', label='Actual')
ma7 = pd.Series(sales).rolling(window=7).mean()
ax.plot(dates, ma7, linewidth=2, color='#3498db', linestyle='--', label='7-day MA')

# Fill between
ax.fill_between(dates, sales, ma7, alpha=0.2, color='gray')

# Annotations
max_idx = sales.argmax()
ax.annotate(f'Peak: ${sales[max_idx]:.0f}',
            xy=(dates[max_idx], sales[max_idx]),
            xytext=(dates[max_idx + 10], sales[max_idx] + 200),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=12, fontweight='bold')

ax.set_title('Sales Analysis with Moving Average', fontsize=16, pad=20)
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Sales ($)', fontsize=12)
ax.legend(loc='upper left', fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

# Lab 39: Seaborn - Statistical Visualization

## Duration: 75 minutes

### Exercise 1: Distribution Plots
```python
import seaborn as sns

# Generate customer data
np.random.seed(42)
customers = pd.DataFrame({
    'age': np.random.normal(40, 15, 1000),
    'income': np.random.normal(60000, 20000, 1000),
    'purchases': np.random.poisson(10, 1000),
    'segment': np.random.choice(['A', 'B', 'C'], 1000)
})

# Histogram with KDE
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.histplot(customers['age'], kde=True, color='#3498db', bins=30)
plt.title('Age Distribution')
plt.xlabel('Age')

plt.subplot(1, 2, 2)
sns.histplot(customers['income'], kde=True, color='#2ecc71', bins=30)
plt.title('Income Distribution')
plt.xlabel('Income ($)')

plt.tight_layout()
plt.show()

# Box plots by segment
plt.figure(figsize=(14, 5))

plt.subplot(1, 3, 1)
sns.boxplot(data=customers, x='segment', y='age', palette='Set2')
plt.title('Age by Segment')

plt.subplot(1, 3, 2)
sns.boxplot(data=customers, x='segment', y='income', palette='Set2')
plt.title('Income by Segment')

plt.subplot(1, 3, 3)
sns.boxplot(data=customers, x='segment', y='purchases', palette='Set2')
plt.title('Purchases by Segment')

plt.tight_layout()
plt.show()

# Violin plots
plt.figure(figsize=(12, 6))
sns.violinplot(data=customers, x='segment', y='income', hue='segment', palette='muted')
plt.title('Income Distribution by Segment (Violin Plot)')
plt.show()
```

### Exercise 2: Relationships
```python
# Scatter with regression
plt.figure(figsize=(10, 6))
sns.scatterplot(data=customers, x='age', y='income', hue='segment', 
                size='purchases', sizes=(20, 200), alpha=0.6)
sns.regplot(data=customers, x='age', y='income', scatter=False, color='red')
plt.title('Age vs Income by Segment')
plt.show()

# Pair plot
sns.pairplot(customers, hue='segment', diag_kind='kde', palette='husl')
plt.suptitle('Customer Attributes Pair Plot', y=1.02)
plt.show()

# Joint plot
sns.jointplot(data=customers, x='age', y='income', kind='hex', color='#3498db')
plt.show()
```

---

# Lab 40: Heatmaps & Correlation Matrices

## Duration: 60 minutes

### Exercise 1: Correlation Analysis
```python
# Correlation matrix
corr_matrix = customers[['age', 'income', 'purchases']].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Feature Correlation Matrix', fontsize=14, pad=20)
plt.show()

# Sales heatmap
sales_data = pd.DataFrame(
    np.random.randint(50, 200, (12, 7)),
    index=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
    columns=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
)

plt.figure(figsize=(12, 8))
sns.heatmap(sales_data, annot=True, fmt='d', cmap='YlGnBu',
            linewidths=0.5, cbar_kws={'label': 'Sales'})
plt.title('Sales Heatmap: Day of Week by Month', fontsize=16)
plt.xlabel('Day of Week')
plt.ylabel('Month')
plt.show()
```

---

# Lab 41: Time Series Visualization

## Duration: 60 minutes

### Exercise 1: Advanced Time Plots
```python
# Generate time series
dates = pd.date_range('2023-01-01', '2024-12-31', freq='D')
trend = np.linspace(1000, 2000, len(dates))
seasonal = 300 * np.sin(2 * np.pi * np.arange(len(dates)) / 365)
noise = np.random.normal(0, 100, len(dates))
sales = trend + seasonal + noise

ts_df = pd.DataFrame({
    'date': dates,
    'sales': sales
})
ts_df = ts_df.set_index('date')

# Decomposition plot
fig, axes = plt.subplots(4, 1, figsize=(14, 10))

# Original
axes[0].plot(ts_df.index, ts_df['sales'], color='#2c3e50')
axes[0].set_title('Original Time Series', fontsize=14)
axes[0].grid(True, alpha=0.3)

# Trend
trend_line = ts_df['sales'].rolling(window=30).mean()
axes[1].plot(ts_df.index, trend_line, color='#e74c3c', linewidth=2)
axes[1].set_title('Trend (30-day MA)', fontsize=14)
axes[1].grid(True, alpha=0.3)

# Seasonal
seasonal_component = ts_df['sales'] - trend_line
axes[2].plot(ts_df.index, seasonal_component, color='#3498db', alpha=0.7)
axes[2].set_title('Seasonal Component', fontsize=14)
axes[2].grid(True, alpha=0.3)

# Residual
residual = ts_df['sales'] - trend_line - seasonal_component.fillna(0)
axes[3].plot(ts_df.index, residual, color='#95a5a6', alpha=0.5)
axes[3].set_title('Residual', fontsize=14)
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

# Lab 42: Interactive Dashboards with Plotly

## Duration: 75 minutes

### Exercise 1: Interactive Plots
```python
import plotly.express as px
import plotly.graph_objects as go

# Interactive line chart
fig = px.line(ts_df.reset_index(), x='date', y='sales',
              title='Interactive Sales Trend')
fig.update_traces(line_color='#2ecc71', line_width=2)
fig.update_layout(hovermode='x unified')
fig.show()

# Interactive scatter
fig = px.scatter(customers, x='age', y='income', color='segment',
                 size='purchases', hover_data=['purchases'],
                 title='Customer Segmentation')
fig.show()

# 3D scatter
fig = px.scatter_3d(customers, x='age', y='income', z='purchases',
                    color='segment', size='purchases',
                    title='3D Customer Analysis')
fig.show()

# Animated bar chart (sales by month)
monthly_sales = pd.DataFrame({
    'month': pd.date_range('2024-01', periods=12, freq='M'),
    'product_a': np.random.randint(100, 300, 12),
    'product_b': np.random.randint(150, 350, 12)
})

fig = go.Figure()
fig.add_trace(go.Bar(x=monthly_sales['month'], y=monthly_sales['product_a'],
                     name='Product A', marker_color='#3498db'))
fig.add_trace(go.Bar(x=monthly_sales['month'], y=monthly_sales['product_b'],
                     name='Product B', marker_color='#e74c3c'))

fig.update_layout(
    title='Monthly Product Sales',
    xaxis_title='Month',
    yaxis_title='Sales',
    barmode='group'
)
fig.show()
```

---

## Deliverables (Labs 37-42)

1. ✅ Sales dashboard with matplotlib
2. ✅ Statistical analysis plots with seaborn
3. ✅ Correlation heatmaps
4. ✅ Time series decomposition
5. ✅ Interactive dashboard with plotly

**Next:** Chapter 11-12 - Machine Learning
