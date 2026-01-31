# Lab 23: Advanced NumPy - Linear Algebra & Statistics

## Chapter 6: NumPy for Data Science

### Learning Objectives
- Perform matrix operations for data science
- Calculate correlations and covariance
- Implement dimensionality reduction

### Duration: 75 minutes

---

## Part 1: Matrix Operations

### Exercise 1.1: Dot Products & Matrix Multiplication
```python
import numpy as np

# Dot product (1D)
prices = np.array([29.99, 39.99, 49.99])
quantities = np.array([10, 5, 8])
total_revenue = np.dot(prices, quantities)
print(f"Total revenue: ${total_revenue:.2f}")

# Matrix multiplication
# Sales: 3 products × 4 days
sales = np.array([
    [10, 12, 8, 15],
    [5, 6, 4, 7],
    [8, 9, 10, 11]
])

# Prices: 3 products × 1
prices = np.array([[29.99], [39.99], [49.99]])

# Daily revenue: (3 × 4) @ (3 × 1)^T = (4, 1)
daily_revenue = sales.T @ prices
print(f"Daily revenue:\n{daily_revenue}")

# Multiple matrix multiplication
# (customers × products) @ (products × features)
customer_purchases = np.random.randint(0, 5, (100, 10))
product_features = np.random.random((10, 5))
customer_features = customer_purchases @ product_features
print(f"Customer features shape: {customer_features.shape}")

# YOUR CODE: Calculate
# 1. Revenue by product by region (3D)
# 2. Weighted average ratings
# 3. Portfolio returns (weights × returns)
```

### Exercise 1.2: Matrix Properties
```python
# Create matrix
A = np.array([[1, 2], [3, 4]])

# Transpose
A_T = A.T
print(f"Transpose:\n{A_T}")

# Inverse
A_inv = np.linalg.inv(A)
print(f"Inverse:\n{A_inv}")

# Verify: A @ A_inv = Identity
identity = A @ A_inv
print(f"A @ A_inv:\n{identity}")

# Determinant
det = np.linalg.det(A)
print(f"Determinant: {det}")

# Rank
rank = np.linalg.matrix_rank(A)
print(f"Rank: {rank}")

# Practical: Solve linear equations
# 2x + 3y = 13
# 5x + 4y = 22
A = np.array([[2, 3], [5, 4]])
b = np.array([13, 22])
solution = np.linalg.solve(A, b)
print(f"Solution: x={solution[0]}, y={solution[1]}")

# YOUR CODE: Solve
# Supply chain optimization
# Calculate optimal production quantities
```

---

## Part 2: Eigenvalues & Eigenvectors

### Exercise 2.1: Principal Components
```python
# Customer data: 1000 customers × 5 features
np.random.seed(42)
data = np.random.randn(1000, 5)

# Center data
data_centered = data - data.mean(axis=0)

# Covariance matrix
cov_matrix = np.cov(data_centered, rowvar=False)
print(f"Covariance matrix shape: {cov_matrix.shape}")

# Eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Sort by eigenvalues
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# Explained variance
explained_var = eigenvalues / eigenvalues.sum()
print(f"Explained variance: {explained_var}")
print(f"First 2 components explain: {explained_var[:2].sum():.1%}")

# Project to 2D
data_2d = data_centered @ eigenvectors[:, :2]
print(f"Reduced data shape: {data_2d.shape}")

# YOUR CODE: Implement PCA class
class PCA:
    def fit(self, X, n_components=2):
        """Fit PCA model"""
        pass
    
    def transform(self, X):
        """Transform data"""
        pass
```

### Exercise 2.2: Singular Value Decomposition
```python
# User-item ratings matrix (sparse)
ratings = np.array([
    [5, 3, 0, 1],
    [4, 0, 0, 1],
    [1, 1, 0, 5],
    [1, 0, 0, 4],
    [0, 1, 5, 4],
])

# SVD
U, s, Vt = np.linalg.svd(ratings, full_matrices=False)

print(f"U shape: {U.shape}")  # Users × concepts
print(f"s shape: {s.shape}")  # Singular values
print(f"Vt shape: {Vt.shape}")  # Concepts × items

# Reconstruct with fewer components
k = 2  # Keep top 2 components
S_k = np.diag(s[:k])
reconstructed = U[:, :k] @ S_k @ Vt[:k, :]

print(f"Original:\n{ratings}")
print(f"Reconstructed:\n{reconstructed.round(2)}")

# Fill missing ratings
print(f"Predicted rating [0, 2]: {reconstructed[0, 2]:.2f}")

# YOUR CODE: Build recommender
# - Predict all missing ratings
# - Recommend top 3 items per user
# - Evaluate MAE
```

---

## Part 3: Statistics & Probability

### Exercise 3.1: Correlation Analysis
```python
# Generate correlated data
np.random.seed(42)
n = 100

# Sales features
advertising_spend = np.random.uniform(1000, 5000, n)
sales = 50 * advertising_spend + np.random.normal(0, 5000, n)
customer_visits = 0.5 * advertising_spend + np.random.normal(0, 200, n)

# Combine into matrix
data = np.column_stack([advertising_spend, sales, customer_visits])

# Correlation matrix
correlation = np.corrcoef(data.T)
print(f"Correlation matrix:\n{correlation}")

# Find strong correlations
high_corr = np.abs(correlation) > 0.8
np.fill_diagonal(high_corr, False)  # Ignore diagonal
print(f"Strong correlations:\n{high_corr}")

# Practical application
features = ['Ad Spend', 'Sales', 'Visits']
for i in range(len(features)):
    for j in range(i + 1, len(features)):
        corr = correlation[i, j]
        if abs(corr) > 0.7:
            print(f"{features[i]} <-> {features[j]}: {corr:.3f}")

# YOUR CODE: Analyze
# - Remove redundant features
# - Find leading indicators
# - Detect multicollinearity
```

### Exercise 3.2: Hypothesis Testing
```python
# A/B test data
group_a = np.random.normal(100, 15, 1000)  # Control
group_b = np.random.normal(105, 15, 1000)  # Treatment

# T-test components
mean_a = group_a.mean()
mean_b = group_b.mean()
std_a = group_a.std()
std_b = group_b.std()
n_a = len(group_a)
n_b = len(group_b)

# Pooled standard error
se = np.sqrt(std_a**2/n_a + std_b**2/n_b)

# T-statistic
t_stat = (mean_b - mean_a) / se
print(f"T-statistic: {t_stat:.3f}")

# Effect size (Cohen's d)
pooled_std = np.sqrt((std_a**2 + std_b**2) / 2)
cohens_d = (mean_b - mean_a) / pooled_std
print(f"Effect size: {cohens_d:.3f}")

# Bootstrap confidence interval
def bootstrap_mean_diff(a, b, n_bootstrap=10000):
    """Bootstrap 95% CI for mean difference"""
    diffs = []
    for _ in range(n_bootstrap):
        sample_a = np.random.choice(a, size=len(a), replace=True)
        sample_b = np.random.choice(b, size=len(b), replace=True)
        diffs.append(sample_b.mean() - sample_a.mean())
    
    diffs = np.array(diffs)
    ci_lower = np.percentile(diffs, 2.5)
    ci_upper = np.percentile(diffs, 97.5)
    return ci_lower, ci_upper

ci = bootstrap_mean_diff(group_a, group_b)
print(f"95% CI: [{ci[0]:.2f}, {ci[1]:.2f}]")

# YOUR CODE: Implement
# - Chi-square test for categorical data
# - ANOVA for multiple groups
# - Power analysis
```

---

## Part 4: Time Series Operations

### Exercise 4.1: Moving Statistics
```python
# Daily sales for 90 days
np.random.seed(42)
sales = np.random.normal(1000, 200, 90) + np.linspace(0, 500, 90)

def moving_average(data, window):
    """Calculate moving average"""
    return np.convolve(data, np.ones(window)/window, mode='valid')

def moving_std(data, window):
    """Calculate moving standard deviation"""
    result = []
    for i in range(len(data) - window + 1):
        result.append(data[i:i+window].std())
    return np.array(result)

# Calculate
ma_7 = moving_average(sales, 7)
ma_30 = moving_average(sales, 30)
std_7 = moving_std(sales, 7)

print(f"7-day MA (last 5): {ma_7[-5:]}")
print(f"30-day MA (last 5): {ma_30[-5:]}")

# Bollinger Bands
upper_band = ma_7 + 2 * std_7
lower_band = ma_7 - 2 * std_7

# Detect anomalies
anomalies = (sales[6:] > upper_band) | (sales[6:] < lower_band)
print(f"Anomalies detected: {anomalies.sum()}")

# YOUR CODE: Calculate
# - Exponential moving average
# - Rate of change
# - Relative strength index
```

### Exercise 4.2: Trend Analysis
```python
# Decompose time series
def decompose_trend(data, window=12):
    """Simple trend decomposition"""
    # Trend (moving average)
    trend = moving_average(data, window)
    
    # Pad trend to match original length
    pad_size = len(data) - len(trend)
    trend_padded = np.pad(trend, (pad_size//2, pad_size - pad_size//2), 
                         mode='edge')
    
    # Detrend
    detrended = data - trend_padded
    
    # Seasonal (repeating pattern)
    seasonal_period = 7  # Weekly
    seasonal = np.zeros(len(data))
    for i in range(seasonal_period):
        seasonal[i::seasonal_period] = detrended[i::seasonal_period].mean()
    
    # Residual
    residual = data - trend_padded - seasonal
    
    return trend_padded, seasonal, residual

trend, seasonal, residual = decompose_trend(sales)

print(f"Trend range: {trend.min():.0f} - {trend.max():.0f}")
print(f"Seasonal range: {seasonal.min():.2f} - {seasonal.max():.2f}")
print(f"Residual std: {residual.std():.2f}")

# Forecast next 7 days
last_trend = trend[-1]
forecast = last_trend + seasonal[-7:]
print(f"7-day forecast: {forecast}")

# YOUR CODE: Implement
# - Seasonal adjustment
# - Growth rate calculation
# - Forecast confidence intervals
```

---

## Part 5: Optimization & Root Finding

### Exercise 5.1: Optimization
```python
# Find optimal pricing
def profit_function(price):
    """Profit = (price - cost) * demand(price)"""
    cost = 10
    demand = 1000 - 20 * price  # Linear demand
    return (price - cost) * demand

# Grid search
prices = np.linspace(10, 50, 100)
profits = np.array([profit_function(p) for p in prices])
optimal_idx = profits.argmax()
optimal_price = prices[optimal_idx]
optimal_profit = profits[optimal_idx]

print(f"Optimal price: ${optimal_price:.2f}")
print(f"Maximum profit: ${optimal_profit:.2f}")

# Derivative (numerical)
def derivative(func, x, h=0.01):
    """Numerical derivative"""
    return (func(x + h) - func(x - h)) / (2 * h)

# Find where derivative = 0
derivatives = np.array([derivative(profit_function, p) for p in prices])
zero_crossings = np.where(np.diff(np.sign(derivatives)))[0]
print(f"Derivative zero at: ${prices[zero_crossings]}")

# YOUR CODE: Optimize
# - Production quantity
# - Ad spend allocation
# - Inventory levels
```

### Exercise 5.2: Polynomial Fitting
```python
# Fit polynomial to data
x = np.arange(12)  # Months
y = np.array([100, 120, 115, 140, 160, 155, 180, 190, 185, 210, 230, 225])

# Fit degree 2 polynomial
coeffs = np.polyfit(x, y, deg=2)
print(f"Coefficients: {coeffs}")

# Predict
poly = np.poly1d(coeffs)
predictions = poly(x)
future = poly(np.array([12, 13, 14]))  # Next 3 months

print(f"R-squared: {1 - ((y - predictions)**2).sum() / ((y - y.mean())**2).sum():.3f}")
print(f"Next 3 months forecast: {future}")

# Residual analysis
residuals = y - predictions
print(f"Residual std: {residuals.std():.2f}")
print(f"Mean residual: {residuals.mean():.2f}")

# YOUR CODE: Compare models
# - Linear, quadratic, cubic fits
# - Calculate AIC/BIC
# - Cross-validation
```

---

## Part 6: Real-World Application

### Exercise 6.1: Customer Churn Prediction
```python
# Customer features
np.random.seed(42)
n_customers = 1000

features = {
    'tenure': np.random.randint(1, 60, n_customers),
    'monthly_spend': np.random.uniform(20, 200, n_customers),
    'support_calls': np.random.poisson(2, n_customers),
    'satisfaction': np.random.uniform(1, 5, n_customers),
}

# Create feature matrix
X = np.column_stack([
    features['tenure'],
    features['monthly_spend'],
    features['support_calls'],
    features['satisfaction']
])

# Generate churn (higher support calls, lower satisfaction = churn)
churn_prob = 1 / (1 + np.exp(-(
    -0.05 * features['tenure'] +
    0.01 * features['support_calls'] -
    0.5 * features['satisfaction'] +
    2
)))
y = (np.random.random(n_customers) < churn_prob).astype(int)

print(f"Churn rate: {y.mean():.1%}")

# Simple logistic regression (manual)
# Normalize features
X_norm = (X - X.mean(axis=0)) / X.std(axis=0)

# Add intercept
X_norm = np.column_stack([np.ones(n_customers), X_norm])

# Initialize weights
weights = np.zeros(X_norm.shape[1])

# Gradient descent
learning_rate = 0.01
n_iterations = 1000

for _ in range(n_iterations):
    # Predictions
    z = X_norm @ weights
    predictions = 1 / (1 + np.exp(-z))
    
    # Gradient
    gradient = X_norm.T @ (predictions - y) / n_customers
    
    # Update
    weights -= learning_rate * gradient

# Final predictions
final_pred = 1 / (1 + np.exp(-(X_norm @ weights)))
accuracy = ((final_pred > 0.5) == y).mean()
print(f"Accuracy: {accuracy:.1%}")

# YOUR CODE: Add
# - Feature importance
# - Confusion matrix
# - ROC curve
```

---

## Deliverables

1. ✅ Matrix operations for data analysis
2. ✅ PCA dimensionality reduction
3. ✅ Correlation analysis
4. ✅ Time series forecasting
5. ✅ Optimization problems
6. ✅ Predictive model (logistic regression)

---

## Challenge: Build Complete ML Pipeline

```python
class DataPipeline:
    """End-to-end ML pipeline using only NumPy"""
    
    def __init__(self):
        pass
    
    def load_data(self, X, y):
        """Load and validate data"""
        pass
    
    def split_data(self, test_size=0.2):
        """Train/test split"""
        pass
    
    def normalize(self):
        """Feature scaling"""
        pass
    
    def handle_missing(self):
        """Impute missing values"""
        pass
    
    def feature_engineering(self):
        """Create new features"""
        pass
    
    def train_model(self):
        """Train logistic regression"""
        pass
    
    def evaluate(self):
        """Calculate metrics"""
        pass
    
    def cross_validate(self, k=5):
        """K-fold CV"""
        pass
```

**Next Lab:** Lab 24 - Pandas DataFrames
