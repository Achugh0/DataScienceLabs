# Lab 08: Descriptive Statistics Fundamentals

## Chapter 3: The Language of Data — Math and Stats for Beginners

### Learning Objectives
- Calculate and interpret measures of central tendency
- Understand measures of spread
- Create and interpret statistical summaries

### Duration
75 minutes

---

## Part 1: Measures of Central Tendency

### Exercise 1.1: Mean, Median, Mode
Calculate by hand, then verify with code:

**Dataset:** [5, 8, 8, 12, 15, 18, 20, 22, 25]

**By Hand:**
- Mean: _______________
- Median: _______________
- Mode: _______________

**Verify with Code:**
```python
import numpy as np
from scipy import stats

data = [5, 8, 8, 12, 15, 18, 20, 22, 25]

# Your code here
mean = _______________
median = _______________
mode = _______________

print(f"Mean: {mean}")
print(f"Median: {median}")
print(f"Mode: {mode}")
```

### Exercise 1.2: When to Use Which Measure
For each scenario, recommend the best measure and explain why:

| Scenario | Best Measure | Why? |
|----------|--------------|------|
| Average house price (with mansion outliers) | | |
| Most common shoe size to stock | | |
| Typical test score | | |
| Average salary (CEO makes 100x more) | | |
| Center of symmetric distribution | | |

### Exercise 1.3: Impact of Outliers
```python
import numpy as np
import matplotlib.pyplot as plt

# Dataset without outliers
normal_data = [45, 48, 50, 52, 55, 53, 51, 49, 54, 52]

# Dataset with outliers
outlier_data = [45, 48, 50, 52, 55, 53, 51, 49, 54, 200]

# Calculate mean and median for both
# Show how outliers affect each measure
# Visualize with box plots

# Your code here
```

**Analysis:**
- How much did the mean change? _______________
- How much did the median change? _______________
- Which is more robust? _______________

---

## Part 2: Measures of Spread

### Exercise 2.1: Range, IQR, Variance, Standard Deviation
For this dataset: [10, 15, 18, 22, 25, 28, 32, 35, 40]

**Calculate:**
```python
import numpy as np

data = [10, 15, 18, 22, 25, 28, 32, 35, 40]

# Range
range_val = _______________

# Interquartile Range (IQR)
q1 = _______________
q3 = _______________
iqr = _______________

# Variance
variance = _______________

# Standard Deviation
std = _______________

print(f"Range: {range_val}")
print(f"IQR: {iqr}")
print(f"Variance: {variance}")
print(f"Std Dev: {std}")
```

### Exercise 2.2: Interpreting Standard Deviation
Three datasets with same mean (50) but different spreads:

**Dataset A:** [48, 49, 50, 51, 52]
**Dataset B:** [30, 40, 50, 60, 70]
**Dataset C:** [10, 25, 50, 75, 90]

```python
# Calculate standard deviation for each
# Create visualizations
# Interpret the differences

# Your code here
```

**Interpretation:**
- Which has highest variability? _______________
- What does this tell you about the data? _______________
- When would high variability be concerning? _______________

### Exercise 2.3: Coefficient of Variation
Compare variability across different scales:

```python
# Product A: mean=$10, std=$2
# Product B: mean=$1000, std=$50

# Which is more variable?
# Calculate coefficient of variation (CV = std/mean * 100%)

cv_a = _______________
cv_b = _______________

# Your interpretation:
```

---

## Part 3: Distributions

### Exercise 3.1: Shape Description
Describe each distribution:

```python
import numpy as np
import matplotlib.pyplot as plt

# Create different distributions
normal = np.random.normal(50, 10, 1000)
skewed_right = np.random.exponential(2, 1000)
skewed_left = 100 - np.random.exponential(2, 1000)
bimodal = np.concatenate([np.random.normal(30, 5, 500), 
                           np.random.normal(70, 5, 500)])

# Plot histograms
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Your plotting code here

# For each, describe:
# - Shape (symmetric, skewed left/right)
# - Center
# - Spread
# - Outliers
```

**Descriptions:**
1. Normal: _______________
2. Skewed Right: _______________
3. Skewed Left: _______________
4. Bimodal: _______________

### Exercise 3.2: Empirical Rule
For a normal distribution with mean=100, std=15:

**68-95-99.7 Rule:**
- 68% of data falls between ___ and ___
- 95% of data falls between ___ and ___
- 99.7% of data falls between ___ and ___

**Verify with simulation:**
```python
import numpy as np

data = np.random.normal(100, 15, 10000)

# Calculate percentages within 1, 2, and 3 standard deviations
# Your code here
```

### Exercise 3.3: Identify Distribution Type
For each histogram, identify if it's:
- Normal
- Uniform
- Exponential
- Bimodal
- Skewed

```python
# Create mystery distributions
dist1 = np.random.uniform(0, 10, 1000)
dist2 = np.random.normal(50, 10, 1000)
dist3 = np.random.exponential(5, 1000)
dist4 = np.concatenate([np.random.normal(30, 3, 500),
                        np.random.normal(70, 3, 500)])

# Plot and identify
# Your code here
```

---

## Part 4: Percentiles and Quartiles

### Exercise 4.1: Calculate Percentiles
For the test score dataset:

```python
scores = [55, 62, 68, 71, 75, 78, 82, 85, 88, 91, 95]

# Calculate:
# - 25th percentile (Q1)
# - 50th percentile (Q2/median)
# - 75th percentile (Q3)
# - 90th percentile
# - 95th percentile

# Your code here using np.percentile()
```

**Interpretation:**
- A score of 85 is at what percentile? _______________
- What score is needed to be in the top 10%? _______________

### Exercise 4.2: Five Number Summary
```python
import numpy as np

data = [23, 25, 28, 29, 31, 32, 35, 38, 40, 42, 45, 48, 52, 55, 60]

# Calculate five-number summary
minimum = _______________
q1 = _______________
median = _______________
q3 = _______________
maximum = _______________

# Create box plot
import matplotlib.pyplot as plt
plt.boxplot(data)
plt.ylabel('Value')
plt.title('Five Number Summary')
plt.show()

# Identify outliers using IQR method
iqr = q3 - q1
lower_fence = q1 - 1.5 * iqr
upper_fence = q3 + 1.5 * iqr

outliers = [x for x in data if x < lower_fence or x > upper_fence]
print(f"Outliers: {outliers}")
```

---

## Part 5: Real-World Application

### Exercise 5.1: Salary Analysis
Analyze this salary data:

```python
import pandas as pd

salaries = {
    'employee': ['E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9', 'CEO'],
    'salary': [45000, 48000, 52000, 55000, 58000, 62000, 65000, 70000, 75000, 500000]
}
df = pd.DataFrame(salaries)

# Calculate all descriptive statistics
# Create visualizations
# Write interpretation for HR report

# Your analysis:
```

**HR Report:**
- What's the "typical" salary? _______________
- What measure should you use and why? _______________
- How spread out are salaries? _______________
- Are there outliers? _______________
- Recommendations: _______________

### Exercise 5.2: Product Performance
Compare performance of three products:

```python
product_a_sales = [120, 125, 118, 130, 122, 128, 119]
product_b_sales = [100, 150, 80, 170, 90, 160, 85]
product_c_sales = [115, 116, 114, 117, 115, 116, 115]

# Calculate descriptive stats for each
# Create comparison visualization
# Recommend which product is:
#   - Most consistent
#   - Highest average
#   - Most variable

# Your code and analysis here
```

**Business Recommendation:** _______________

---

## Part 6: Grouped Statistics

### Exercise 6.1: Statistics by Category
```python
import pandas as pd

df = pd.DataFrame({
    'region': ['North', 'North', 'South', 'South', 'East', 'East', 'West', 'West'],
    'sales': [100, 120, 90, 110, 130, 140, 95, 105],
    'customers': [50, 55, 45, 48, 60, 65, 47, 52]
})

# Calculate statistics by region
# Use groupby and agg
summary = df.groupby('region').agg({
    'sales': ['mean', 'median', 'std', 'min', 'max'],
    'customers': ['mean', 'median', 'std', 'min', 'max']
})

print(summary)

# Which region performs best?
# Which has most variability?
```

### Exercise 6.2: Comparative Box Plots
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Create box plots comparing groups
sns.boxplot(x='region', y='sales', data=df)
plt.title('Sales Distribution by Region')
plt.show()

# Interpret the differences
```

---

## Part 7: Data Profiling Practice

### Exercise 7.1: Complete Data Profile
Profile this dataset completely:

```python
import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')

# Create comprehensive summary:
# 1. For numerical columns: mean, median, std, min, max, quartiles
# 2. For categorical columns: unique values, mode, frequency
# 3. Missing values
# 4. Outliers

# Your comprehensive profiling code here

def profile_dataframe(df):
    """Generate complete statistical profile"""
    # Your implementation
    pass

profile = profile_dataframe(df)
```

---

## Deliverables

Submit:
1. ✅ Central tendency calculations (Exercise 1.1)
2. ✅ When to use which measure (Exercise 1.2)
3. ✅ Outlier impact analysis (Exercise 1.3)
4. ✅ Spread calculations (Exercise 2.1)
5. ✅ Standard deviation interpretation (Exercise 2.2)
6. ✅ Distribution descriptions (Exercise 3.1)
7. ✅ Empirical rule verification (Exercise 3.2)
8. ✅ Percentile calculations (Exercise 4.1)
9. ✅ Five number summary (Exercise 4.2)
10. ✅ Salary analysis report (Exercise 5.1)
11. ✅ Product performance analysis (Exercise 5.2)
12. ✅ Complete data profile (Exercise 7.1)

---

## Reflection Questions

1. Why is the median often better than the mean?
2. When is high variability good? When is it bad?
3. How do outliers affect your choice of statistics?

---

**Next Lab:** Lab 09 - Inferential Statistics Introduction
