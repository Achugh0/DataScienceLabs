# Lab 11: Probability Fundamentals and Distributions

## Chapter 3: The Language of Data — Math and Stats for Beginners

### Learning Objectives
- Understand probability concepts
- Work with common probability distributions
- Apply probability to data science problems

### Duration
75 minutes

---

## Part 1: Basic Probability

### Exercise 1.1: Calculate Probabilities
For a standard deck of 52 cards:

1. P(drawing an Ace) = _______________
2. P(drawing a Heart) = _______________
3. P(drawing a Face Card) = _______________
4. P(drawing Red Card) = _______________
5. P(drawing Ace AND Heart) = _______________
6. P(drawing Ace OR Heart) = _______________

**Verify with simulation:**
```python
import random
import numpy as np

# Create deck
suits = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
deck = [(rank, suit) for suit in suits for rank in ranks]

# Simulate 10,000 draws
num_simulations = 10000
ace_count = 0
heart_count = 0
# Your code here
```

### Exercise 1.2: Conditional Probability
Given:
- P(Rain) = 0.3
- P(Traffic | Rain) = 0.7
- P(Traffic | No Rain) = 0.2

**Calculate:**
1. P(Traffic AND Rain) = _______________
2. P(Traffic) = _______________
3. P(Rain | Traffic) = _______________ (Bayes' Theorem)

**Show your work:**

### Exercise 1.3: Independence
Determine if events are independent:

**Scenario 1:** Rolling two dice
- Event A: First die shows 6
- Event B: Second die shows 6
- Independent? _______________
- Why? _______________

**Scenario 2:** Drawing two cards without replacement
- Event A: First card is Ace
- Event B: Second card is Ace
- Independent? _______________
- Why? _______________

---

## Part 2: Probability Distributions

### Exercise 2.1: Binomial Distribution
```python
from scipy.stats import binom
import matplotlib.pyplot as plt
import numpy as np

# Scenario: Flip a coin 10 times
n = 10  # trials
p = 0.5  # probability of success

# Calculate probabilities
k = np.arange(0, n+1)
probabilities = binom.pmf(k, n, p)

# Plot
plt.bar(k, probabilities)
plt.xlabel('Number of Heads')
plt.ylabel('Probability')
plt.title('Binomial Distribution (n=10, p=0.5)')
plt.show()

# Questions:
# 1. What's the probability of exactly 5 heads?
p_5_heads = _______________

# 2. What's the probability of at least 7 heads?
p_at_least_7 = _______________

# 3. What's the expected number of heads?
expected = _______________
```

### Exercise 2.2: Normal Distribution
```python
from scipy.stats import norm

# Test scores: mean=75, std=10
mean = 75
std = 10

# 1. What percentage score above 85?
p_above_85 = _______________

# 2. What score is the 90th percentile?
score_90th = _______________

# 3. What's the probability of scoring between 70 and 80?
p_between = _______________

# Visualize
x = np.linspace(40, 110, 1000)
y = norm.pdf(x, mean, std)
plt.plot(x, y)
plt.fill_between(x, y, where=(x>=70) & (x<=80), alpha=0.3)
plt.xlabel('Score')
plt.ylabel('Density')
plt.title('Normal Distribution')
plt.show()
```

### Exercise 2.3: Poisson Distribution
```python
from scipy.stats import poisson

# Average 3 customers per hour
lambda_rate = 3

# 1. Probability of exactly 5 customers in an hour?
p_5 = _______________

# 2. Probability of at least 1 customer?
p_at_least_1 = _______________

# 3. Probability of 0 customers?
p_0 = _______________

# Plot distribution
k = np.arange(0, 15)
probs = poisson.pmf(k, lambda_rate)
plt.bar(k, probs)
plt.xlabel('Number of Customers')
plt.ylabel('Probability')
plt.title(f'Poisson Distribution (λ={lambda_rate})')
plt.show()
```

---

## Part 3: Real-World Applications

### Exercise 3.1: Medical Testing
A disease affects 1% of population. Test has:
- Sensitivity (True Positive Rate): 95%
- Specificity (True Negative Rate): 90%

**Calculate using Bayes' Theorem:**
If you test positive, what's probability you have disease?

```python
# Your calculation
p_disease = 0.01
p_no_disease = 0.99
p_positive_given_disease = 0.95
p_positive_given_no_disease = 0.10

# P(Disease | Positive) = ?
p_disease_given_positive = _______________

# Show your work
```

**Answer:** _______________
**What does this mean?** _______________

### Exercise 3.2: A/B Test Planning
Planning an A/B test:
- Current conversion rate: 10%
- Want to detect 20% relative improvement (to 12%)
- Significance level: 0.05
- Power: 0.80

```python
from statsmodels.stats.power import zt_ind_solve_power
from statsmodels.stats.proportion import proportion_effectsize

# Calculate required sample size
baseline = 0.10
target = 0.12

effect_size = proportion_effectsize(baseline, target)
n_per_group = zt_ind_solve_power(effect_size=effect_size, 
                                  alpha=0.05, 
                                  power=0.80)

print(f"Required sample size per group: {np.ceil(n_per_group)}")

# Your interpretation:
```

### Exercise 3.3: Risk Assessment
Calculate expected value:

**Scenario:** Investment opportunity
- 60% chance of $1000 profit
- 30% chance of $0 (break even)
- 10% chance of $500 loss

**Expected Value:**
```python
outcomes = [1000, 0, -500]
probabilities = [0.6, 0.3, 0.1]

expected_value = _______________

# Should you invest?
```

---

## Part 4: Monte Carlo Simulation

### Exercise 4.1: Estimate Pi
```python
import numpy as np
import matplotlib.pyplot as plt

# Simulate throwing darts at a square with inscribed circle
n_simulations = 10000

# Generate random points
x = np.random.uniform(-1, 1, n_simulations)
y = np.random.uniform(-1, 1, n_simulations)

# Check if inside circle
inside_circle = (x**2 + y**2) <= 1

# Estimate pi
pi_estimate = 4 * np.sum(inside_circle) / n_simulations
print(f"Estimated π: {pi_estimate:.4f}")
print(f"Actual π: {np.pi:.4f}")
print(f"Error: {abs(pi_estimate - np.pi):.4f}")

# Visualize
plt.figure(figsize=(8, 8))
plt.scatter(x[inside_circle], y[inside_circle], c='blue', s=1, alpha=0.5)
plt.scatter(x[~inside_circle], y[~inside_circle], c='red', s=1, alpha=0.5)
plt.axis('equal')
plt.title(f'Monte Carlo π Estimation: {pi_estimate:.4f}')
plt.show()
```

### Exercise 4.2: Portfolio Simulation
```python
# Simulate stock portfolio returns
np.random.seed(42)

# Assume annual return: 8%, std: 15%
n_years = 30
n_simulations = 1000
initial_investment = 10000

returns_matrix = np.random.normal(0.08, 0.15, (n_simulations, n_years))

# Calculate portfolio values over time
portfolio_values = np.zeros((n_simulations, n_years + 1))
portfolio_values[:, 0] = initial_investment

for year in range(1, n_years + 1):
    portfolio_values[:, year] = portfolio_values[:, year-1] * (1 + returns_matrix[:, year-1])

# Plot percentiles
plt.figure(figsize=(12, 6))
plt.plot(portfolio_values.T, alpha=0.01, color='blue')
plt.plot(np.median(portfolio_values, axis=0), label='Median', linewidth=2, color='red')
plt.fill_between(range(n_years + 1), 
                 np.percentile(portfolio_values, 25, axis=0),
                 np.percentile(portfolio_values, 75, axis=0),
                 alpha=0.3, label='25-75 percentile')
plt.xlabel('Year')
plt.ylabel('Portfolio Value ($)')
plt.title('Portfolio Simulation (30 Years)')
plt.legend()
plt.show()

# Analysis:
# - Median final value: _______________
# - 10th percentile: _______________
# - 90th percentile: _______________
```

---

## Deliverables

Submit:
1. ✅ Card probability calculations (Exercise 1.1)
2. ✅ Conditional probability problems (Exercise 1.2)
3. ✅ Distribution plots and calculations (Exercises 2.1-2.3)
4. ✅ Medical test Bayes' calculation (Exercise 3.1)
5. ✅ A/B test sample size (Exercise 3.2)
6. ✅ Monte Carlo pi estimation (Exercise 4.1)
7. ✅ Portfolio simulation analysis (Exercise 4.2)

---

**Next Lab:** Lab 12 - A/B Testing Case Study
