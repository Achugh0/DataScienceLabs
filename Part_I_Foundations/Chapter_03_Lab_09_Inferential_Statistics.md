# Lab 09: Inferential Statistics and Hypothesis Testing

## Chapter 3: The Language of Data — Math and Stats for Beginners

### Learning Objectives
- Understand populations vs samples
- Conduct hypothesis tests
- Interpret p-values and confidence intervals

### Duration
90 minutes

---

## Part 1: Populations vs Samples

### Exercise 1.1: Identify Population and Sample
For each scenario, identify the population and sample:

| Scenario | Population | Sample | Sampling Method |
|----------|------------|--------|-----------------|
| Survey 1000 voters to predict election | | | |
| Test 50 light bulbs from factory | | | |
| Measure height of 100 students in university | | | |
| Analyze 10% of customer transactions | | | |

### Exercise 1.2: Sampling Methods
Match scenarios to sampling methods:

**Methods:** Simple Random, Stratified, Cluster, Systematic, Convenience

| Scenario | Method | Pros | Cons |
|----------|--------|------|------|
| Every 10th customer | | | |
| Random selection ensuring equal gender representation | | | |
| First 100 people you meet | | | |
| Randomly select 5 stores, survey all customers | | | |
| Draw names from hat | | | |

### Exercise 1.3: Simulate Sampling
```python
import numpy as np
import matplotlib.pyplot as plt

# Create a population
population = np.random.normal(100, 15, 10000)
true_mean = population.mean()
print(f"True population mean: {true_mean:.2f}")

# Take multiple samples
sample_sizes = [10, 50, 100, 500]
num_samples = 1000

for n in sample_sizes:
    sample_means = []
    for _ in range(num_samples):
        sample = np.random.choice(population, size=n, replace=False)
        sample_means.append(sample.mean())
    
    # Plot distribution of sample means
    plt.hist(sample_means, alpha=0.5, label=f'n={n}')
    
plt.xlabel('Sample Mean')
plt.ylabel('Frequency')
plt.legend()
plt.title('Distribution of Sample Means (Central Limit Theorem)')
plt.axvline(true_mean, color='red', linestyle='--', label='True Mean')
plt.show()

# What do you observe about sample size and accuracy?
```

**Observations:**
- Effect of sample size: _______________
- Shape of sampling distribution: _______________
- Variability: _______________

---

## Part 2: Hypothesis Testing Fundamentals

### Exercise 2.1: Null and Alternative Hypotheses
Write null (H₀) and alternative (H₁) hypotheses:

**Scenario 1:** Test if new drug reduces blood pressure
- H₀: _______________
- H₁: _______________
- Type of test: One-tailed / Two-tailed

**Scenario 2:** Test if coin is fair
- H₀: _______________
- H₁: _______________
- Type of test: One-tailed / Two-tailed

**Scenario 3:** Test if mean test score ≠ 75
- H₀: _______________
- H₁: _______________
- Type of test: One-tailed / Two-tailed

### Exercise 2.2: Type I and Type II Errors
For each scenario, describe the errors:

**Scenario:** Testing if new teaching method improves scores

| Error Type | What It Means | Consequence |
|------------|---------------|-------------|
| **Type I (False Positive)** | | |
| **Type II (False Negative)** | | |

**Which error is worse in this scenario?** _______________

### Exercise 2.3: Significance Level (α)
Explain the meaning of significance levels:

- α = 0.05 means: _______________
- α = 0.01 means: _______________
- When would you use α = 0.01? _______________
- When might α = 0.10 be acceptable? _______________

---

## Part 3: Conducting T-Tests

### Exercise 3.1: One-Sample T-Test
Test if sample mean differs from population mean:

```python
from scipy import stats
import numpy as np

# Example: Test if average exam score differs from 75
scores = [72, 78, 81, 75, 79, 83, 70, 76, 82, 77]

# Conduct one-sample t-test
hypothesized_mean = 75
t_statistic, p_value = stats.ttest_1samp(scores, hypothesized_mean)

print(f"T-statistic: {t_statistic:.4f}")
print(f"P-value: {p_value:.4f}")

# Interpret results
alpha = 0.05
if p_value < alpha:
    print("Reject null hypothesis")
else:
    print("Fail to reject null hypothesis")

# Your interpretation:
```

**Your Answer:**
- T-statistic: _______________
- P-value: _______________
- Conclusion: _______________
- In plain English: _______________

### Exercise 3.2: Two-Sample T-Test
Compare two groups:

```python
# Compare test scores: traditional vs new teaching method
traditional = [72, 75, 71, 78, 74, 76, 73, 77, 75, 74]
new_method = [78, 82, 80, 85, 79, 83, 81, 84, 80, 82]

# Conduct independent t-test
t_stat, p_val = stats.ttest_ind(traditional, new_method)

print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_val:.4f}")

# Calculate effect size (Cohen's d)
def cohens_d(group1, group2):
    # Your implementation
    pass

effect_size = cohens_d(traditional, new_method)
print(f"Effect size: {effect_size:.4f}")
```

**Analysis:**
- Is the difference statistically significant? _______________
- Is the difference practically significant? _______________
- What would you recommend? _______________

### Exercise 3.3: Paired T-Test
Test before and after measurements:

```python
# Weight before and after diet program (same people)
before = [180, 195, 210, 175, 188, 192, 205, 178, 183, 198]
after = [175, 190, 205, 170, 182, 188, 200, 175, 178, 195]

# Paired t-test
t_stat, p_val = stats.ttest_rel(before, after)

# Calculate mean difference
mean_diff = np.mean(np.array(before) - np.array(after))

print(f"Mean weight loss: {mean_diff:.2f} lbs")
print(f"P-value: {p_val:.4f}")

# Your conclusion:
```

---

## Part 4: Confidence Intervals

### Exercise 4.1: Calculate Confidence Intervals
```python
from scipy import stats
import numpy as np

data = [45, 48, 52, 50, 55, 47, 53, 49, 51, 54]

# Calculate 95% confidence interval
mean = np.mean(data)
sem = stats.sem(data)  # Standard error of mean
ci = stats.t.interval(0.95, len(data)-1, loc=mean, scale=sem)

print(f"Mean: {mean:.2f}")
print(f"95% CI: ({ci[0]:.2f}, {ci[1]:.2f})")

# Calculate 90% and 99% CIs
ci_90 = stats.t.interval(0.90, len(data)-1, loc=mean, scale=sem)
ci_99 = stats.t.interval(0.99, len(data)-1, loc=mean, scale=sem)

# Compare the widths
```

**Interpretation:**
- What does the 95% CI mean? _______________
- Why is 99% CI wider than 95%? _______________
- What affects CI width? _______________

### Exercise 4.2: CI Visualization
```python
import matplotlib.pyplot as plt

# Simulate multiple samples and their CIs
population = np.random.normal(50, 10, 10000)
true_mean = 50

sample_means = []
cis = []

for _ in range(100):
    sample = np.random.choice(population, 30)
    mean = np.mean(sample)
    sem = stats.sem(sample)
    ci = stats.t.interval(0.95, len(sample)-1, loc=mean, scale=sem)
    
    sample_means.append(mean)
    cis.append(ci)

# Plot CIs
fig, ax = plt.subplots(figsize=(10, 6))
for i, (mean, ci) in enumerate(zip(sample_means, cis)):
    color = 'red' if ci[0] > true_mean or ci[1] < true_mean else 'blue'
    ax.plot([i, i], ci, color=color, alpha=0.5)
    ax.plot(i, mean, 'o', color=color, markersize=2)

ax.axhline(true_mean, color='black', linestyle='--', label='True Mean')
ax.set_xlabel('Sample Number')
ax.set_ylabel('Value')
ax.set_title('95% Confidence Intervals (Red = Miss True Mean)')
ax.legend()
plt.show()

# Count how many CIs capture true mean
```

---

## Part 5: P-Values Deep Dive

### Exercise 5.1: Interpret P-Values
For each p-value, interpret the result:

| P-Value | α = 0.05 | Interpretation | Strength of Evidence |
|---------|----------|----------------|----------------------|
| 0.001 | | | |
| 0.04 | | | |
| 0.06 | | | |
| 0.50 | | | |

### Exercise 5.2: P-Value Misinterpretations
Mark each statement as TRUE or FALSE:

- [ ] p < 0.05 means there's a 95% chance the result is real
- [ ] p > 0.05 means there's no effect
- [ ] p-value measures the size of the effect
- [ ] Smaller p-value = more important finding
- [ ] p-value is probability null hypothesis is true
- [ ] p < 0.05 means result is practically significant

**Correct interpretations:** _______________

### Exercise 5.3: Power Analysis
```python
from statsmodels.stats.power import ttest_power

# Calculate required sample size
effect_size = 0.5  # Cohen's d
alpha = 0.05
power = 0.80

# Your code to calculate sample size
from statsmodels.stats.power import tt_solve_power

n = tt_solve_power(effect_size=effect_size, alpha=alpha, power=power)
print(f"Required sample size per group: {np.ceil(n)}")

# How does sample size change with:
# - Different effect sizes
# - Different power requirements
# - Different alpha levels
```

---

## Part 6: ANOVA

### Exercise 6.1: One-Way ANOVA
Compare more than two groups:

```python
from scipy.stats import f_oneway

# Compare three diet programs
diet_a = [5, 7, 6, 8, 7, 6, 9, 7, 8, 6]
diet_b = [8, 9, 10, 9, 11, 10, 9, 10, 11, 9]
diet_c = [6, 7, 7, 8, 6, 7, 8, 7, 6, 7]

# Conduct ANOVA
f_stat, p_val = f_oneway(diet_a, diet_b, diet_c)

print(f"F-statistic: {f_stat:.4f}")
print(f"P-value: {p_val:.4f}")

# Post-hoc tests if significant
if p_val < 0.05:
    # Conduct pairwise t-tests with correction
    from scipy.stats import ttest_ind
    # Your code here
    pass
```

**Interpretation:**
- Are there significant differences? _______________
- Which groups differ? _______________
- What would you recommend? _______________

---

## Part 7: Real-World Hypothesis Testing

### Exercise 7.1: A/B Test Analysis
```python
# Website conversion rates
control = [1, 0, 0, 1, 0, 1, 0, 0, 1, 1] * 10  # 40% conversion
treatment = [1, 1, 0, 1, 0, 1, 1, 0, 1, 1] * 10  # 60% conversion

# Conduct chi-square test for proportions
from scipy.stats import chi2_contingency

# Create contingency table
converted_control = sum(control)
not_converted_control = len(control) - converted_control
converted_treatment = sum(treatment)
not_converted_treatment = len(treatment) - converted_treatment

table = [[converted_control, not_converted_control],
         [converted_treatment, not_converted_treatment]]

chi2, p_val, dof, expected = chi2_contingency(table)

print(f"Chi-square: {chi2:.4f}")
print(f"P-value: {p_val:.4f}")

# Calculate confidence interval for difference in proportions
# Your code here
```

**Business Decision:** _______________

---

## Deliverables

Submit:
1. ✅ Population/sample identification (Exercise 1.1)
2. ✅ Sampling simulation observations (Exercise 1.3)
3. ✅ Hypotheses for all scenarios (Exercise 2.1)
4. ✅ Type I/II error explanations (Exercise 2.2)
5. ✅ One-sample t-test results (Exercise 3.1)
6. ✅ Two-sample t-test analysis (Exercise 3.2)
7. ✅ Paired t-test conclusions (Exercise 3.3)
8. ✅ Confidence interval interpretations (Exercise 4.1)
9. ✅ P-value interpretation table (Exercise 5.1)
10. ✅ True/false p-value statements (Exercise 5.2)
11. ✅ ANOVA results (Exercise 6.1)
12. ✅ A/B test business decision (Exercise 7.1)

---

**Next Lab:** Lab 10 - Correlation vs Causation
