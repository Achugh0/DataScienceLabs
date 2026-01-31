# Lab 12: A/B Testing Case Study

## Chapter 3: The Language of Data — Math and Stats for Beginners

### Learning Objectives
- Design and execute A/B tests
- Analyze A/B test results
- Make data-driven recommendations

### Duration
90 minutes

---

## Part 1: A/B Test Design

### Exercise 1.1: Test Planning
You're testing a new website design:

**1. Research Question:**
_______________

**2. Hypothesis:**
- H₀: _______________
- H₁: _______________

**3. Metrics:**
- Primary: _______________
- Secondary: _______________
- Guardrail: _______________

**4. Minimum Detectable Effect:**
- Current conversion rate: 15%
- Minimum improvement worth detecting: _______________
- Why this value? _______________

**5. Sample Size:**
```python
from statsmodels.stats.power import zt_ind_solve_power
from statsmodels.stats.proportion import proportion_effectsize

baseline = 0.15
target = _______________  # Fill in based on your MDE

effect_size = proportion_effectsize(baseline, target)
n_per_group = zt_ind_solve_power(effect_size=effect_size, 
                                  alpha=0.05, 
                                  power=0.80,
                                  alternative='larger')

print(f"Required sample size per group: {np.ceil(n_per_group)}")
```

**6. Test Duration:**
- Expected daily traffic: _______________
- Days needed: _______________

**7. Randomization:**
- Unit: User / Session / Other
- Method: _______________

**8. Success Criteria:**
_______________

---

## Part 2: Generate Synthetic Data

### Exercise 2.1: Simulate A/B Test
```python
import numpy as np
import pandas as pd

np.random.seed(42)

# Parameters
n_per_group = 5000
control_rate = 0.15
treatment_rate = 0.18  # 20% relative lift

# Generate data
control = np.random.binomial(1, control_rate, n_per_group)
treatment = np.random.binomial(1, treatment_rate, n_per_group)

# Create DataFrame
df = pd.DataFrame({
    'group': ['control']*n_per_group + ['treatment']*n_per_group,
    'converted': np.concatenate([control, treatment]),
    'revenue': np.concatenate([
        control * np.random.normal(50, 10, n_per_group),
        treatment * np.random.normal(50, 10, n_per_group)
    ])
})

print(df.groupby('group').agg({
    'converted': ['sum', 'mean', 'count'],
    'revenue': ['sum', 'mean']
}))
```

---

## Part 3: Analyze Results

### Exercise 3.1: Conversion Rate Analysis
```python
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt

# Calculate conversion rates
conversion_summary = df.groupby('group')['converted'].agg(['sum', 'count'])
conversion_summary['rate'] = conversion_summary['sum'] / conversion_summary['count']
print(conversion_summary)

# Chi-square test
contingency_table = pd.crosstab(df['group'], df['converted'])
chi2, p_value, dof, expected = chi2_contingency(contingency_table)

print(f"\nChi-square statistic: {chi2:.4f}")
print(f"P-value: {p_value:.4f}")

# Interpret
if p_value < 0.05:
    print("✅ Statistically significant")
else:
    print("❌ Not statistically significant")

# Visualize
conversion_summary['rate'].plot(kind='bar')
plt.ylabel('Conversion Rate')
plt.title('A/B Test Results')
plt.xticks(rotation=0)
plt.show()
```

**Your interpretation:**
- Control rate: _______________
- Treatment rate: _______________
- Relative lift: _______________
- Absolute lift: _______________
- Statistically significant? _______________

### Exercise 3.2: Confidence Intervals
```python
from statsmodels.stats.proportion import proportion_confint

# Calculate 95% CI for each group
for group in ['control', 'treatment']:
    group_data = df[df['group'] == group]
    successes = group_data['converted'].sum()
    trials = len(group_data)
    
    ci_low, ci_high = proportion_confint(successes, trials, alpha=0.05, method='wilson')
    
    print(f"{group}:")
    print(f"  Rate: {successes/trials:.4f}")
    print(f"  95% CI: ({ci_low:.4f}, {ci_high:.4f})")

# Calculate CI for difference
# Your code here using bootstrap or delta method
```

### Exercise 3.3: Power Analysis (Post-Hoc)
```python
from statsmodels.stats.power import zt_ind_solve_power
from statsmodels.stats.proportion import proportion_effectsize

# Calculate achieved power
control_rate = df[df['group']=='control']['converted'].mean()
treatment_rate = df[df['group']=='treatment']['converted'].mean()
n = len(df[df['group']=='control'])

effect_size = proportion_effectsize(control_rate, treatment_rate)
achieved_power = zt_ind_solve_power(effect_size=effect_size,
                                    nobs1=n,
                                    alpha=0.05,
                                    alternative='larger')

print(f"Achieved power: {achieved_power:.4f}")
```

---

## Part 4: Segmentation Analysis

### Exercise 4.1: Subgroup Analysis
```python
# Add user segments
np.random.seed(42)
df['device'] = np.random.choice(['mobile', 'desktop'], len(df), p=[0.6, 0.4])
df['new_user'] = np.random.choice([True, False], len(df), p=[0.3, 0.7])

# Analyze by segment
for segment_col in ['device', 'new_user']:
    print(f"\n=== Analysis by {segment_col} ===")
    segment_results = df.groupby(['group', segment_col])['converted'].agg(['sum', 'count', 'mean'])
    print(segment_results)
    
    # Visualize
    pivot = df.pivot_table(values='converted', index='group', columns=segment_col, aggfunc='mean')
    pivot.plot(kind='bar')
    plt.title(f'Conversion Rate by {segment_col}')
    plt.ylabel('Conversion Rate')
    plt.xticks(rotation=0)
    plt.legend(title=segment_col)
    plt.show()
```

**Insights:**
- Which segment showed biggest lift? _______________
- Any segments where treatment performed worse? _______________
- Should results be segmented? Why/why not? _______________

---

## Part 5: Common Pitfalls

### Exercise 5.1: Peeking Problem
```python
# Simulate continuous monitoring (wrong!)
results_over_time = []

for day in range(1, 31):
    # Accumulate data
    n_so_far = day * 100
    control_so_far = df[df['group']=='control'].iloc[:n_so_far]
    treatment_so_far = df[df['group']=='treatment'].iloc[:n_so_far]
    
    # Test each day
    table = pd.crosstab(
        pd.concat([control_so_far, treatment_so_far])['group'],
        pd.concat([control_so_far, treatment_so_far])['converted']
    )
    _, p_val, _, _ = chi2_contingency(table)
    
    results_over_time.append({
        'day': day,
        'p_value': p_val,
        'significant': p_val < 0.05
    })

results_df = pd.DataFrame(results_over_time)

# Plot
plt.figure(figsize=(12, 6))
plt.plot(results_df['day'], results_df['p_value'], marker='o')
plt.axhline(0.05, color='red', linestyle='--', label='α=0.05')
plt.xlabel('Day')
plt.ylabel('P-value')
plt.title('P-value Over Time (Demonstrates Peeking Problem)')
plt.legend()
plt.show()

# How many times did we cross threshold?
crossings = (results_df['p_value'] < 0.05).sum()
print(f"Number of times p < 0.05: {crossings}")
```

**Why is this a problem?** _______________
**Correct approach:** _______________

### Exercise 5.2: Sample Ratio Mismatch
```python
# Check if randomization worked
observed_ratio = df.groupby('group').size()
print("Observed counts per group:")
print(observed_ratio)

# Statistical test
expected_ratio = len(df) / 2
chi2_srm, p_srm = stats.chisquare(observed_ratio, f_exp=[expected_ratio, expected_ratio])

print(f"\nSample Ratio Mismatch Test:")
print(f"Chi-square: {chi2_srm:.4f}")
print(f"P-value: {p_srm:.4f}")

if p_srm < 0.01:  # More stringent threshold
    print("⚠️  Warning: Sample ratio mismatch detected!")
else:
    print("✅ Randomization looks good")
```

### Exercise 5.3: Multiple Testing
```python
# Testing multiple metrics
metrics = ['converted', 'revenue']
p_values = []

for metric in metrics:
    control_vals = df[df['group']=='control'][metric]
    treatment_vals = df[df['group']=='treatment'][metric]
    
    if metric == 'converted':
        # Use chi-square for binary
        table = pd.crosstab(df['group'], df[metric])
        _, p, _, _ = chi2_contingency(table)
    else:
        # Use t-test for continuous
        _, p = stats.ttest_ind(control_vals, treatment_vals)
    
    p_values.append(p)
    print(f"{metric}: p={p:.4f}")

# Bonferroni correction
adjusted_alpha = 0.05 / len(metrics)
print(f"\nAdjusted α (Bonferroni): {adjusted_alpha:.4f}")

# Which results survive correction?
for metric, p in zip(metrics, p_values):
    if p < adjusted_alpha:
        print(f"✅ {metric}: Significant after correction")
    else:
        print(f"❌ {metric}: Not significant after correction")
```

---

## Part 6: Business Decision

### Exercise 6.1: Go/No-Go Decision Framework
Complete this decision framework:

**Statistical Significance:**
- P-value: _______________
- Significant? Yes / No

**Practical Significance:**
- Relative lift: _______________
- Absolute lift: _______________
- Meets minimum threshold? Yes / No

**Business Impact:**
- Expected annual revenue impact: $_______________
- Implementation cost: $_______________
- ROI: _______________

**Risk Assessment:**
- Confidence in result: High / Medium / Low
- Risk of negative impact: High / Medium / Low
- Risk of no impact: High / Medium / Low

**Secondary Metrics:**
- Revenue: Impact _______________
- User engagement: Impact _______________
- Other KPIs: Impact _______________

**Recommendation:**
- Deploy to 100% / Deploy to X% / Run longer / Don't deploy

**Justification:**
_______________

---

## Part 7: Real-World Case Study

### Exercise 7.1: Booking.com Case
Research and analyze:

**Background:** Booking.com ran an A/B test on showing scarcity messages

**Your Analysis:**
1. **What was tested?** _______________
2. **What metric improved?** _______________
3. **What was the trade-off?** _______________
4. **Ethical considerations:** _______________
5. **Long-term vs short-term effects:** _______________

### Exercise 7.2: Your Own Test
Design an A/B test for a real product:

**Product:** _______________

**Test Idea:** _______________

**Complete Test Plan:**
- Hypothesis: _______________
- Primary metric: _______________
- Sample size: _______________
- Duration: _______________
- Success criteria: _______________
- Risks: _______________

---

## Deliverables

Submit:
1. ✅ Complete test plan (Exercise 1.1)
2. ✅ Simulated data and initial analysis (Exercise 2.1)
3. ✅ Statistical analysis with interpretation (Exercises 3.1-3.3)
4. ✅ Segmentation insights (Exercise 4.1)
5. ✅ Common pitfalls analysis (Exercises 5.1-5.3)
6. ✅ Go/no-go decision framework (Exercise 6.1)
7. ✅ Your own A/B test design (Exercise 7.2)

---

## Reflection Questions

1. What's harder: running the test or interpreting results?
2. When should you not run an A/B test?
3. How do you balance statistical and practical significance?

---

**Next Lab:** Lab 13 - Career Stages and Skills
