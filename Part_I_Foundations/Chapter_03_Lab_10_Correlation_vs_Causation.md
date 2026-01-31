# Lab 10: Correlation vs Causation

## Chapter 3: The Language of Data — Math and Stats for Beginners

### Learning Objectives
- Calculate and interpret correlation coefficients
- Understand the difference between correlation and causation
- Identify confounding variables

### Duration
75 minutes

---

## Part 1: Understanding Correlation

### Exercise 1.1: Calculate Correlations
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create datasets with different correlations
np.random.seed(42)

# Strong positive correlation
x1 = np.random.rand(100)
y1 = 2 * x1 + np.random.rand(100) * 0.1

# Strong negative correlation
x2 = np.random.rand(100)
y2 = -2 * x2 + np.random.rand(100) * 0.1

# No correlation
x3 = np.random.rand(100)
y3 = np.random.rand(100)

# Non-linear relationship
x4 = np.linspace(-3, 3, 100)
y4 = x4**2 + np.random.rand(100) * 0.5

# Calculate Pearson correlation for each
from scipy.stats import pearsonr

corr1, p1 = pearsonr(x1, y1)
corr2, p2 = pearsonr(x2, y2)
corr3, p3 = pearsonr(x3, y3)
corr4, p4 = pearsonr(x4, y4)

# Plot all four
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes[0,0].scatter(x1, y1)
axes[0,0].set_title(f'Strong Positive (r={corr1:.2f})')
# Complete the plots...

# Your code here
```

**Fill in:**
- Strong positive r = _______________
- Strong negative r = _______________
- No correlation r = _______________
- Non-linear r = _______________

**What does this tell you about Pearson correlation?** _______________

### Exercise 1.2: Interpret Correlation Strength
Classify these correlations:

| Correlation | Strength | Direction | Interpretation |
|-------------|----------|-----------|----------------|
| r = 0.95 | | | |
| r = -0.75 | | | |
| r = 0.15 | | | |
| r = -0.40 | | | |
| r = 0.02 | | | |

### Exercise 1.3: Correlation Matrix
```python
# Real dataset: examine relationships
df = pd.DataFrame({
    'age': [25, 35, 45, 55, 65, 30, 40, 50, 60, 28],
    'income': [40000, 60000, 80000, 70000, 50000, 55000, 75000, 85000, 65000, 45000],
    'experience': [2, 10, 20, 30, 40, 5, 15, 25, 35, 3],
    'satisfaction': [7, 8, 6, 9, 7, 8, 7, 9, 8, 7]
})

# Create correlation matrix
corr_matrix = df.corr()
print(corr_matrix)

# Visualize with heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.show()

# Identify:
# - Strongest positive correlation: _______________
# - Strongest negative correlation: _______________
# - Weakest correlation: _______________
```

---

## Part 2: Causation vs Correlation

### Exercise 2.1: Identify the Fallacy
For each scenario, explain why correlation ≠ causation:

**Scenario 1:** Ice cream sales and drowning deaths are correlated (r=0.9)
- **Actual relationship:** _______________
- **Confounding variable:** _______________
- **True cause:** _______________

**Scenario 2:** Number of firefighters at scene correlates with damage amount
- **Fallacy:** _______________
- **Confounding variable:** _______________
- **True cause:** _______________

**Scenario 3:** Shoe size and reading ability in children
- **Correlation:** _______________
- **Confounding variable:** _______________
- **True cause:** _______________

### Exercise 2.2: Spurious Correlations
Research real examples of spurious correlations:

**Find three:**
1. _______________ correlates with _______________
   - Explanation: _______________

2. _______________ correlates with _______________
   - Explanation: _______________

3. _______________ correlates with _______________
   - Explanation: _______________

**Source:** tylervigen.com/spurious-correlations

### Exercise 2.3: Design Causal Test
For this correlation, design a study to test causation:

**Observation:** Exercise frequency and happiness are correlated

**Your Experimental Design:**
1. **Hypothesis:** _______________
2. **Independent Variable:** _______________
3. **Dependent Variable:** _______________
4. **Control Group:** _______________
5. **Treatment Group:** _______________
6. **Confounding Variables to Control:**
   - _______________
   - _______________
   - _______________
7. **Measurement Method:** _______________
8. **Sample Size:** _______________
9. **Duration:** _______________

---

## Part 3: Confounding Variables

### Exercise 3.1: Identify Confounders
For each relationship, identify potential confounders:

**Relationship A:** Coffee consumption → Heart disease
**Potential Confounders:**
1. _______________
2. _______________
3. _______________
4. _______________

**Relationship B:** Education level → Income
**Potential Confounders:**
1. _______________
2. _______________
3. _______________
4. _______________

**Relationship C:** Social media use → Depression
**Potential Confounders:**
1. _______________
2. _______________
3. _______________
4. _______________

### Exercise 3.2: Simulate Simpson's Paradox
```python
import pandas as pd
import matplotlib.pyplot as plt

# Simpson's Paradox example: Hospital treatment success rates

# Hospital A (treats severe cases)
hospital_a = pd.DataFrame({
    'treatment': ['A', 'B'] * 50,
    'success': [45, 40],  # Out of 50 each
    'severity': ['high'] * 100
})

# Hospital B (treats mild cases)
hospital_b = pd.DataFrame({
    'treatment': ['A', 'B'] * 50,
    'success': [85, 90],  # Out of 50 each
    'severity': ['low'] * 100
})

# Aggregate without considering severity
# Aggregate with severity

# Your code to demonstrate Simpson's Paradox
```

**What happens?** _______________
**Why is this important?** _______________

---

## Part 4: Establishing Causation

### Exercise 4.1: Bradford Hill Criteria
Apply Bradford Hill criteria to assess causation:

**Claim:** Smoking causes lung cancer

| Criterion | Evidence | Score (1-5) |
|-----------|----------|-------------|
| **Strength of Association** | | |
| **Consistency** | | |
| **Specificity** | | |
| **Temporality** | | |
| **Biological Gradient** | | |
| **Plausibility** | | |
| **Coherence** | | |
| **Experiment** | | |
| **Analogy** | | |

**Overall Assessment:** _______________

### Exercise 4.2: Reverse Causation
Identify which direction causality flows:

**Scenario 1:** Depression and social isolation
- **Possible Direction A:** Depression → Isolation
- **Possible Direction B:** Isolation → Depression
- **Most Likely:** _______________
- **Evidence Needed:** _______________

**Scenario 2:** Wealth and health
- **Possible Direction A:** Wealth → Health
- **Possible Direction B:** Health → Wealth
- **Most Likely:** _______________
- **Evidence Needed:** _______________

---

## Part 5: Experimental Design

### Exercise 5.1: Randomized Controlled Trial (RCT)
Design an RCT:

**Research Question:** Does Method X improve learning outcomes?

**Your RCT Design:**

**1. Participants:**
- Population: _______________
- Sample Size: _______________
- Sampling Method: _______________

**2. Randomization:**
- Method: _______________
- Balance Check: _______________

**3. Groups:**
- Control: _______________
- Treatment: _______________

**4. Blinding:**
- Single-blind / Double-blind / None
- How: _______________

**5. Duration:**
- Length: _______________
- Follow-ups: _______________

**6. Measurements:**
- Baseline: _______________
- Outcome: _______________
- Method: _______________

**7. Analysis Plan:**
- Primary outcome: _______________
- Statistical test: _______________
- Success criteria: _______________

### Exercise 5.2: Natural Experiments
Identify natural experiments:

**Scenario:** Study effect of minimum wage on employment

**Natural Experiment Approach:**
- **Treatment Group:** _______________
- **Control Group:** _______________
- **What makes it "natural":** _______________
- **Confounders to watch:** _______________
- **Limitations:** _______________

---

## Part 6: Time Series and Causation

### Exercise 6.1: Granger Causality
```python
from statsmodels.tsa.stattools import grangercausalitytests
import numpy as np

# Create time series data
np.random.seed(42)
time = np.arange(100)

# X causes Y (with lag)
x = np.random.randn(100).cumsum()
y = np.zeros(100)
y[0] = np.random.randn()
for t in range(1, 100):
    y[t] = 0.7 * y[t-1] + 0.3 * x[t-1] + np.random.randn()

# Test Granger causality
data = np.column_stack([y, x])

# Your code to test if x Granger-causes y
# grangercausalitytests(data, maxlag=5)
```

**Interpretation:** _______________
**Limitations:** _______________

---

## Part 7: Real-World Cases

### Exercise 7.1: Evaluate Claims
For each claim, assess the evidence:

**Claim 1:** "Our new feature increased user engagement by 30%"
- **What evidence do you need?**
  - [ ] Before/after comparison
  - [ ] A/B test results
  - [ ] Control for seasonality
  - [ ] User feedback
  - [ ] Other: _______________

**Claim 2:** "Meditation reduces stress"
- **Current evidence:** _______________
- **Correlation or causation?** _______________
- **What's missing?** _______________

**Claim 3:** "Vaccine caused adverse event"
- **How to investigate:**
  1. _______________
  2. _______________
  3. _______________

### Exercise 7.2: Media Literacy
Find a news article claiming causation:

**Article Title:** _______________
**Claim:** _______________
**Evidence Presented:** _______________
**Study Design:** _______________

**Your Evaluation:**
- Is causation justified? _______________
- What's the evidence quality? _______________
- What's missing? _______________
- Better headline: _______________

---

## Part 8: Practical Application

### Exercise 8.1: Your Own Analysis
Find a correlation in real data and investigate:

```python
# Load a dataset of your choice
# Find correlations
# Investigate potential causation

# Your analysis here
```

**Report:**
1. **Correlation Found:** _______________
2. **Correlation Coefficient:** _______________
3. **Potential Causal Relationship:** _______________
4. **Alternative Explanations:**
   - _______________
   - _______________
   - _______________
5. **Confounding Variables:** _______________
6. **Experiment to Test Causation:** _______________
7. **Conclusion:** _______________

---

## Deliverables

Submit:
1. ✅ Correlation calculations and plots (Exercise 1.1)
2. ✅ Correlation matrix analysis (Exercise 1.3)
3. ✅ Spurious correlation explanations (Exercise 2.1)
4. ✅ Three real spurious correlations (Exercise 2.2)
5. ✅ Experimental design (Exercise 2.3)
6. ✅ Confounding variables identified (Exercise 3.1)
7. ✅ Bradford Hill assessment (Exercise 4.1)
8. ✅ RCT design (Exercise 5.1)
9. ✅ Media article evaluation (Exercise 7.2)
10. ✅ Your own correlation analysis (Exercise 8.1)

---

## Reflection Questions

1. Why is establishing causation so difficult?
2. When is correlation useful even without causation?
3. How can we make better causal claims?

---

**Next Lab:** Lab 11 - Probability and Distributions
