# Lab 06: Problem Framing Workshop

## Chapter 2: The Data Scientist's Mindset

### Learning Objectives
- Master problem framing techniques
- Translate business goals to data science problems
- Identify appropriate problem types

### Duration
60 minutes

---

## Part 1: Problem Type Recognition

### Exercise 1.1: Classification vs Regression vs Clustering
For each business question, identify the ML problem type:

| Business Question | Problem Type | Target Variable | Evaluation Metric |
|-------------------|--------------|-----------------|-------------------|
| Will this customer churn? | | | |
| How much will we sell next month? | | | |
| Which customers are similar? | | | |
| Is this transaction fraudulent? | | | |
| What's the lifetime value of this customer? | | | |
| How to segment our market? | | | |
| Will this email be spam? | | | |
| How many units will we need? | | | |

### Exercise 1.2: Supervised vs Unsupervised
Categorize these scenarios:

**Scenario A:** Find patterns in customer behavior without predefined categories
- **Type:** Supervised / Unsupervised
- **Why:** _______________
- **Algorithm examples:** _______________

**Scenario B:** Predict house prices based on historical sales
- **Type:** Supervised / Unsupervised
- **Why:** _______________
- **Algorithm examples:** _______________

**Scenario C:** Detect anomalies in network traffic
- **Type:** Supervised / Unsupervised / Both
- **Why:** _______________
- **Algorithm examples:** _______________

---

## Part 2: The Problem Canvas

### Exercise 2.1: Complete Problem Canvas
For the scenario below, complete the problem canvas:

> **Scenario:** A streaming service wants to reduce subscription cancellations

**PROBLEM CANVAS:**

**1. Business Objective:**
_______________

**2. Success Metrics:**
- Primary: _______________
- Secondary: _______________

**3. Constraints:**
- Time: _______________
- Budget: _______________
- Resources: _______________

**4. Data Requirements:**
- Must Have: _______________
- Nice to Have: _______________

**5. Stakeholders:**
- Primary: _______________
- Secondary: _______________

**6. Risks:**
- Technical: _______________
- Business: _______________
- Ethical: _______________

**7. Definition of Done:**
_______________

### Exercise 2.2: From Vague to Specific
Transform vague requests into well-defined problems:

**Vague:** "We need to understand our customers better"
**Specific:**
- Problem: _______________
- Measurable Goal: _______________
- Data Needed: _______________
- Success Criteria: _______________
- Timeline: _______________

**Vague:** "Improve our operations"
**Specific:**
- Problem: _______________
- Measurable Goal: _______________
- Data Needed: _______________
- Success Criteria: _______________
- Timeline: _______________

---

## Part 3: Feature and Target Definition

### Exercise 3.1: Target Variable Selection
For each problem, define the target variable precisely:

**Problem:** Predict customer lifetime value
- **Target Variable Name:** _______________
- **Data Type:** _______________
- **Calculation:** _______________
- **Time Window:** _______________

**Problem:** Identify high-risk loans
- **Target Variable Name:** _______________
- **Data Type:** _______________
- **Definition:** _______________
- **Threshold:** _______________

### Exercise 3.2: Feature Brainstorming
For predicting employee attrition, brainstorm features:

**Demographic Features:**
1. _______________
2. _______________
3. _______________

**Behavioral Features:**
1. _______________
2. _______________
3. _______________

**Performance Features:**
1. _______________
2. _______________
3. _______________

**Engagement Features:**
1. _______________
2. _______________
3. _______________

**Derived Features:**
1. _______________
2. _______________
3. _______________

---

## Part 4: Feasibility Analysis

### Exercise 4.1: Can This Be Solved?
Assess feasibility for each problem:

**Problem A:** Predict lottery numbers
- **Feasibility Score (1-10):** ___
- **Why/Why Not:** _______________
- **Data Availability:** _______________
- **Signal Strength:** _______________

**Problem B:** Predict customer churn
- **Feasibility Score (1-10):** ___
- **Why/Why Not:** _______________
- **Data Availability:** _______________
- **Signal Strength:** _______________

**Problem C:** Predict stock prices 1 year out
- **Feasibility Score (1-10):** ___
- **Why/Why Not:** _______________
- **Data Availability:** _______________
- **Signal Strength:** _______________

### Exercise 4.2: Data Requirements Assessment
For building a recommendation system:

**Required Data:**
| Data Type | Essential? | Currently Have? | How to Get? |
|-----------|------------|-----------------|-------------|
| User clicks | | | |
| Purchase history | | | |
| User ratings | | | |
| Product metadata | | | |
| User demographics | | | |

---

## Part 5: Baseline Establishment

### Exercise 5.1: Define Baselines
For each problem, establish baseline performance:

**Problem:** Email spam detection
- **Naive Baseline:** _______________
- **Expected Performance:** _______________
- **Target Performance:** _______________
- **World-Class Performance:** _______________

**Problem:** Sales forecasting
- **Naive Baseline:** _______________
- **Expected Performance:** _______________
- **Target Performance:** _______________
- **World-Class Performance:** _______________

### Exercise 5.2: Code a Simple Baseline
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
df = pd.read_csv('classification_data.csv')
X = df.drop('target', axis=1)
y = df['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Baseline 1: Always predict majority class
majority_class = y_train.mode()[0]
y_pred_baseline = [majority_class] * len(y_test)
baseline_acc = accuracy_score(y_test, y_pred_baseline)

print(f"Baseline Accuracy: {baseline_acc:.3f}")

# Your turn: Create a slightly smarter baseline
# Hint: Use simple rules or random forest with default params
```

---

## Part 6: Real-World Problem Framing

### Exercise 6.1: Healthcare Case
Frame this problem:

> **Situation:** A hospital has high readmission rates for heart failure patients

**Your Framing:**
1. **Problem Statement:** _______________
2. **Predictive Question:** _______________
3. **Target Variable:** _______________
4. **Time Window:** _______________
5. **Features:** _______________
6. **Success Metric:** _______________
7. **Baseline:** _______________
8. **Ethical Considerations:** _______________

### Exercise 6.2: Retail Case
Frame this problem:

> **Situation:** An online retailer wants to optimize inventory

**Your Framing:**
1. **Problem Statement:** _______________
2. **Predictive Question:** _______________
3. **Target Variable:** _______________
4. **Time Window:** _______________
5. **Features:** _______________
6. **Success Metric:** _______________
7. **Baseline:** _______________
8. **Business Constraints:** _______________

### Exercise 6.3: Finance Case
Frame this problem:

> **Situation:** A bank wants to detect fraudulent transactions

**Your Framing:**
1. **Problem Statement:** _______________
2. **Predictive Question:** _______________
3. **Target Variable:** _______________
4. **Time Window:** _______________
5. **Features:** _______________
6. **Success Metric:** _______________
7. **Class Imbalance Strategy:** _______________
8. **False Positive vs False Negative Trade-off:** _______________

---

## Part 7: Your Own Problem

### Exercise 7.1: Frame Your Project
Choose a problem you're interested in and fully frame it:

**Domain:** _______________

**Business Context:** _______________

**Problem Statement:** _______________

**Data Science Translation:** _______________

**Problem Type:** _______________

**Target Variable:** _______________

**Potential Features (list 10):**
1. _______________
2. _______________
3. _______________
4. _______________
5. _______________
6. _______________
7. _______________
8. _______________
9. _______________
10. _______________

**Success Metrics:**
- Primary: _______________
- Business: _______________

**Baseline Approach:** _______________

**Data Availability:** _______________

**Feasibility Assessment:** _______________

**Timeline:** _______________

**Risks:** _______________

---

## Deliverables

Submit:
1. ✅ Problem type table (Exercise 1.1)
2. ✅ Supervised vs unsupervised categorization (Exercise 1.2)
3. ✅ Completed problem canvas (Exercise 2.1)
4. ✅ Specific problem definitions (Exercise 2.2)
5. ✅ Target variable definitions (Exercise 3.1)
6. ✅ Feature brainstorming (Exercise 3.2)
7. ✅ Feasibility assessments (Exercise 4.1)
8. ✅ Baseline code and results (Exercise 5.2)
9. ✅ Three case study framings (Exercises 6.1-6.3)
10. ✅ Your own project framing (Exercise 7.1)

---

## Reflection Questions

1. What makes a well-framed problem?
2. Why is baseline establishment important?
3. How does problem framing affect project success?

---

**Next Lab:** Lab 07 - Data Ethics and Bias
