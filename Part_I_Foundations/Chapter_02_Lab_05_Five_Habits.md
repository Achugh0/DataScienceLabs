# Lab 05: The Five Habits of Effective Data Scientists

## Chapter 2: The Data Scientist's Mindset

### Learning Objectives
- Develop curiosity-driven analysis skills
- Practice healthy skepticism
- Build systematic problem-solving habits

### Duration
75 minutes

---

## Part 1: Habit #1 - Relentless Curiosity

### Exercise 1.1: The "Five Whys" Technique
Practice deep questioning for this scenario:

> **Observation:** Website traffic dropped 30% last week

**Apply Five Whys:**
1. **Why did traffic drop?** _______________
2. **Why?** _______________
3. **Why?** _______________
4. **Why?** _______________
5. **Why?** _______________

**Root Cause:** _______________

### Exercise 1.2: Curiosity-Driven Exploration
Given this simple dataset, generate 10 curious questions:

```python
import pandas as pd

data = {
    'date': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04'],
    'sales': [100, 150, 90, 200],
    'visitors': [50, 75, 45, 80],
    'region': ['North', 'South', 'North', 'East']
}
df = pd.DataFrame(data)
```

**Your 10 Questions:**
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

**Now answer 3 of them with code:**
```python
# Your exploration code here
```

---

## Part 2: Habit #2 - Healthy Skepticism

### Exercise 2.1: Spotting Misleading Visualizations
Analyze these scenarios and identify issues:

**Scenario A:** A bar chart shows "Sales Growth" with y-axis starting at 95 instead of 0
**Issue:** _______________
**How it misleads:** _______________
**Better approach:** _______________

**Scenario B:** A news article claims "Ice cream sales cause drowning deaths" (correlation = 0.9)
**Issue:** _______________
**Alternative explanation:** _______________
**What's missing:** _______________

**Scenario C:** Survey says "90% of users love our app!" (n=10, all employees)
**Issues:**
1. _______________
2. _______________
3. _______________

### Exercise 2.2: Question Everything
For this claim, list 5 skeptical questions to ask:

> **Claim:** "Our new algorithm increased user engagement by 50%"

**Your Skeptical Questions:**
1. _______________
2. _______________
3. _______________
4. _______________
5. _______________

### Exercise 2.3: Data Quality Red Flags
Code to check for suspicious patterns:

```python
import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('suspicious_data.csv')

# Check for red flags:
# 1. Too many duplicate rows
duplicates = df.duplicated().sum()
print(f"Duplicates: {duplicates} ({duplicates/len(df)*100:.1f}%)")

# 2. Suspiciously round numbers
# Your code here

# 3. Missing data patterns
# Your code here

# 4. Outliers
# Your code here

# 5. Time gaps
# Your code here
```

---

## Part 3: Habit #3 - Structured Problem Solving

### Exercise 3.1: The CRISP-DM Framework
Apply CRISP-DM to this problem:

> **Problem:** A hospital wants to reduce patient readmission rates

**1. Business Understanding:**
- Objective: _______________
- Success Criteria: _______________
- Constraints: _______________

**2. Data Understanding:**
- Available Data: _______________
- Data Quality Issues: _______________
- Initial Insights: _______________

**3. Data Preparation:**
- Cleaning Steps: _______________
- Feature Engineering: _______________
- Data Transformations: _______________

**4. Modeling:**
- Algorithm Choice: _______________
- Justification: _______________
- Validation Strategy: _______________

**5. Evaluation:**
- Metrics: _______________
- Results: _______________
- Business Impact: _______________

**6. Deployment:**
- Implementation: _______________
- Monitoring: _______________
- Maintenance: _______________

### Exercise 3.2: Problem Decomposition
Break this complex problem into manageable sub-problems:

> **Problem:** Build a recommendation system for an e-commerce site

**Sub-Problems:**
1. _______________
   - Tasks: _______________
   - Dependencies: _______________
   - Estimated Time: _______________

2. _______________
   - Tasks: _______________
   - Dependencies: _______________
   - Estimated Time: _______________

3. _______________
   - Tasks: _______________
   - Dependencies: _______________
   - Estimated Time: _______________

---

## Part 4: Habit #4 - Clear Communication

### Exercise 4.1: Technical to Non-Technical Translation
Translate these technical explanations for different audiences:

**Technical:** "We'll use a random forest classifier with hyperparameter tuning via grid search to optimize F1 score"

**For Business Executive:**
_______________

**For Product Manager:**
_______________

**For Customer:**
_______________

### Exercise 4.2: Storytelling with Data
Create a narrative for these findings:

**Data Points:**
- Customer churn increased from 5% to 8%
- Average order value decreased by 15%
- Support tickets increased by 40%
- New competitor launched 3 months ago

**Your Data Story:**
```
Title: _______________

Introduction: _______________

Rising Action: _______________

Climax (Key Insight): _______________

Resolution (Recommendation): _______________
```

### Exercise 4.3: Visualization Choice
For each scenario, choose the best visualization and explain why:

| Scenario | Best Viz | Why? |
|----------|----------|------|
| Show market share of 5 companies | | |
| Display trend over 2 years | | |
| Compare distributions of two groups | | |
| Show relationship between two variables | | |
| Display part-to-whole relationship | | |

---

## Part 5: Habit #5 - Ethical Awareness

### Exercise 5.1: Bias Detection
Identify potential biases in these scenarios:

**Scenario A:** A hiring algorithm trained on historical hiring data (company was 80% male)
**Bias Type:** _______________
**Impact:** _______________
**Mitigation:** _______________

**Scenario B:** Facial recognition system trained primarily on Caucasian faces
**Bias Type:** _______________
**Impact:** _______________
**Mitigation:** _______________

**Scenario C:** Credit scoring model that uses ZIP code as a feature
**Bias Type:** _______________
**Impact:** _______________
**Mitigation:** _______________

### Exercise 5.2: Ethical Decision Framework
For each situation, work through this framework:

**Situation:** You discover your model performs worse for a minority group

1. **What's the impact?** _______________
2. **Who is affected?** _______________
3. **What are the alternatives?** _______________
4. **What's the right action?** _______________
5. **How to prevent in future?** _______________

### Exercise 5.3: Privacy Considerations
Review this code and identify privacy risks:

```python
# Customer analysis
df = pd.read_csv('customer_data.csv')

# Create report
report = df.groupby('name').agg({
    'ssn': 'first',  # Include SSN
    'email': 'first',
    'purchases': 'sum',
    'credit_score': 'mean'
})

# Share publicly
report.to_csv('public_report.csv')
print(report.head(20))  # Print to console
```

**Identified Risks:**
1. _______________
2. _______________
3. _______________

**Fixed Version:**
```python
# Your improved, privacy-aware code here
```

---

## Part 6: Integrated Case Study

### Exercise 6.1: Apply All Five Habits
Analyze this scenario using all five habits:

> **Scenario:** Your e-commerce recommendation model shows a 20% increase in click-through rate, but revenue is flat.

**Curiosity (Ask 5 questions):**
1. _______________
2. _______________
3. _______________
4. _______________
5. _______________

**Skepticism (What to verify):**
1. _______________
2. _______________
3. _______________

**Structured Approach (Steps to investigate):**
1. _______________
2. _______________
3. _______________

**Communication (Explain to stakeholders):**
_______________

**Ethics (Consider implications):**
_______________

---

## Part 7: Building Your Habit Tracker

### Exercise 7.1: Personal Habit Assessment
Rate yourself (1-5) on each habit:

| Habit | Current Level | Target | Action Plan |
|-------|---------------|--------|-------------|
| Curiosity | /5 | /5 | |
| Skepticism | /5 | /5 | |
| Structure | /5 | /5 | |
| Communication | /5 | /5 | |
| Ethics | /5 | /5 | |

### Exercise 7.2: 30-Day Challenge
Design a 30-day plan to improve your weakest habit:

**Target Habit:** _______________

**Week 1:** _______________
**Week 2:** _______________
**Week 3:** _______________
**Week 4:** _______________

**Daily Practice:** _______________
**Success Metric:** _______________

---

## Deliverables

Submit:
1. ✅ Five Whys analysis (Exercise 1.1)
2. ✅ 10 curious questions + code (Exercise 1.2)
3. ✅ Misleading visualization analysis (Exercise 2.1)
4. ✅ Skeptical questions (Exercise 2.2)
5. ✅ CRISP-DM application (Exercise 3.1)
6. ✅ Problem decomposition (Exercise 3.2)
7. ✅ Technical translations (Exercise 4.1)
8. ✅ Data story (Exercise 4.2)
9. ✅ Bias detection + mitigation (Exercise 5.1)
10. ✅ Privacy-fixed code (Exercise 5.3)
11. ✅ Integrated case study (Exercise 6.1)
12. ✅ 30-day improvement plan (Exercise 7.2)

---

## Reflection Questions

1. Which habit is your strongest? How did you develop it?
2. Which habit needs most work? Why?
3. How do these habits interact with each other?

---

**Next Lab:** Lab 06 - Problem Framing Workshop
