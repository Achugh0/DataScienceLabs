# Lab 02: Understanding the Data Science Workflow

## Chapter 1: Charting the Course: What Is Data Science?

### Learning Objectives
- Navigate a complete data science workflow
- Understand the iterative nature of data projects
- Practice problem framing

### Duration
60 minutes

### Prerequisites
- Lab 01 completed
- Google Colab account set up

---

## Part 1: The Iterative Nature of Data Science

### Exercise 1.1: Workflow Diagram
Draw a flowchart showing the data science workflow. Include:
- Decision points (diamonds)
- Process steps (rectangles)
- Feedback loops (arrows going back)

**Key elements to include:**
- Start with business problem
- Data collection
- EDA checkpoint
- Model building
- Evaluation checkpoint
- Deployment decision
- Monitoring feedback loop

---

## Part 2: From Business Problem to Data Problem

### Exercise 2.1: Problem Translation
For each business problem, translate it into a specific data science problem:

**Business Problem 1:** "We're losing customers"
- **Data Science Problem:** _____________________
- **Type of Problem:** Classification / Regression / Clustering / Other
- **Target Variable:** _____________________
- **Key Features (list 3):** _____________________

**Business Problem 2:** "We need to optimize our inventory"
- **Data Science Problem:** _____________________
- **Type of Problem:** Classification / Regression / Clustering / Other
- **Target Variable:** _____________________
- **Key Features (list 3):** _____________________

**Business Problem 3:** "Our marketing campaigns aren't effective"
- **Data Science Problem:** _____________________
- **Type of Problem:** Classification / Regression / Clustering / Other
- **Target Variable:** _____________________
- **Key Features (list 3):** _____________________

### Exercise 2.2: SMART Goals for Data Science
Rewrite these vague goals as SMART goals (Specific, Measurable, Achievable, Relevant, Time-bound):

**Vague:** "Improve customer satisfaction"
**SMART:** _____________________

**Vague:** "Make better predictions"
**SMART:** _____________________

**Vague:** "Understand our data"
**SMART:** _____________________

---

## Part 3: Data Requirements Planning

### Exercise 3.1: Data Audit
For the scenario below, identify what data you need:

> **Scenario:** You're building a model to predict which customers will subscribe to a premium service.

**Complete the data audit:**

| Data Category | Examples | Available? | How to Get It? |
|---------------|----------|------------|----------------|
| **Customer Demographics** | | ☐ Yes ☐ No ☐ Partial | |
| **Behavioral Data** | | ☐ Yes ☐ No ☐ Partial | |
| **Historical Purchases** | | ☐ Yes ☐ No ☐ Partial | |
| **Engagement Metrics** | | ☐ Yes ☐ No ☐ Partial | |
| **Customer Service Data** | | ☐ Yes ☐ No ☐ Partial | |

### Exercise 3.2: Data Quality Checklist
For a new dataset, what questions should you ask? Complete this checklist:

**Completeness:**
- [ ] _____________________
- [ ] _____________________

**Accuracy:**
- [ ] _____________________
- [ ] _____________________

**Consistency:**
- [ ] _____________________
- [ ] _____________________

**Timeliness:**
- [ ] _____________________
- [ ] _____________________

**Relevance:**
- [ ] _____________________
- [ ] _____________________

---

## Part 4: First Python Workflow

### Exercise 4.1: Hello Data Science (Code)
Open Google Colab and create a new notebook. Run this workflow:

```python
# Step 1: Import libraries
import pandas as pd
import numpy as np

# Step 2: Create sample data
data = {
    'customer_id': [1, 2, 3, 4, 5],
    'age': [25, 34, 28, 42, 31],
    'purchases': [5, 12, 7, 20, 9],
    'subscription': ['No', 'Yes', 'No', 'Yes', 'No']
}

# Step 3: Create DataFrame
df = pd.DataFrame(data)

# Step 4: Explore the data
print("Dataset Shape:", df.shape)
print("\nFirst Few Rows:")
print(df.head())
print("\nData Types:")
print(df.dtypes)
print("\nBasic Statistics:")
print(df.describe())

# Step 5: Simple analysis
avg_age = df['age'].mean()
total_purchases = df['purchases'].sum()

print(f"\nAverage Age: {avg_age}")
print(f"Total Purchases: {total_purchases}")
```

**Your Tasks:**
1. Run this code
2. Add a comment explaining what each step does
3. Modify the code to answer: "What's the average number of purchases for subscribers vs. non-subscribers?"
4. Take a screenshot of your output

---

## Part 5: Case Study Walkthrough

### Exercise 5.1: Netflix Case Study
Read this simplified scenario:

> Netflix wants to recommend movies to users. They have data on:
> - User watch history
> - Movie ratings
> - Genre preferences
> - Watch time patterns

**Map out the workflow:**

1. **Business Objective:**
   - _____________________

2. **Data Science Problem:**
   - _____________________

3. **Data Sources:**
   - _____________________
   - _____________________
   - _____________________

4. **Key Challenges:**
   - _____________________
   - _____________________

5. **Success Metrics:**
   - _____________________
   - _____________________

6. **Potential Pitfalls:**
   - _____________________
   - _____________________

### Exercise 5.2: Your Own Case Study
Choose a company or industry you're interested in. Design a mini data science project:

**Company/Industry:** _____________________

**Business Problem:** _____________________

**Data Science Solution:** _____________________

**Required Data:** _____________________

**Expected Impact:** _____________________

**Timeline:** _____________________

---

## Part 6: Communication Planning

### Exercise 6.1: Stakeholder Analysis
For the Netflix case study, identify stakeholders and their needs:

| Stakeholder | Their Question | What They Need to See | Technical Level |
|-------------|----------------|------------------------|-----------------|
| Product Manager | | | |
| Engineering Lead | | | |
| Executive Team | | | |
| End Users | | | |

---

## Deliverables

Submit:
1. ✅ Workflow diagram (Exercise 1.1)
2. ✅ Three translated business problems (Exercise 2.1)
3. ✅ SMART goals (Exercise 2.2)
4. ✅ Data audit table (Exercise 3.1)
5. ✅ Data quality checklist (Exercise 3.2)
6. ✅ Modified Python code and screenshot (Exercise 4.1)
7. ✅ Netflix case study workflow (Exercise 5.1)
8. ✅ Your own case study design (Exercise 5.2)
9. ✅ Stakeholder analysis (Exercise 6.1)

---

## Reflection Questions

1. What's the hardest part of translating business problems to data problems?
2. Why is the workflow iterative rather than linear?
3. What happens if you skip the problem definition phase?

---

**Next Lab:** Lab 03 - Data Types and Sources
