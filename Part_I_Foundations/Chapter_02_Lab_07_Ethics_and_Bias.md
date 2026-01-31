# Lab 07: Data Ethics and Bias

## Chapter 2: The Data Scientist's Mindset

### Learning Objectives
- Identify sources of bias in data and models
- Understand privacy and transparency requirements
- Apply ethical frameworks to data science decisions

### Duration
90 minutes

---

## Part 1: Types of Bias

### Exercise 1.1: Bias Identification
Identify the type of bias in each scenario:

**Types:** Selection Bias, Measurement Bias, Confirmation Bias, Historical Bias, Algorithmic Bias

| Scenario | Bias Type | Explanation |
|----------|-----------|-------------|
| Training face recognition only on celebrity photos | | |
| Survey conducted only via landline phones | | |
| Researcher only analyzing data that supports hypothesis | | |
| Hiring algorithm trained on historically male-dominated field | | |
| Self-reported height data (people tend to round up) | | |

### Exercise 1.2: Bias Detection in Code
Analyze this dataset for bias:

```python
import pandas as pd
import numpy as np

# Loan approval data
data = {
    'age': [25, 35, 28, 45, 32, 50, 29, 38],
    'income': [40000, 80000, 45000, 95000, 55000, 100000, 42000, 85000],
    'zip_code': ['10001', '94102', '10001', '94102', '60601', '94102', '10001', '60601'],
    'credit_score': [650, 750, 680, 780, 700, 800, 670, 760],
    'approved': [0, 1, 0, 1, 1, 1, 0, 1]
}
df = pd.DataFrame(data)

# Your analysis:
# 1. Check approval rates by protected attributes (if proxy exists)
# 2. Analyze feature correlations
# 3. Look for disparate impact

# Your code here:
```

**Questions to investigate:**
1. Is there geographic bias (zip code)?
2. Is there age discrimination?
3. What's the approval rate by income level?
4. Could any features be proxies for protected attributes?

---

## Part 2: Fairness Metrics

### Exercise 2.1: Understanding Fairness Definitions
Compare these fairness definitions:

**Scenario:** Credit scoring model

| Fairness Definition | What It Means | How to Measure | Trade-offs |
|---------------------|---------------|----------------|------------|
| **Demographic Parity** | | | |
| **Equal Opportunity** | | | |
| **Equalized Odds** | | | |
| **Predictive Parity** | | | |

### Exercise 2.2: Calculate Fairness Metrics
```python
import pandas as pd
from sklearn.metrics import confusion_matrix

# Model predictions for two groups
group_a_actual = [1, 1, 0, 0, 1, 0, 1, 1, 0, 0]
group_a_pred =   [1, 0, 0, 0, 1, 0, 1, 1, 0, 1]

group_b_actual = [1, 1, 0, 0, 1, 0, 1, 1, 0, 0]
group_b_pred =   [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]

# Calculate for each group:
# 1. Positive prediction rate
# 2. True positive rate (recall)
# 3. False positive rate
# 4. Precision

def fairness_metrics(y_true, y_pred, group_name):
    # Your code here
    pass

fairness_metrics(group_a_actual, group_a_pred, "Group A")
fairness_metrics(group_b_actual, group_b_pred, "Group B")

# Compare the groups - is the model fair?
```

---

## Part 3: Privacy and Data Protection

### Exercise 3.1: PII Identification
Review this dataset and identify all PII (Personally Identifiable Information):

```python
customer_data = {
    'customer_id': [1, 2, 3],
    'name': ['John Doe', 'Jane Smith', 'Bob Johnson'],
    'email': ['john@email.com', 'jane@email.com', 'bob@email.com'],
    'age': [35, 28, 42],
    'zip_code': ['10001', '94102', '60601'],
    'last_purchase': ['2024-01-15', '2024-01-10', '2024-01-20'],
    'total_spent': [500, 1200, 800],
    'ip_address': ['192.168.1.1', '192.168.1.2', '192.168.1.3']
}
```

**PII Fields:**
- Direct PII: _______________
- Quasi-identifiers: _______________
- Sensitive Attributes: _______________

**De-identification Strategy:**
```python
# Your code to anonymize the data
def anonymize_data(df):
    # 1. Remove direct identifiers
    # 2. Generalize quasi-identifiers
    # 3. Add noise where appropriate
    pass
```

### Exercise 3.2: K-Anonymity
Implement k-anonymity:

```python
import pandas as pd

# Medical data
data = {
    'age': [25, 26, 27, 25, 26],
    'zip': ['12345', '12345', '12346', '12345', '12347'],
    'condition': ['flu', 'cold', 'flu', 'cold', 'flu']
}
df = pd.DataFrame(data)

# Achieve 2-anonymity by generalizing age and zip
# Your code here
```

---

## Part 4: Transparency and Explainability

### Exercise 4.1: Model Documentation Template
Create documentation for a model:

**MODEL CARD**

**Model Name:** _______________

**Model Type:** _______________

**Intended Use:**
- Primary: _______________
- Out of Scope: _______________

**Training Data:**
- Source: _______________
- Size: _______________
- Time Period: _______________
- Demographics: _______________

**Performance:**
- Overall Accuracy: _______________
- Performance by Group:
  - Group A: _______________
  - Group B: _______________

**Limitations:**
- _______________
- _______________

**Ethical Considerations:**
- _______________
- _______________

**Maintenance:**
- Retraining Frequency: _______________
- Monitoring Metrics: _______________

### Exercise 4.2: Explainable Predictions
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

# Train a simple model
# (Assume X_train, y_train, X_test exist)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# Feature importance
importance = model.feature_importances_

# Plot
plt.barh(range(len(importance)), importance)
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance')
plt.show()

# For a single prediction, explain:
# Your code here using SHAP or LIME
```

---

## Part 5: Ethical Decision-Making Framework

### Exercise 5.1: The Ethics Checklist
For each decision point, complete the checklist:

**Scenario:** You're building a predictive policing model

**ETHICS CHECKLIST:**

**1. Beneficence (Doing Good)**
- Who benefits? _______________
- What's the positive impact? _______________

**2. Non-maleficence (Avoiding Harm)**
- Who could be harmed? _______________
- What are the risks? _______________

**3. Autonomy**
- Do people have choice? _______________
- Is consent informed? _______________

**4. Justice**
- Are burdens distributed fairly? _______________
- Are benefits distributed fairly? _______________

**5. Transparency**
- Can decisions be explained? _______________
- Is the process transparent? _______________

**Decision:** Proceed / Modify / Cancel

**Justification:** _______________

### Exercise 5.2: Ethical Dilemmas
Analyze these dilemmas:

**Dilemma 1:** Your model is 90% accurate overall but only 70% accurate for a minority group. Do you deploy it?

**Your Analysis:**
- Stakeholders affected: _______________
- Consequences of deployment: _______________
- Consequences of non-deployment: _______________
- Alternative approaches: _______________
- Your decision: _______________

**Dilemma 2:** You're asked to predict employee performance to guide promotions. Historical data shows bias.

**Your Analysis:**
- Ethical concerns: _______________
- Data bias issues: _______________
- Perpetuation risk: _______________
- Mitigation strategies: _______________
- Your recommendation: _______________

---

## Part 6: Real-World Ethics Cases

### Exercise 6.1: Case Study - Healthcare
**Scenario:** An algorithm to allocate healthcare resources was found to discriminate against Black patients because it used healthcare costs as a proxy for health needs.

**Analysis:**
1. **What went wrong?** _______________
2. **Why did this happen?** _______________
3. **Who was harmed?** _______________
4. **How could it be prevented?** _______________
5. **Lessons learned:** _______________

### Exercise 6.2: Case Study - Hiring
**Scenario:** Amazon's hiring algorithm showed bias against women because it was trained on historical hires (predominantly male).

**Analysis:**
1. **Root cause:** _______________
2. **Type of bias:** _______________
3. **Warning signs that were missed:** _______________
4. **Better approach:** _______________
5. **Policy changes needed:** _______________

### Exercise 6.3: Case Study - Criminal Justice
**Scenario:** COMPAS risk assessment tool showed racial disparities in false positive rates.

**Analysis:**
1. **Fairness definition violated:** _______________
2. **Impact on society:** _______________
3. **Transparency issues:** _______________
4. **Alternative approaches:** _______________
5. **Oversight mechanisms needed:** _______________

---

## Part 7: Building an Ethics Policy

### Exercise 7.1: Create Data Ethics Guidelines
Draft guidelines for your organization:

**DATA ETHICS POLICY**

**1. Data Collection**
- Principle: _______________
- Guidelines: _______________
- Red lines: _______________

**2. Data Usage**
- Principle: _______________
- Guidelines: _______________
- Red lines: _______________

**3. Model Development**
- Principle: _______________
- Guidelines: _______________
- Red lines: _______________

**4. Deployment**
- Principle: _______________
- Guidelines: _______________
- Red lines: _______________

**5. Monitoring**
- Principle: _______________
- Guidelines: _______________
- Red lines: _______________

### Exercise 7.2: Ethics Review Board
Design an ethics review process:

**REVIEW CHECKLIST:**

**Phase 1: Pre-Project**
- [ ] Clear beneficial purpose defined
- [ ] Potential harms identified
- [ ] Alternative approaches considered
- [ ] Stakeholders consulted

**Phase 2: Development**
- [ ] Training data audited for bias
- [ ] Fairness metrics defined
- [ ] Protected attributes identified
- [ ] Baseline fairness established

**Phase 3: Pre-Deployment**
- [ ] Subgroup performance analyzed
- [ ] Explainability mechanisms in place
- [ ] Documentation complete
- [ ] Monitoring plan established

**Phase 4: Post-Deployment**
- [ ] Regular fairness audits
- [ ] Feedback mechanisms
- [ ] Incident response plan
- [ ] Retraining schedule

---

## Part 8: Practical Bias Mitigation

### Exercise 8.1: Bias Mitigation Techniques
Implement bias mitigation:

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load biased dataset
df = pd.read_csv('biased_data.csv')

# Technique 1: Balanced sampling
def balance_dataset(df, target_col, sensitive_col):
    # Your code to balance across sensitive attribute
    pass

# Technique 2: Reweighting
def reweight_samples(df, target_col, sensitive_col):
    # Assign weights to balance across groups
    pass

# Technique 3: Threshold optimization
def optimize_thresholds(model, X, y, sensitive_attr):
    # Find optimal thresholds per group
    pass

# Implement and compare results
```

---

## Deliverables

Submit:
1. ✅ Bias identification table (Exercise 1.1)
2. ✅ Bias detection code and analysis (Exercise 1.2)
3. ✅ Fairness metrics calculation (Exercise 2.2)
4. ✅ Anonymization code (Exercise 3.1)
5. ✅ K-anonymity implementation (Exercise 3.2)
6. ✅ Model card documentation (Exercise 4.1)
7. ✅ Ethics checklist (Exercise 5.1)
8. ✅ Ethical dilemma analyses (Exercise 5.2)
9. ✅ Three case study analyses (Exercises 6.1-6.3)
10. ✅ Data ethics policy (Exercise 7.1)
11. ✅ Ethics review checklist (Exercise 7.2)
12. ✅ Bias mitigation code (Exercise 8.1)

---

## Reflection Questions

1. Can a model be "fair" to everyone? Why or why not?
2. What's more important: transparency or performance?
3. Who should be responsible for ethical AI decisions?

---

**Next Lab:** Lab 08 - Descriptive Statistics Fundamentals
