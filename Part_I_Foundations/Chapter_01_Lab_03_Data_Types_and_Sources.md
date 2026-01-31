# Lab 03: Data Types and Sources

## Chapter 1: Charting the Course: What Is Data Science?

### Learning Objectives
- Distinguish between structured, semi-structured, and unstructured data
- Identify common data sources
- Practice accessing real-world datasets

### Duration
60 minutes

---

## Part 1: Understanding Data Types

### Exercise 1.1: Data Type Classification
Classify each example as Structured, Semi-Structured, or Unstructured:

1. Customer transaction table in a SQL database: _______________
2. JSON API response from Twitter: _______________
3. PDF research papers: _______________
4. Excel spreadsheet with sales data: _______________
5. Email content: _______________
6. XML configuration files: _______________
7. Images from security cameras: _______________
8. Server logs: _______________
9. Audio recordings of customer service calls: _______________
10. CSV file of sensor readings: _______________

### Exercise 1.2: Data Structure Design
For each scenario, design an appropriate data structure:

**Scenario A: E-commerce Orders**
```
Table Name: orders
Columns needed:
- _______________
- _______________
- _______________
- _______________
- _______________
```

**Scenario B: Social Media Posts**
```
JSON Structure:
{
  "_______________": _______________,
  "_______________": _______________,
  "_______________": _______________
}
```

---

## Part 2: Exploring Data Sources

### Exercise 2.1: Data Source Catalog
Research and document sources for each type:

**Public APIs:**
1. _______________ (URL: _______________)
2. _______________ (URL: _______________)
3. _______________ (URL: _______________)

**Open Data Repositories:**
1. _______________ (URL: _______________)
2. _______________ (URL: _______________)
3. _______________ (URL: _______________)

**Web Scraping Targets:**
1. _______________ (URL: _______________)
2. _______________ (URL: _______________)

**Database Types:**
1. _______________
2. _______________
3. _______________

### Exercise 2.2: Kaggle Exploration
1. Create a free Kaggle account (kaggle.com)
2. Browse datasets
3. Find three interesting datasets:

| Dataset Name | Domain | Size | Potential Question |
|--------------|---------|------|-------------------|
| | | | |
| | | | |
| | | | |

---

## Part 3: Hands-On Data Access

### Exercise 3.1: Loading Data in Python
In Google Colab, practice loading different data formats:

```python
import pandas as pd
import json

# 1. Load CSV from URL
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df_csv = pd.read_csv(url)
print("CSV Shape:", df_csv.shape)
print(df_csv.head())

# 2. Create and load JSON
json_data = '''
{
  "users": [
    {"id": 1, "name": "Alice", "score": 85},
    {"id": 2, "name": "Bob", "score": 92}
  ]
}
'''
data = json.loads(json_data)
df_json = pd.DataFrame(data['users'])
print("\nJSON DataFrame:")
print(df_json)

# 3. Create DataFrame from dictionary
dict_data = {
    'product': ['A', 'B', 'C'],
    'price': [10, 20, 15],
    'stock': [100, 50, 75]
}
df_dict = pd.DataFrame(dict_data)
print("\nDictionary DataFrame:")
print(df_dict)
```

**Your Tasks:**
1. Run this code
2. Modify to load a different dataset from Kaggle (download and upload to Colab)
3. Print the first 10 rows
4. Print column names
5. Print data types of each column

---

## Part 4: Data Quality Assessment

### Exercise 4.1: Quality Checklist
Using the Titanic dataset from Exercise 3.1, assess quality:

```python
# Your code here to check:
# 1. Missing values
# 2. Duplicate rows
# 3. Data types
# 4. Value ranges
# 5. Unexpected values
```

**Document findings:**
- Missing values: _______________
- Duplicates found: _______________
- Type issues: _______________
- Outliers noted: _______________

### Exercise 4.2: Data Profiling
Create a data profile report:

```python
# Calculate for each numeric column:
# - Mean
# - Median
# - Min/Max
# - Standard deviation
# - Quartiles

# Your code here
```

---

## Part 5: Real-World Data Scenario

### Exercise 5.1: Multi-Source Integration
You're analyzing a food delivery service. You have:
- **SQL Database:** Restaurant info, menu items
- **JSON API:** Real-time order data
- **CSV Files:** Customer reviews
- **Logs:** Delivery driver locations

**Plan the integration:**
1. What's your primary key to join data?
2. What challenges might you face?
3. What preprocessing is needed?
4. Design the final unified table structure

---

## Deliverables

Submit:
1. ✅ Data type classification (Exercise 1.1)
2. ✅ Data structure designs (Exercise 1.2)
3. ✅ Data source catalog (Exercise 2.1)
4. ✅ Three Kaggle datasets (Exercise 2.2)
5. ✅ Modified Python code with new dataset (Exercise 3.1)
6. ✅ Quality assessment findings (Exercise 4.1)
7. ✅ Data profiling code and results (Exercise 4.2)
8. ✅ Multi-source integration plan (Exercise 5.1)

---

## Bonus Challenge

Access a real API (e.g., OpenWeatherMap, NASA, REST Countries) and:
1. Make an API request
2. Parse the JSON response
3. Convert to pandas DataFrame
4. Analyze the data

---

**Next Lab:** Lab 04 - Setting Up Your Complete Data Science Environment
