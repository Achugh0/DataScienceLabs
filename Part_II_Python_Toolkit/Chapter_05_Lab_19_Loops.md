# Lab 19: Loops - Processing Data at Scale

## Chapter 5: Python for Data Science

### Learning Objectives
- Master for and while loops for data processing
- Process large datasets efficiently
- Build ETL pipelines

### Duration: 60 minutes

---

## Part 1: Process Real CSV Data

### Exercise 1.1: Sales Data Analysis
```python
# Simulate reading CSV data
sales_data = [
    ['2024-01-01', 'Product A', 100, 29.99],
    ['2024-01-01', 'Product B', 50, 49.99],
    ['2024-01-02', 'Product A', 75, 29.99],
    ['2024-01-02', 'Product C', 200, 19.99],
    ['2024-01-03', 'Product B', 60, 49.99],
]

# Process with loops
total_revenue = 0
products_sold = {}

for row in sales_data:
    date, product, quantity, price = row
    revenue = quantity * price
    total_revenue += revenue
    
    if product in products_sold:
        products_sold[product] += quantity
    else:
        products_sold[product] = quantity

print(f"Total Revenue: ${total_revenue:,.2f}")
print("Products Sold:", products_sold)

# YOUR CODE: Calculate
# 1. Revenue by day
# 2. Best-selling product
# 3. Average order value
# 4. Total units sold
```

### Exercise 1.2: Clean Customer Data
```python
# Messy customer data
customers = [
    {'name': '  Alice Johnson  ', 'email': 'ALICE@EMAIL.COM', 'phone': '555-1234'},
    {'name': 'bob smith', 'email': 'bob@email.com', 'phone': '(555) 2345'},
    {'name': 'Charlie_Brown', 'email': 'charlie@EMAIL.COM', 'phone': '555.3456'},
]

# Clean data
cleaned_customers = []
for customer in customers:
    cleaned = {
        'name': customer['name'].strip().title().replace('_', ' '),
        'email': customer['email'].strip().lower(),
        'phone': ''.join(c for c in customer['phone'] if c.isdigit())
    }
    cleaned_customers.append(cleaned)

# YOUR CODE: Add validation
# Skip records with:
# - Invalid email (no @)
# - Invalid phone (not 7 or 10 digits)
# - Empty name
```

---

## Part 2: While Loops for User Input

### Exercise 2.1: Data Entry System
```python
def data_entry_system():
    """Collect user data until done"""
    entries = []
    
    while True:
        name = input("Enter name (or 'done' to finish): ")
        if name.lower() == 'done':
            break
        
        try:
            age = int(input("Enter age: "))
            if age < 0 or age > 120:
                print("Invalid age!")
                continue
        except ValueError:
            print("Please enter a number!")
            continue
        
        entries.append({'name': name, 'age': age})
        print(f"Added {name}")
    
    return entries

# YOUR CODE: Extend this
# Add email collection
# Add data validation
# Show summary stats at end
```

### Exercise 2.2: Retry Logic
```python
import random

def fetch_data_with_retry(max_retries=3):
    """Simulate API call with retries"""
    attempts = 0
    
    while attempts < max_retries:
        attempts += 1
        print(f"Attempt {attempts}...")
        
        # Simulate 70% success rate
        if random.random() > 0.3:
            print("Success!")
            return True
        else:
            print("Failed, retrying...")
    
    print("Max retries reached")
    return False

# YOUR CODE: Build production retry logic
def api_call_with_exponential_backoff():
    """Retry with increasing delays"""
    # Wait 1s, 2s, 4s, 8s, etc.
    pass
```

---

## Part 3: Nested Loops (Matrix Operations)

### Exercise 3.1: Process 2D Data
```python
# Student grades by subject
grades = [
    [85, 90, 78],  # Student 1
    [92, 88, 95],  # Student 2
    [76, 82, 80],  # Student 3
]

# Calculate averages
student_averages = []
for student_grades in grades:
    avg = sum(student_grades) / len(student_grades)
    student_averages.append(avg)

print("Student Averages:", student_averages)

# Subject averages (transpose)
num_subjects = len(grades[0])
subject_averages = []

for subject_idx in range(num_subjects):
    subject_total = 0
    for student_grades in grades:
        subject_total += student_grades[subject_idx]
    subject_avg = subject_total / len(grades)
    subject_averages.append(subject_avg)

print("Subject Averages:", subject_averages)

# YOUR CODE: Find
# 1. Highest grade overall
# 2. Which student has highest average
# 3. Which subject has highest average
# 4. Count grades above 85
```

### Exercise 3.2: Transaction Matching
```python
# Match orders to shipments
orders = [
    {'order_id': 1, 'customer': 'Alice', 'amount': 100},
    {'order_id': 2, 'customer': 'Bob', 'amount': 150},
    {'order_id': 3, 'customer': 'Charlie', 'amount': 200},
]

shipments = [
    {'order_id': 2, 'tracking': 'TRACK123', 'status': 'shipped'},
    {'order_id': 1, 'tracking': 'TRACK456', 'status': 'delivered'},
]

# Match and combine
matched = []
for order in orders:
    order_data = order.copy()
    order_data['shipment'] = None
    
    for shipment in shipments:
        if shipment['order_id'] == order['order_id']:
            order_data['shipment'] = shipment
            break
    
    matched.append(order_data)

# YOUR CODE: Find unshipped orders
# Calculate delivery rate
```

---

## Part 4: Loop Control (break, continue, else)

### Exercise 4.1: Search and Stop
```python
# Find first customer over credit limit
transactions = [
    {'customer': 'Alice', 'amount': 50},
    {'customer': 'Bob', 'amount': 1500},  # Over limit!
    {'customer': 'Charlie', 'amount': 100},
]

credit_limit = 1000
found = False

for trans in transactions:
    if trans['amount'] > credit_limit:
        print(f"Alert: {trans['customer']} exceeded limit!")
        found = True
        break

if not found:
    print("All transactions OK")

# YOUR CODE: Fraud detection
# Stop processing if:
# - Amount > $5000
# - More than 5 transactions in batch
# - Negative amount
```

### Exercise 4.2: Skip Invalid Records
```python
# Process data, skip errors
data = [
    {'name': 'Alice', 'score': 85},
    {'name': 'Bob', 'score': -1},  # Invalid
    {'name': '', 'score': 90},     # Invalid
    {'name': 'Charlie', 'score': 78},
]

valid_count = 0
for record in data:
    # Skip invalid
    if not record['name']:
        continue
    if record['score'] < 0 or record['score'] > 100:
        continue
    
    # Process valid record
    print(f"{record['name']}: {record['score']}")
    valid_count += 1

print(f"Processed {valid_count} valid records")

# YOUR CODE: Log invalid records separately
```

---

## Part 5: Real ETL Pipeline

### Exercise 5.1: Complete Data Pipeline
```python
def etl_pipeline(raw_data):
    """Extract, Transform, Load"""
    
    # Extract
    print("Extracting data...")
    extracted = []
    for record in raw_data:
        if isinstance(record, dict) and 'value' in record:
            extracted.append(record)
    print(f"Extracted {len(extracted)} records")
    
    # Transform
    print("Transforming data...")
    transformed = []
    for record in extracted:
        try:
            value = float(record['value'])
            if value > 0:  # Filter negatives
                record['value'] = value
                record['value_squared'] = value ** 2
                transformed.append(record)
        except (ValueError, TypeError):
            continue  # Skip bad data
    print(f"Transformed {len(transformed)} records")
    
    # Load (aggregate)
    print("Loading data...")
    total = sum(r['value'] for r in transformed)
    avg = total / len(transformed) if transformed else 0
    
    return {
        'records_processed': len(transformed),
        'total': total,
        'average': avg,
        'data': transformed
    }

# Test with messy data
raw = [
    {'value': '100'},
    {'value': '200'},
    {'value': 'invalid'},
    {'value': -50},
    {'value': '75.5'},
]

result = etl_pipeline(raw)
print(result)

# YOUR CODE: Extend pipeline
# - Add timestamp
# - Calculate running total
# - Detect outliers (>2 std devs)
# - Write summary report
```

### Exercise 5.2: Batch Processing
```python
def process_in_batches(data, batch_size=10):
    """Process large dataset in batches"""
    total_records = len(data)
    processed = 0
    
    for i in range(0, total_records, batch_size):
        batch = data[i:i + batch_size]
        
        print(f"Processing batch {i//batch_size + 1}...")
        
        # Process batch
        for record in batch:
            # Your processing logic
            processed += 1
        
        progress = (processed / total_records) * 100
        print(f"Progress: {progress:.1f}%")
    
    return processed

# Simulate large dataset
large_dataset = [{'id': i, 'value': i*10} for i in range(100)]

# Process in batches
processed_count = process_in_batches(large_dataset, batch_size=25)
print(f"Processed {processed_count} records")

# YOUR CODE: Add
# - Error handling per batch
# - Summary stats per batch
# - Ability to resume from failure
```

---

## Part 6: Performance Optimization

### Exercise 6.1: Enumerate for Index + Value
```python
# Don't do this
names = ['Alice', 'Bob', 'Charlie']
for i in range(len(names)):
    print(f"{i}: {names[i]}")

# Do this instead
for i, name in enumerate(names):
    print(f"{i}: {name}")

# Practical use: Track position
sales = [100, 200, 150, 300, 250]
peaks = []

for i, value in enumerate(sales):
    if i > 0 and i < len(sales) - 1:
        if value > sales[i-1] and value > sales[i+1]:
            peaks.append((i, value))

print(f"Peak sales at positions: {peaks}")
```

### Exercise 6.2: Zip for Parallel Iteration
```python
# Process multiple lists together
names = ['Alice', 'Bob', 'Charlie']
scores = [85, 92, 78]
grades = ['B', 'A', 'C']

# Don't use indices
for i in range(len(names)):
    print(f"{names[i]}: {scores[i]} ({grades[i]})")

# Use zip
for name, score, grade in zip(names, scores, grades):
    print(f"{name}: {score} ({grade})")

# YOUR CODE: Combine datasets
dates = ['2024-01-01', '2024-01-02', '2024-01-03']
products = ['A', 'B', 'C']
quantities = [100, 150, 200]
prices = [29.99, 39.99, 19.99]

# Create combined sales report
```

---

## Deliverables

Build these systems:

1. ✅ CSV data processor
2. ✅ Data cleaning pipeline
3. ✅ Batch processing system
4. ✅ ETL pipeline with error handling
5. ✅ Transaction matching system

---

## Challenge: Web Scraping Simulator

```python
def scrape_and_process(pages):
    """
    Simulate scraping multiple pages
    - Retry failed pages
    - Clean extracted data
    - Aggregate results
    """
    results = []
    failed_pages = []
    
    for page_num in range(1, pages + 1):
        max_retries = 3
        for attempt in range(max_retries):
            # Simulate scraping (70% success)
            if random.random() > 0.3:
                # Extract data
                data = extract_page_data(page_num)
                # Clean data
                cleaned = clean_data(data)
                results.extend(cleaned)
                break
            else:
                if attempt == max_retries - 1:
                    failed_pages.append(page_num)
    
    return results, failed_pages

# Implement the helper functions
```

**Next Lab:** Lab 20 - Functions and Modules
