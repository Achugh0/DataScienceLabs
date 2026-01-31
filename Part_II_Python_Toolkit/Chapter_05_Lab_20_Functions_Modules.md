# Lab 20: Functions & Modules - Building Reusable Code

## Chapter 5: Python for Data Science

### Learning Objectives
- Write clean, reusable functions
- Handle errors with try/except
- Create custom modules
- Document code professionally

### Duration: 75 minutes

---

## Part 1: Basic Functions

### Exercise 1.1: Data Cleaning Functions
```python
def clean_email(email):
    """Clean and validate email address"""
    email = email.strip().lower()
    if '@' in email and '.' in email:
        return email
    return None

def clean_phone(phone):
    """Extract digits from phone number"""
    digits = ''.join(c for c in phone if c.isdigit())
    if len(digits) in [7, 10, 11]:
        return digits
    return None

def clean_name(name):
    """Standardize name formatting"""
    return name.strip().title()

# Test the functions
emails = ['  ALICE@test.com  ', 'invalid', 'bob@company.com']
for email in emails:
    cleaned = clean_email(email)
    print(f"{email} -> {cleaned}")

# YOUR CODE: Create clean_date function
# Input: '01/15/2024', '2024-01-15', 'Jan 15, 2024'
# Output: '2024-01-15' (standardized)
```

### Exercise 1.2: Statistical Functions
```python
def calculate_mean(values):
    """Calculate average of a list"""
    if not values:
        return 0
    return sum(values) / len(values)

def calculate_median(values):
    """Calculate median of a list"""
    if not values:
        return 0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    mid = n // 2
    
    if n % 2 == 0:
        return (sorted_vals[mid-1] + sorted_vals[mid]) / 2
    return sorted_vals[mid]

def calculate_mode(values):
    """Find most common value"""
    if not values:
        return None
    
    counts = {}
    for val in values:
        counts[val] = counts.get(val, 0) + 1
    
    return max(counts, key=counts.get)

# Test with sales data
daily_sales = [100, 150, 100, 200, 150, 100, 175]
print(f"Mean: ${calculate_mean(daily_sales):.2f}")
print(f"Median: ${calculate_median(daily_sales):.2f}")
print(f"Mode: ${calculate_mode(daily_sales):.2f}")

# YOUR CODE: Add calculate_std_dev function
```

---

## Part 2: Default Arguments & Keyword Arguments

### Exercise 2.1: Flexible Data Processing
```python
def filter_data(data, min_value=0, max_value=float('inf'), 
                sort=False, reverse=False):
    """
    Filter and optionally sort data
    
    Args:
        data: List of numbers
        min_value: Minimum threshold (default: 0)
        max_value: Maximum threshold (default: infinity)
        sort: Whether to sort results (default: False)
        reverse: Sort in descending order (default: False)
    
    Returns:
        Filtered (and optionally sorted) list
    """
    # Filter
    filtered = [x for x in data if min_value <= x <= max_value]
    
    # Sort if requested
    if sort:
        filtered.sort(reverse=reverse)
    
    return filtered

# Test different configurations
sales = [50, 150, 75, 200, 300, 25, 175]

print(filter_data(sales))  # All data
print(filter_data(sales, min_value=100))  # Above 100
print(filter_data(sales, min_value=50, max_value=200))  # Range
print(filter_data(sales, sort=True))  # Sorted
print(filter_data(sales, sort=True, reverse=True))  # Descending

# YOUR CODE: Create process_transactions function
# Parameters: transactions, currency='USD', tax_rate=0.0, 
#            discount=0.0, round_result=True
```

### Exercise 2.2: Report Generator
```python
def generate_report(data, title="Data Report", include_stats=True,
                   include_chart=False, format='text'):
    """Generate customizable data report"""
    
    report = []
    report.append(f"{'=' * 50}")
    report.append(f"{title:^50}")
    report.append(f"{'=' * 50}\n")
    
    # Basic info
    report.append(f"Total Records: {len(data)}")
    
    # Optional statistics
    if include_stats:
        report.append(f"Mean: {sum(data)/len(data):.2f}")
        report.append(f"Min: {min(data)}")
        report.append(f"Max: {max(data)}")
    
    # Optional chart (simple text chart)
    if include_chart:
        report.append("\nData Distribution:")
        for i, value in enumerate(data[:10]):  # First 10
            bar = '#' * int(value / 10)
            report.append(f"{i+1:2d}: {bar} ({value})")
    
    return '\n'.join(report)

# Test
sales = [100, 150, 200, 175, 125, 180, 140, 160, 190, 145]

print(generate_report(sales))
print("\n" + "="*50 + "\n")
print(generate_report(sales, title="Q1 Sales Report", 
                     include_chart=True))
```

---

## Part 3: Return Values & Multiple Returns

### Exercise 3.1: Analysis Functions
```python
def analyze_sales(sales_data):
    """
    Comprehensive sales analysis
    
    Returns tuple: (total, average, min, max, trend)
    """
    if not sales_data:
        return 0, 0, 0, 0, 'no data'
    
    total = sum(sales_data)
    average = total / len(sales_data)
    minimum = min(sales_data)
    maximum = max(sales_data)
    
    # Simple trend analysis
    if len(sales_data) >= 2:
        if sales_data[-1] > sales_data[0]:
            trend = 'increasing'
        elif sales_data[-1] < sales_data[0]:
            trend = 'decreasing'
        else:
            trend = 'stable'
    else:
        trend = 'insufficient data'
    
    return total, average, minimum, maximum, trend

# Unpack results
weekly_sales = [1200, 1350, 1280, 1450, 1500, 1420, 1480]
total, avg, min_val, max_val, trend = analyze_sales(weekly_sales)

print(f"Total: ${total:,.2f}")
print(f"Average: ${avg:,.2f}")
print(f"Range: ${min_val} - ${max_val}")
print(f"Trend: {trend}")

# YOUR CODE: Create validate_data function
# Return: (is_valid, errors, warnings, cleaned_data)
```

### Exercise 3.2: Data Transformation
```python
def transform_customer_data(raw_data):
    """
    Transform and validate customer records
    
    Returns:
        valid_records: List of cleaned records
        invalid_records: List of rejected records
        stats: Dictionary with processing stats
    """
    valid = []
    invalid = []
    
    for record in raw_data:
        # Validate
        if not record.get('email') or '@' not in record['email']:
            invalid.append(record)
            continue
        
        # Clean and transform
        cleaned = {
            'name': record['name'].strip().title(),
            'email': record['email'].strip().lower(),
            'age': int(record.get('age', 0))
        }
        valid.append(cleaned)
    
    stats = {
        'total': len(raw_data),
        'valid': len(valid),
        'invalid': len(invalid),
        'success_rate': len(valid) / len(raw_data) * 100 if raw_data else 0
    }
    
    return valid, invalid, stats

# Test
customers = [
    {'name': 'alice', 'email': 'alice@test.com', 'age': '25'},
    {'name': 'bob', 'email': 'invalid', 'age': '30'},
    {'name': 'charlie', 'email': 'charlie@test.com', 'age': '35'},
]

valid, invalid, stats = transform_customer_data(customers)
print(f"Processed: {stats['total']}")
print(f"Valid: {stats['valid']}")
print(f"Success Rate: {stats['success_rate']:.1f}%")
```

---

## Part 4: Error Handling

### Exercise 4.1: Safe Data Processing
```python
def safe_divide(a, b):
    """Division with error handling"""
    try:
        result = a / b
        return result, None
    except ZeroDivisionError:
        return None, "Cannot divide by zero"
    except TypeError:
        return None, "Invalid data types"

# Test
print(safe_divide(10, 2))  # (5.0, None)
print(safe_divide(10, 0))  # (None, 'Cannot divide by zero')
print(safe_divide('10', 2))  # (None, 'Invalid data types')

def process_value(value):
    """Convert and validate value"""
    try:
        # Try to convert to float
        num = float(value)
        
        # Validate range
        if num < 0:
            raise ValueError("Value cannot be negative")
        if num > 1000000:
            raise ValueError("Value exceeds maximum")
        
        return num, None
    
    except ValueError as e:
        return None, f"Validation error: {e}"
    except Exception as e:
        return None, f"Unexpected error: {e}"

# YOUR CODE: Create safe_parse_date function
# Handle multiple date formats
# Return (date_object, error_message)
```

### Exercise 4.2: Robust File Operations
```python
def read_data_file(filename):
    """Read data file with comprehensive error handling"""
    try:
        # Simulate reading file
        if not filename.endswith('.txt'):
            raise ValueError("Only .txt files supported")
        
        # Simulate file not found
        if 'missing' in filename:
            raise FileNotFoundError(f"File {filename} not found")
        
        # Simulate data
        data = ['100', '200', 'invalid', '300']
        
        # Parse data
        parsed = []
        errors = []
        
        for i, value in enumerate(data):
            try:
                parsed.append(float(value))
            except ValueError:
                errors.append(f"Line {i+1}: Invalid value '{value}'")
        
        return {
            'success': True,
            'data': parsed,
            'errors': errors,
            'message': f"Loaded {len(parsed)} values with {len(errors)} errors"
        }
    
    except FileNotFoundError as e:
        return {'success': False, 'message': str(e)}
    except ValueError as e:
        return {'success': False, 'message': str(e)}
    except Exception as e:
        return {'success': False, 'message': f"Unexpected error: {e}"}

# Test
result = read_data_file('data.txt')
print(result)

result = read_data_file('missing.txt')
print(result)
```

---

## Part 5: Lambda Functions & Functional Programming

### Exercise 5.1: Data Transformations
```python
# Lambda basics
square = lambda x: x ** 2
print(square(5))  # 25

# Practical use with map/filter
sales = [100, 150, 200, 175, 125]

# Add tax
with_tax = list(map(lambda x: x * 1.10, sales))
print(f"With tax: {with_tax}")

# Filter high sales
high_sales = list(filter(lambda x: x >= 150, sales))
print(f"High sales: {high_sales}")

# Sort by custom key
transactions = [
    {'date': '2024-01-03', 'amount': 100},
    {'date': '2024-01-01', 'amount': 200},
    {'date': '2024-01-02', 'amount': 150},
]

sorted_by_date = sorted(transactions, key=lambda x: x['date'])
sorted_by_amount = sorted(transactions, key=lambda x: x['amount'], reverse=True)

print("By date:", sorted_by_date)
print("By amount:", sorted_by_amount)

# YOUR CODE: Use lambda to:
# 1. Calculate discount (20% off amounts over 150)
# 2. Extract customer names from dict
# 3. Group transactions by date
```

### Exercise 5.2: Data Pipeline Functions
```python
def apply_pipeline(data, *functions):
    """Apply multiple functions in sequence"""
    result = data
    for func in functions:
        result = func(result)
    return result

# Define transformation functions
def remove_outliers(data, threshold=2):
    """Remove values > threshold * std dev from mean"""
    if len(data) < 2:
        return data
    mean = sum(data) / len(data)
    variance = sum((x - mean) ** 2 for x in data) / len(data)
    std_dev = variance ** 0.5
    return [x for x in data if abs(x - mean) <= threshold * std_dev]

def normalize(data):
    """Scale to 0-1 range"""
    if not data:
        return []
    min_val = min(data)
    max_val = max(data)
    if max_val == min_val:
        return [0.5] * len(data)
    return [(x - min_val) / (max_val - min_val) for x in data]

def round_values(data, decimals=2):
    """Round to specified decimals"""
    return [round(x, decimals) for x in data]

# Apply pipeline
raw_data = [100, 150, 200, 1000, 175, 125, 180]  # 1000 is outlier

cleaned = apply_pipeline(
    raw_data,
    remove_outliers,
    normalize,
    round_values
)

print(f"Raw: {raw_data}")
print(f"Cleaned: {cleaned}")
```

---

## Part 6: Modules & Code Organization

### Exercise 6.1: Create data_utils Module
```python
# Create file: data_utils.py

"""
Data utility functions for common operations
"""

def validate_email(email):
    """Validate email format"""
    return '@' in email and '.' in email

def validate_phone(phone):
    """Validate phone number"""
    digits = ''.join(c for c in str(phone) if c.isdigit())
    return len(digits) in [7, 10, 11]

def calculate_statistics(data):
    """Calculate common statistics"""
    if not data:
        return {}
    
    return {
        'count': len(data),
        'sum': sum(data),
        'mean': sum(data) / len(data),
        'min': min(data),
        'max': max(data),
    }

def format_currency(amount, currency='USD'):
    """Format amount as currency"""
    symbols = {'USD': '$', 'EUR': '€', 'GBP': '£'}
    symbol = symbols.get(currency, '$')
    return f"{symbol}{amount:,.2f}"

# To use this module:
# import data_utils
# or
# from data_utils import validate_email, calculate_statistics
```

### Exercise 6.2: Create sales_analysis Module
```python
# Create file: sales_analysis.py

"""
Sales data analysis functions
"""

def daily_revenue(transactions):
    """Calculate revenue by day"""
    revenue_by_day = {}
    for trans in transactions:
        date = trans['date']
        amount = trans['amount']
        revenue_by_day[date] = revenue_by_day.get(date, 0) + amount
    return revenue_by_day

def top_products(transactions, n=5):
    """Find top N selling products"""
    product_sales = {}
    for trans in transactions:
        product = trans['product']
        quantity = trans['quantity']
        product_sales[product] = product_sales.get(product, 0) + quantity
    
    # Sort and return top N
    sorted_products = sorted(product_sales.items(), 
                           key=lambda x: x[1], 
                           reverse=True)
    return sorted_products[:n]

def customer_lifetime_value(transactions):
    """Calculate total spent per customer"""
    clv = {}
    for trans in transactions:
        customer = trans['customer']
        amount = trans['amount']
        clv[customer] = clv.get(customer, 0) + amount
    return clv

# YOUR CODE: Add functions for:
# - average_order_value
# - sales_by_category
# - growth_rate
# - customer_segmentation
```

---

## Part 7: Decorators (Advanced)

### Exercise 7.1: Timing Function
```python
import time

def timing_decorator(func):
    """Measure function execution time"""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.4f} seconds")
        return result
    return wrapper

@timing_decorator
def process_large_dataset(size):
    """Simulate expensive operation"""
    data = list(range(size))
    result = sum(x ** 2 for x in data)
    return result

# Test
result = process_large_dataset(1000000)
```

### Exercise 7.2: Validation Decorator
```python
def validate_inputs(func):
    """Validate function inputs"""
    def wrapper(data, *args, **kwargs):
        if not isinstance(data, list):
            raise TypeError("Data must be a list")
        if not data:
            raise ValueError("Data cannot be empty")
        return func(data, *args, **kwargs)
    return wrapper

@validate_inputs
def analyze_data(data):
    """Analyze data"""
    return {
        'count': len(data),
        'mean': sum(data) / len(data)
    }

# Test
print(analyze_data([1, 2, 3, 4, 5]))
# print(analyze_data([]))  # Raises ValueError
```

---

## Deliverables

Build a complete data analysis module:

```python
# analysis_toolkit.py

class DataAnalyzer:
    """Complete data analysis toolkit"""
    
    def __init__(self, data):
        self.data = data
        self.cleaned_data = None
    
    def clean(self):
        """Remove invalid values"""
        pass
    
    def describe(self):
        """Generate statistical summary"""
        pass
    
    def filter(self, condition):
        """Filter data by condition"""
        pass
    
    def transform(self, func):
        """Apply transformation"""
        pass
    
    def export(self, filename):
        """Export results"""
        pass
```

---

## Challenge: Build ETL Framework

Create a reusable ETL (Extract, Transform, Load) framework with:

1. **Extractors**: Read from multiple sources (CSV, JSON, API)
2. **Transformers**: Chain multiple transformations
3. **Validators**: Check data quality
4. **Loaders**: Write to multiple destinations
5. **Logger**: Track all operations
6. **Error Handler**: Robust error recovery

```python
class ETLPipeline:
    def __init__(self):
        self.extractors = []
        self.transformers = []
        self.validators = []
        self.loaders = []
    
    def add_extractor(self, func):
        pass
    
    def add_transformer(self, func):
        pass
    
    def run(self):
        pass
```

**Next Chapter:** Chapter 6 - NumPy Fundamentals
