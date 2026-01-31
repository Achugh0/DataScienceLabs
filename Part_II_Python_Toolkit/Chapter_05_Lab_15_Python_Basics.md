# Lab 15: Python Fundamentals - Variables and Data Types

## Chapter 5: Python for Data Science — Your First Steps in Code

### Learning Objectives
- Master Python variables and data types
- Work with strings, numbers, and booleans
- Understand type conversion and operations

### Duration
60 minutes

---

## Part 1: Variables and Assignment

### Exercise 1.1: Variable Basics
```python
# Create variables
name = "Data Scientist"
age = 25
salary = 85000.50
is_employed = True

# Print variables
print(f"Name: {name}")
print(f"Age: {age}")
print(f"Salary: ${salary}")
print(f"Employed: {is_employed}")

# Your turn: Create variables for yourself
my_name = _______________
my_age = _______________
my_city = _______________
loves_python = _______________
```

### Exercise 1.2: Naming Conventions
Fix these variable names:

```python
# Bad names - fix them
1st_name = "John"  # _______________
First-Name = "John"  # _______________
class = "Data Science"  # _______________
MY VARIABLE = 100  # _______________

# Good names
first_name = "John"
user_count = 100
CONSTANT_VALUE = 3.14
```

### Exercise 1.3: Multiple Assignment
```python
# Multiple assignment
x, y, z = 1, 2, 3
print(x, y, z)

# Same value to multiple variables
a = b = c = 0
print(a, b, c)

# Swapping variables
x, y = 5, 10
print(f"Before swap: x={x}, y={y}")
# Your code to swap
_______________
print(f"After swap: x={x}, y={y}")
```

---

## Part 2: Numeric Types

### Exercise 2.1: Integers and Floats
```python
# Integer operations
a = 10
b = 3

addition = a + b
subtraction = a - b
multiplication = a * b
division = a / b  # Float division
floor_division = a // b  # Integer division
modulus = a % b  # Remainder
power = a ** b  # Exponentiation

print(f"Addition: {addition}")
print(f"Division: {division}")
print(f"Floor Division: {floor_division}")
print(f"Modulus: {modulus}")

# Your turn: Calculate
num1 = 17
num2 = 5

# What's the quotient and remainder when dividing num1 by num2?
quotient = _______________
remainder = _______________
```

### Exercise 2.2: Float Precision
```python
# Be aware of float precision
result = 0.1 + 0.2
print(result)  # Not exactly 0.3!

# Use round() for display
print(round(result, 2))

# Your turn: Calculate average
scores = [85, 92, 78, 95, 88]
average = sum(scores) / len(scores)
print(f"Average: {round(average, 2)}")

# Calculate with more precision
sales_data = [1234.56, 2345.67, 3456.78]
total_sales = _______________
average_sale = _______________
print(f"Total: ${total_sales:.2f}")
```

### Exercise 2.3: Type Conversion
```python
# String to number
age_str = "25"
age_int = int(age_str)
print(type(age_int))

salary_str = "85000.50"
salary_float = float(salary_str)

# Number to string
count = 100
count_str = str(count)

# Your turn: Convert and calculate
price_str = "29.99"
quantity_str = "5"

# Calculate total
total = _______________
print(f"Total: ${total}")

# Handle errors
try:
    invalid = int("abc")
except ValueError as e:
    print(f"Error: {e}")
```

---

## Part 3: Strings

### Exercise 3.1: String Basics
```python
# String creation
single_quote = 'Hello'
double_quote = "World"
triple_quote = """
Multiple
Lines
"""

# String concatenation
full_name = "John" + " " + "Doe"
print(full_name)

# String repetition
separator = "=" * 50
print(separator)

# Your turn
first_name = "Alice"
last_name = "Smith"
greeting = _______________  # "Hello, Alice Smith!"
```

### Exercise 3.2: String Methods
```python
text = "  Data Science is Amazing  "

# Common methods
print(text.upper())
print(text.lower())
print(text.strip())
print(text.replace("Amazing", "Awesome"))
print(text.split())

# Your turn: Clean and format this data
messy_email = "  JoHN.DOE@EXAMPLE.COM  "
clean_email = _______________  # Should be: "john.doe@example.com"

# Check methods
email = "user@example.com"
print(email.startswith("user"))  # True
print(email.endswith(".com"))  # True
print("@" in email)  # True
```

### Exercise 3.3: String Formatting
```python
# f-strings (Python 3.6+)
name = "Alice"
age = 30
city = "New York"

# Method 1: f-strings (recommended)
message = f"Hi, I'm {name}, {age} years old from {city}"
print(message)

# Method 2: format()
message2 = "Hi, I'm {}, {} years old from {}".format(name, age, city)

# Method 3: % formatting (old style)
message3 = "Hi, I'm %s, %d years old from %s" % (name, age, city)

# Your turn: Create formatted output
product = "Laptop"
price = 999.99
quantity = 3
total = price * quantity

# Format as: "Product: Laptop | Price: $999.99 | Qty: 3 | Total: $2999.97"
formatted = _______________

# Number formatting
pi = 3.14159265359
print(f"Pi to 2 decimals: {pi:.2f}")
print(f"Pi to 4 decimals: {pi:.4f}")

large_number = 1000000
print(f"With commas: {large_number:,}")
```

### Exercise 3.4: String Slicing
```python
text = "Python Programming"

# Slicing syntax: string[start:end:step]
print(text[0])      # First character
print(text[-1])     # Last character
print(text[0:6])    # First 6 characters
print(text[7:])     # From index 7 to end
print(text[:6])     # From start to index 6
print(text[::2])    # Every 2nd character
print(text[::-1])   # Reverse

# Your turn
email = "john.doe@example.com"
username = _______________  # Extract "john.doe"
domain = _______________    # Extract "example.com"
```

---

## Part 4: Booleans

### Exercise 4.1: Boolean Operations
```python
# Boolean values
is_active = True
is_verified = False

# Comparison operators
x = 10
y = 5

print(x > y)   # Greater than
print(x < y)   # Less than
print(x == y)  # Equal to
print(x != y)  # Not equal to
print(x >= y)  # Greater than or equal
print(x <= y)  # Less than or equal

# Your turn
age = 25
# Is age between 18 and 65?
is_adult = _______________
is_senior = _______________
is_working_age = _______________
```

### Exercise 4.2: Logical Operators
```python
# and, or, not
age = 25
has_license = True
has_insurance = True

can_drive = has_license and has_insurance
print(f"Can drive: {can_drive}")

# Your turn: Eligibility check
income = 50000
credit_score = 720
has_job = True

# Eligible for loan if:
# - Income > 40000 AND credit_score > 700
# OR has_job is True AND income > 30000
eligible_for_loan = _______________

# Discount eligibility
is_student = True
is_senior = False
is_member = True

# Gets discount if student OR senior OR member
gets_discount = _______________
```

### Exercise 4.3: Truthy and Falsy
```python
# Falsy values in Python
# False, None, 0, 0.0, '', [], {}, ()

# All others are truthy
values = [0, 1, "", "hello", [], [1, 2], None, True, False]

for value in values:
    if value:
        print(f"{value} is truthy")
    else:
        print(f"{value} is falsy")

# Your turn: Validate user input
username = input("Enter username: ")
if username:
    print("Valid username")
else:
    print("Username cannot be empty")
```

---

## Part 5: Type Checking

### Exercise 5.1: Type Function
```python
# Check types
print(type(42))          # int
print(type(3.14))        # float
print(type("hello"))     # str
print(type(True))        # bool
print(type([1, 2, 3]))   # list

# isinstance()
x = 10
print(isinstance(x, int))     # True
print(isinstance(x, float))   # False
print(isinstance(x, (int, float)))  # True (checks if either)

# Your turn: Type validation
def process_data(value):
    if isinstance(value, str):
        return value.upper()
    elif isinstance(value, (int, float)):
        return value * 2
    else:
        return "Unsupported type"

# Test it
print(process_data("hello"))  # HELLO
print(process_data(5))        # 10
print(process_data(3.5))      # 7.0
```

---

## Part 6: Practice Problems

### Exercise 6.1: Temperature Converter
```python
def celsius_to_fahrenheit(celsius):
    """Convert Celsius to Fahrenheit"""
    # Your code here
    pass

def fahrenheit_to_celsius(fahrenheit):
    """Convert Fahrenheit to Celsius"""
    # Your code here
    pass

# Test
print(celsius_to_fahrenheit(0))    # Should be 32
print(celsius_to_fahrenheit(100))  # Should be 212
print(fahrenheit_to_celsius(32))   # Should be 0
```

### Exercise 6.2: Data Validator
```python
def validate_email(email):
    """
    Return True if email is valid:
    - Contains @
    - Has characters before and after @
    - Ends with .com, .org, or .edu
    """
    # Your code here
    pass

# Test
print(validate_email("user@example.com"))  # True
print(validate_email("invalid"))            # False
print(validate_email("@example.com"))       # False
```

### Exercise 6.3: Price Calculator
```python
def calculate_total(price, quantity, tax_rate=0.08):
    """
    Calculate total with tax
    
    Args:
        price: Item price
        quantity: Number of items
        tax_rate: Tax rate (default 0.08 = 8%)
    
    Returns:
        Total price with tax
    """
    # Your code here
    pass

# Test
print(calculate_total(10, 3))         # $32.40
print(calculate_total(10, 3, 0.10))   # $33.00
```

---

## Deliverables

Submit:
1. ✅ All completed exercises
2. ✅ Three practice problems solved
3. ✅ Code with comments explaining logic

---

**Next Lab:** Lab 16 - Lists and Tuples
