# Lab 16: Lists and Tuples - Working with Sequences

## Chapter 5: Python for Data Science — Your First Steps in Code

### Learning Objectives
- Master list operations and methods
- Understand tuple immutability
- Practice list comprehensions
- Work with nested sequences

### Duration
75 minutes

---

## Part 1: List Basics

### Exercise 1.1: Creating and Accessing Lists
```python
# Create lists
numbers = [1, 2, 3, 4, 5]
names = ["Alice", "Bob", "Charlie"]
mixed = [1, "two", 3.0, True]
empty = []

# Access elements
print(numbers[0])   # First element
print(numbers[-1])  # Last element
print(names[1])     # Second element

# Your turn: Create and access
favorite_movies = _______________  # List of 5 movies
print(f"First movie: {favorite_movies[0]}")
print(f"Last movie: {favorite_movies[-1]}")
print(f"Middle movie: {favorite_movies[___]}")
```

### Exercise 1.2: Slicing Lists
```python
numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# Slicing [start:end:step]
print(numbers[2:7])    # Elements 2-6
print(numbers[:5])     # First 5
print(numbers[5:])     # From 5 onwards
print(numbers[::2])    # Every second element
print(numbers[::-1])   # Reverse

# Your turn
data = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
first_half = _______________
last_three = _______________
every_third = _______________
reversed_data = _______________
```

### Exercise 1.3: Modifying Lists
```python
# Lists are mutable
fruits = ["apple", "banana", "cherry"]

# Change element
fruits[1] = "blueberry"
print(fruits)

# Add elements
fruits.append("date")        # Add to end
fruits.insert(1, "avocado")  # Insert at position
fruits.extend(["elderberry", "fig"])  # Add multiple

# Remove elements
fruits.remove("cherry")      # Remove by value
last_fruit = fruits.pop()    # Remove and return last
second_fruit = fruits.pop(1) # Remove at index
del fruits[0]                # Delete by index

# Your turn: Shopping cart
cart = ["milk", "eggs", "bread"]
# Add "butter" to the cart
# Remove "eggs"
# Add ["cheese", "yogurt"]
# Print final cart
```

---

## Part 2: List Methods

### Exercise 2.1: Common List Methods
```python
numbers = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]

# Sort
numbers.sort()  # In-place
print(numbers)

sorted_nums = sorted(numbers)  # Returns new list

# Reverse
numbers.reverse()
print(numbers)

# Count and index
count_of_5 = numbers.count(5)
index_of_9 = numbers.index(9)

# Clear
numbers.clear()  # Empty the list

# Your turn: Analyze scores
scores = [85, 92, 78, 92, 88, 75, 92, 95]
# How many students scored 92?
# What's the position of first 95?
# Sort scores from highest to lowest
# Calculate median (middle value)
```

### Exercise 2.2: List Operations
```python
# Concatenation
list1 = [1, 2, 3]
list2 = [4, 5, 6]
combined = list1 + list2

# Repetition
repeated = [0] * 5  # [0, 0, 0, 0, 0]

# Membership
print(3 in list1)        # True
print(10 not in list1)   # True

# Length
print(len(combined))

# Min, max, sum (for numeric lists)
print(min(list1))
print(max(list1))
print(sum(list1))

# Your turn: Combine data
sales_q1 = [1000, 1200, 1100]
sales_q2 = [1300, 1250, 1400]
# Combine quarters
# Calculate total annual sales
# Find best month
# Find worst month
```

---

## Part 3: List Comprehensions

### Exercise 3.1: Basic Comprehensions
```python
# Traditional way
squares = []
for x in range(10):
    squares.append(x**2)

# List comprehension way
squares = [x**2 for x in range(10)]

# With condition
evens = [x for x in range(20) if x % 2 == 0]

# Transformation
names = ["alice", "bob", "charlie"]
upper_names = [name.upper() for name in names]

# Your turn
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# Create list of cubes
# Create list of even numbers
# Create list of numbers divisible by 3
# Double each number
```

### Exercise 3.2: Advanced Comprehensions
```python
# Nested comprehension
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
flattened = [num for row in matrix for num in row]

# With if-else
result = [x if x > 0 else 0 for x in [-2, -1, 0, 1, 2]]

# Multiple conditions
filtered = [x for x in range(100) if x % 2 == 0 if x % 5 == 0]

# Your turn: Data cleaning
messy_data = ["  apple ", "BANANA", "  Cherry  ", "date"]
# Clean: strip whitespace and lowercase all
cleaned = _______________

# Your turn: Extract info
emails = ["user1@gmail.com", "user2@yahoo.com", "user3@gmail.com"]
# Extract just the usernames (before @)
usernames = _______________
```

---

## Part 4: Tuples - Immutable Sequences

### Exercise 4.1: Tuple Basics
```python
# Create tuples
point = (3, 4)
person = ("Alice", 25, "NYC")
single = (42,)  # Note the comma!

# Access like lists
print(point[0])
print(person[1])

# But tuples are immutable
# person[1] = 26  # This will error!

# Your turn
coordinates = (40.7128, -74.0060)  # NYC lat, long
# Extract latitude and longitude
lat = _______________
long = _______________

# Tuples can be unpacked
x, y = point
name, age, city = person

# Your turn: Swap without temp variable
a = 5
b = 10
# Swap using tuple unpacking
```

### Exercise 4.2: When to Use Tuples vs Lists

```python
# Use tuples for:
# 1. Fixed data that shouldn't change
RGB_RED = (255, 0, 0)
SCREEN_SIZE = (1920, 1080)

# 2. Dictionary keys (lists can't be keys)
locations = {
    (40.7128, -74.0060): "NYC",
    (51.5074, -0.1278): "London"
}

# 3. Multiple return values
def get_stats(numbers):
    return min(numbers), max(numbers), sum(numbers)/len(numbers)

minimum, maximum, average = get_stats([1, 2, 3, 4, 5])

# Your turn: Function that returns multiple values
def analyze_text(text):
    """Return word_count, char_count, avg_word_length"""
    # Your code here
    pass

words, chars, avg = analyze_text("The quick brown fox")
```

---

## Part 5: Real-World Applications

### Exercise 5.1: Data Processing Pipeline
```python
# Sales data processing
raw_sales = [
    ("Product A", 100),
    ("Product B", 150),
    ("Product C", 75),
    ("Product A", 120),
    ("Product B", 200)
]

# Extract all products
products = [item[0] for item in raw_sales]

# Extract all amounts
amounts = [item[1] for item in raw_sales]

# Filter high-value sales (>100)
high_value = [item for item in raw_sales if item[1] > 100]

# Calculate total revenue
total = sum(amounts)

# Your turn: More analysis
# 1. Find unique products
# 2. Calculate average sale amount
# 3. Find best-selling product
# 4. Count sales per product
```

### Exercise 5.2: Student Grade Management
```python
# Student records: (name, grades_list)
students = [
    ("Alice", [85, 90, 78, 92]),
    ("Bob", [70, 75, 68, 72]),
    ("Charlie", [95, 98, 94, 97]),
    ("David", [80, 82, 85, 83])
]

# Calculate average for each student
averages = [(name, sum(grades)/len(grades)) 
            for name, grades in students]

# Find students with average > 85
high_performers = [name for name, grades in students 
                   if sum(grades)/len(grades) > 85]

# Your turn:
# 1. Find highest single grade across all students
# 2. Find student with lowest average
# 3. Calculate class average
# 4. Count students passing (avg >= 70)
```

### Exercise 5.3: Time Series Data
```python
# Daily temperatures for a week
temperatures = [72, 75, 78, 74, 70, 73, 76]
days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# Combine using zip
week_data = list(zip(days, temperatures))
print(week_data)

# Find warmest day
warmest_temp = max(temperatures)
warmest_day = days[temperatures.index(warmest_temp)]

# Your turn:
# 1. Calculate average temperature
# 2. Find coldest day
# 3. Count days above 75°
# 4. Calculate temperature trend (increasing/decreasing)
```

---

## Part 6: Common Patterns

### Exercise 6.1: Filtering and Mapping
```python
# Map: Transform each element
def double(x):
    return x * 2

numbers = [1, 2, 3, 4, 5]
doubled = list(map(double, numbers))
# Or with lambda: list(map(lambda x: x*2, numbers))
# Or with comprehension: [x*2 for x in numbers]

# Filter: Keep elements meeting condition
def is_even(x):
    return x % 2 == 0

evens = list(filter(is_even, numbers))
# Or with comprehension: [x for x in numbers if x % 2 == 0]

# Your turn: Data transformation
prices = [19.99, 29.99, 39.99, 49.99]
# Add 8% tax to each price
# Filter prices under $35
# Round to 2 decimal places
```

### Exercise 6.2: Accumulation Pattern
```python
# Running total
sales = [100, 150, 200, 175, 225]
running_total = []
total = 0

for sale in sales:
    total += sale
    running_total.append(total)

print(running_total)  # [100, 250, 450, 625, 850]

# Your turn: Moving average
def moving_average(data, window=3):
    """Calculate moving average with given window size"""
    # Your code here
    pass

stock_prices = [100, 102, 101, 105, 108, 107, 110]
avg_prices = moving_average(stock_prices, 3)
```

---

## Part 7: Debugging Common Errors

### Exercise 7.1: Index Errors
```python
# Common mistake
my_list = [1, 2, 3]
# print(my_list[3])  # IndexError!

# Fix with len()
if len(my_list) > 3:
    print(my_list[3])

# Or with try-except
try:
    print(my_list[3])
except IndexError:
    print("Index out of range")

# Your turn: Safe list access
def safe_get(lst, index, default=None):
    """Safely get element or return default"""
    # Your code here
    pass
```

### Exercise 7.2: Modifying While Iterating
```python
# WRONG way
numbers = [1, 2, 3, 4, 5]
for num in numbers:
    if num % 2 == 0:
        numbers.remove(num)  # Don't do this!

# RIGHT way 1: Create new list
numbers = [1, 2, 3, 4, 5]
odd_numbers = [num for num in numbers if num % 2 != 0]

# RIGHT way 2: Iterate over copy
numbers = [1, 2, 3, 4, 5]
for num in numbers[:]:  # Note the [:]
    if num % 2 == 0:
        numbers.remove(num)

# Your turn: Remove negative numbers safely
values = [-5, 10, -3, 8, -1, 6]
# Remove all negative values
```

---

## Deliverables

Submit:
1. ✅ All completed exercises with output
2. ✅ Sales data analysis (Exercise 5.1)
3. ✅ Student grade management (Exercise 5.2)
4. ✅ Time series analysis (Exercise 5.3)
5. ✅ Moving average function (Exercise 6.2)

---

## Practice Challenge

Create a simple todo list manager:

```python
def todo_manager():
    todos = []
    
    while True:
        print("\n1. Add task")
        print("2. View tasks")
        print("3. Complete task")
        print("4. Exit")
        
        choice = input("Choose option: ")
        
        # Your implementation here
        pass

# Run it
todo_manager()
```

---

**Next Lab:** Lab 17 - Dictionaries and Sets
