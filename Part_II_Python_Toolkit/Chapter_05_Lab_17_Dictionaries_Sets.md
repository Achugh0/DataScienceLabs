# Lab 17: Dictionaries and Sets - Advanced Data Structures

## Chapter 5: Python for Data Science

### Learning Objectives
- Master dictionary operations for data storage
- Use sets for unique collections
- Practice real-world data manipulation

### Duration: 60 minutes

---

## Part 1: Dictionary Basics (Hands-On)

### Exercise 1.1: Build a Customer Database
```python
# Create customer records
customer = {
    'id': 1001,
    'name': 'Alice Johnson',
    'email': 'alice@email.com',
    'purchases': 5,
    'total_spent': 450.75,
    'is_premium': True
}

# Access values
print(customer['name'])
print(customer.get('phone', 'Not provided'))  # Safe access

# Your turn: Create 3 customers
customers = [
    {'id': 1001, 'name': 'Alice', 'total_spent': 450.75},
    {'id': 1002, 'name': 'Bob', 'total_spent': 320.50},
    {'id': 1003, 'name': 'Charlie', 'total_spent': 680.25}
]

# Calculate total revenue
total_revenue = sum(c['total_spent'] for c in customers)
print(f"Total Revenue: ${total_revenue}")

# Find highest spender
top_customer = max(customers, key=lambda c: c['total_spent'])
print(f"Top Customer: {top_customer['name']}")
```

### Exercise 1.2: Product Inventory System
```python
# Build inventory
inventory = {
    'LAP001': {'name': 'Laptop', 'price': 999.99, 'stock': 15},
    'MOU001': {'name': 'Mouse', 'price': 29.99, 'stock': 50},
    'KEY001': {'name': 'Keyboard', 'price': 79.99, 'stock': 30}
}

# Add new product
inventory['MON001'] = {'name': 'Monitor', 'price': 299.99, 'stock': 20}

# Update stock
inventory['LAP001']['stock'] -= 1  # Sold one laptop

# Check if product exists
if 'LAP001' in inventory:
    print(f"In stock: {inventory['LAP001']['stock']}")

# YOUR CODE: Process sale
def process_sale(inventory, product_id, quantity):
    """Process a sale and update inventory"""
    # Check if product exists
    # Check if enough stock
    # Update stock
    # Return total price
    pass

# Test it
total = process_sale(inventory, 'MOU001', 3)
```

---

## Part 2: Dictionary Methods (Practical)

### Exercise 2.1: User Analytics
```python
# Page views by user
page_views = {
    'user1': ['home', 'products', 'cart'],
    'user2': ['home', 'about'],
    'user3': ['home', 'products', 'checkout', 'confirmation']
}

# Get all keys (users)
users = list(page_views.keys())

# Get all values (paths)
all_paths = list(page_views.values())

# Get key-value pairs
for user, pages in page_views.items():
    print(f"{user} viewed {len(pages)} pages")

# YOUR CODE: Analytics
# 1. Which user viewed most pages?
# 2. What's the average pages per user?
# 3. How many users visited 'cart'?
# 4. Create a dict of page -> count
page_counts = {}
for paths in page_views.values():
    for page in paths:
        page_counts[page] = page_counts.get(page, 0) + 1
print(page_counts)
```

### Exercise 2.2: Merge Customer Data
```python
# Data from different sources
basic_info = {'id': 1001, 'name': 'Alice', 'email': 'alice@email.com'}
purchase_info = {'id': 1001, 'orders': 5, 'total': 450.75}
preferences = {'id': 1001, 'newsletter': True, 'theme': 'dark'}

# Merge dictionaries (Python 3.9+)
complete_profile = basic_info | purchase_info | preferences

# Or using update
customer_profile = {}
customer_profile.update(basic_info)
customer_profile.update(purchase_info)
customer_profile.update(preferences)

# YOUR CODE: Merge and validate
def merge_customer_data(sources):
    """Merge multiple data sources and validate"""
    merged = {}
    for source in sources:
        merged.update(source)
    
    # Validate required fields
    required = ['id', 'name', 'email']
    for field in required:
        if field not in merged:
            raise ValueError(f"Missing {field}")
    
    return merged
```

---

## Part 3: Real Dataset Practice

### Exercise 3.1: Sales Data Analysis
```python
# Real sales transactions
sales_data = [
    {'date': '2024-01-01', 'product': 'Laptop', 'quantity': 2, 'price': 999.99},
    {'date': '2024-01-01', 'product': 'Mouse', 'quantity': 5, 'price': 29.99},
    {'date': '2024-01-02', 'product': 'Laptop', 'quantity': 1, 'price': 999.99},
    {'date': '2024-01-02', 'product': 'Keyboard', 'quantity': 3, 'price': 79.99},
    {'date': '2024-01-03', 'product': 'Mouse', 'quantity': 8, 'price': 29.99},
]

# YOUR CODE: Analyze sales
# 1. Total revenue by product
revenue_by_product = {}
for sale in sales_data:
    product = sale['product']
    revenue = sale['quantity'] * sale['price']
    revenue_by_product[product] = revenue_by_product.get(product, 0) + revenue

print("Revenue by Product:")
for product, revenue in sorted(revenue_by_product.items(), key=lambda x: x[1], reverse=True):
    print(f"  {product}: ${revenue:,.2f}")

# 2. Total quantity sold by product
# 3. Revenue by date
# 4. Best-selling product by quantity
# 5. Best-selling product by revenue
```

### Exercise 3.2: Build a Simple Database
```python
# Student grades database
students_db = {}

def add_student(student_id, name, grades):
    """Add student to database"""
    students_db[student_id] = {
        'name': name,
        'grades': grades,
        'average': sum(grades) / len(grades)
    }

def get_student(student_id):
    """Retrieve student record"""
    return students_db.get(student_id, {})

def update_grade(student_id, new_grade):
    """Add a grade to student"""
    if student_id in students_db:
        students_db[student_id]['grades'].append(new_grade)
        grades = students_db[student_id]['grades']
        students_db[student_id]['average'] = sum(grades) / len(grades)

def get_top_students(n=5):
    """Get top N students by average"""
    sorted_students = sorted(
        students_db.items(),
        key=lambda x: x[1]['average'],
        reverse=True
    )
    return sorted_students[:n]

# YOUR CODE: Use the database
# 1. Add 5 students
# 2. Update grades for 2 students
# 3. Find top 3 students
# 4. Calculate class average
# 5. Find students with average > 85
```

---

## Part 4: Sets (Practical)

### Exercise 4.1: Unique Visitors Analysis
```python
# Daily website visitors
monday = {'user1', 'user2', 'user3', 'user4', 'user5'}
tuesday = {'user2', 'user4', 'user6', 'user7'}
wednesday = {'user1', 'user3', 'user6', 'user8', 'user9'}

# Union: all unique visitors
all_visitors = monday | tuesday | wednesday
print(f"Total unique visitors: {len(all_visitors)}")

# Intersection: returned every day
daily_users = monday & tuesday & wednesday
print(f"Daily users: {daily_users}")

# Difference: Monday only
monday_only = monday - tuesday - wednesday
print(f"Monday only: {monday_only}")

# YOUR CODE: More analytics
# 1. Users who visited both Monday and Tuesday
# 2. Users who visited Monday or Tuesday but not Wednesday
# 3. Users who visited exactly 2 days
# 4. Calculate retention rate (return visitors / total visitors)
```

### Exercise 4.2: Email List Management
```python
# Email lists from different sources
newsletter_subscribers = {'alice@email.com', 'bob@email.com', 'charlie@email.com'}
purchasers = {'bob@email.com', 'david@email.com', 'eve@email.com'}
event_attendees = {'alice@email.com', 'eve@email.com', 'frank@email.com'}

# YOUR CODE: Segment the audience
# 1. All contacts (union)
all_contacts = newsletter_subscribers | purchasers | event_attendees

# 2. Customers who haven't subscribed to newsletter
# 3. Newsletter subscribers who haven't purchased
# 4. People engaged in all three channels
# 5. Create targeted groups for campaigns

def create_segments(subscribers, purchasers, attendees):
    """Create marketing segments"""
    segments = {
        'high_engagement': subscribers & purchasers & attendees,
        'need_conversion': subscribers - purchasers,
        'need_reengagement': purchasers - subscribers,
        'cold_leads': (subscribers | purchasers | attendees) - (subscribers & purchasers)
    }
    return segments

segments = create_segments(newsletter_subscribers, purchasers, event_attendees)
for segment, emails in segments.items():
    print(f"{segment}: {len(emails)} contacts")
```

---

## Part 5: Combined Practical Project

### Exercise 5.1: E-commerce Order Processor
```python
# Order processing system
orders = []
inventory = {}
customers = {}

def process_order(customer_id, items):
    """
    Process an order
    items: list of (product_id, quantity) tuples
    """
    # Check inventory
    for product_id, quantity in items:
        if product_id not in inventory:
            return {"status": "error", "message": f"Product {product_id} not found"}
        if inventory[product_id]['stock'] < quantity:
            return {"status": "error", "message": f"Insufficient stock for {product_id}"}
    
    # Calculate total
    total = 0
    for product_id, quantity in items:
        total += inventory[product_id]['price'] * quantity
    
    # Update inventory
    for product_id, quantity in items:
        inventory[product_id]['stock'] -= quantity
    
    # Record order
    order = {
        'order_id': len(orders) + 1,
        'customer_id': customer_id,
        'items': items,
        'total': total,
        'status': 'completed'
    }
    orders.append(order)
    
    # Update customer stats
    if customer_id not in customers:
        customers[customer_id] = {'orders': 0, 'total_spent': 0}
    customers[customer_id]['orders'] += 1
    customers[customer_id]['total_spent'] += total
    
    return {"status": "success", "order": order}

# YOUR CODE: Build the system
# 1. Add 10 products to inventory
# 2. Process 5 orders
# 3. Generate sales report
# 4. Find low-stock items (< 5 units)
# 5. Identify VIP customers (>$1000 spent)
```

### Exercise 5.2: Movie Recommendation Data
```python
# User movie ratings
user_ratings = {
    'Alice': {'Inception': 5, 'Matrix': 4, 'Interstellar': 5},
    'Bob': {'Inception': 4, 'Godfather': 5, 'Matrix': 3},
    'Charlie': {'Matrix': 5, 'Godfather': 4, 'Shawshank': 5},
}

# YOUR CODE: Recommendation logic
def get_common_movies(user1, user2):
    """Find movies both users have rated"""
    return set(user_ratings[user1].keys()) & set(user_ratings[user2].keys())

def find_similar_users(target_user):
    """Find users with similar taste"""
    target_movies = set(user_ratings[target_user].keys())
    similar_users = []
    
    for user in user_ratings:
        if user != target_user:
            common = len(target_movies & set(user_ratings[user].keys()))
            if common >= 2:  # At least 2 movies in common
                similar_users.append((user, common))
    
    return sorted(similar_users, key=lambda x: x[1], reverse=True)

def recommend_movies(target_user):
    """Recommend unwatched movies from similar users"""
    # Your implementation
    pass

# Test recommendations
print(f"Recommendations for Alice: {recommend_movies('Alice')}")
```

---

## Part 6: Performance and Best Practices

### Exercise 6.1: Dictionary vs List Lookup
```python
import time

# Create test data
data_list = list(range(100000))
data_dict = {i: i for i in range(100000)}
data_set = set(range(100000))

# Test lookup performance
def time_lookup(collection, value):
    start = time.time()
    for _ in range(1000):
        _ = value in collection
    return time.time() - start

# Compare
print("List lookup:", time_lookup(data_list, 99999))
print("Dict lookup:", time_lookup(data_dict, 99999))
print("Set lookup:", time_lookup(data_set, 99999))

# YOUR CODE: When to use each?
# Explain the O(n) vs O(1) difference
```

### Exercise 6.2: Dictionary Comprehensions
```python
# Create dictionaries efficiently
squares = {x: x**2 for x in range(10)}

# From lists
names = ['Alice', 'Bob', 'Charlie']
ids = [1, 2, 3]
user_map = {id: name for id, name in zip(ids, names)}

# With conditions
evens_squared = {x: x**2 for x in range(20) if x % 2 == 0}

# YOUR CODE: Real applications
# 1. Convert list of tuples to dict
transactions = [('2024-01-01', 100), ('2024-01-02', 150), ('2024-01-03', 200)]
sales_by_date = {date: amount for date, amount in transactions}

# 2. Count word frequency
text = "the quick brown fox jumps over the lazy dog the fox"
word_freq = {}  # Your code here

# 3. Invert a dictionary (swap keys and values)
original = {'a': 1, 'b': 2, 'c': 3}
inverted = {v: k for k, v in original.items()}
```

---

## Deliverables

Complete these practical projects:

1. ✅ Customer database with CRUD operations
2. ✅ Sales analytics report
3. ✅ Email list segmentation
4. ✅ Order processing system
5. ✅ Movie recommendation engine

---

## Challenge Project

Build a complete contact management system:
- Add/edit/delete contacts
- Search by name, email, phone
- Tag contacts with categories
- Export to dict/JSON format
- Import from dict/JSON format

```python
class ContactManager:
    def __init__(self):
        self.contacts = {}
        self.tags = {}
    
    def add_contact(self, name, email, phone, tags=[]):
        # Your implementation
        pass
    
    def search(self, query):
        # Your implementation
        pass
    
    def get_by_tag(self, tag):
        # Your implementation
        pass

# Use it
manager = ContactManager()
# Test all functions
```

**Next Lab:** Lab 18 - Control Flow and Logic
