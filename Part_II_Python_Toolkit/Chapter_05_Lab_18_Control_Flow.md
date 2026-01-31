# Lab 18: Control Flow - Making Decisions in Code

## Chapter 5: Python for Data Science

### Learning Objectives
- Master if/elif/else logic for data validation
- Use conditional expressions in data processing
- Build decision trees for real problems

### Duration: 60 minutes

---

## Part 1: Data Validation (Hands-On)

### Exercise 1.1: Validate User Input
```python
# Real-world validation
def validate_age(age):
    if age < 0:
        return "Error: Age cannot be negative"
    elif age < 18:
        return "Minor"
    elif age < 65:
        return "Adult"
    else:
        return "Senior"

# YOUR CODE: Validate email
def validate_email(email):
    """Check if email is valid"""
    if '@' not in email:
        return False, "Missing @"
    if '.' not in email.split('@')[1]:
        return False, "Missing domain extension"
    if len(email) < 5:
        return False, "Email too short"
    return True, "Valid"

# Test with real data
emails = ["user@example.com", "invalid", "no@domain", "a@b.c"]
for email in emails:
    valid, message = validate_email(email)
    print(f"{email}: {message}")
```

### Exercise 1.2: Process Transactions
```python
# Transaction processing logic
def process_transaction(amount, account_balance, daily_limit=1000):
    """
    Process ATM transaction
    Returns: (success, new_balance, message)
    """
    if amount <= 0:
        return False, account_balance, "Invalid amount"
    
    if amount > daily_limit:
        return False, account_balance, "Exceeds daily limit"
    
    if amount > account_balance:
        return False, account_balance, "Insufficient funds"
    
    new_balance = account_balance - amount
    return True, new_balance, "Transaction successful"

# YOUR CODE: Test scenarios
test_cases = [
    (500, 1000),   # Normal withdrawal
    (-100, 1000),  # Negative amount
    (1500, 2000),  # Exceeds limit
    (1500, 1200),  # Insufficient funds
]

for amount, balance in test_cases:
    success, new_bal, msg = process_transaction(amount, balance)
    print(f"Withdraw ${amount} from ${balance}: {msg}")
```

---

## Part 2: Grade Calculator (Practical)

### Exercise 2.1: Student Grading System
```python
def calculate_grade(scores):
    """Calculate letter grade from scores"""
    if not scores:
        return "No scores"
    
    average = sum(scores) / len(scores)
    
    if average >= 90:
        letter = 'A'
        gpa = 4.0
    elif average >= 80:
        letter = 'B'
        gpa = 3.0
    elif average >= 70:
        letter = 'C'
        gpa = 2.0
    elif average >= 60:
        letter = 'D'
        gpa = 1.0
    else:
        letter = 'F'
        gpa = 0.0
    
    return {
        'average': average,
        'letter': letter,
        'gpa': gpa,
        'passed': average >= 60
    }

# YOUR CODE: Process class data
students = [
    {'name': 'Alice', 'scores': [85, 90, 88, 92]},
    {'name': 'Bob', 'scores': [70, 65, 72, 68]},
    {'name': 'Charlie', 'scores': [95, 98, 96, 94]},
    {'name': 'David', 'scores': [55, 60, 58, 52]},
]

# Calculate grades for all students
# Find class average
# Count how many passed
# Find students on academic probation (< 2.0 GPA)
```

### Exercise 2.2: Weighted Grading
```python
def calculate_weighted_grade(homework, midterm, final, participation):
    """
    Calculate weighted grade:
    - Homework: 30%
    - Midterm: 25%
    - Final: 35%
    - Participation: 10%
    """
    # YOUR CODE: Calculate and classify
    weighted_avg = (homework * 0.30 + 
                    midterm * 0.25 + 
                    final * 0.35 + 
                    participation * 0.10)
    
    # Determine if honors (>= 93), pass (>= 60), or fail
    if weighted_avg >= 93:
        status = "Honors"
    elif weighted_avg >= 60:
        status = "Pass"
    else:
        status = "Fail"
    
    return weighted_avg, status

# Test with real student data
test_student = {
    'homework': 85,
    'midterm': 78,
    'final': 88,
    'participation': 95
}

grade, status = calculate_weighted_grade(**test_student)
print(f"Final Grade: {grade:.2f} - {status}")
```

---

## Part 3: Data Classification (Real Datasets)

### Exercise 3.1: Customer Segmentation
```python
def segment_customer(age, income, purchases, member_years):
    """Classify customer into marketing segment"""
    
    # High-value: high income + many purchases
    if income > 100000 and purchases > 20:
        return "VIP"
    
    # Potential high-value: high income, few purchases
    elif income > 100000 and purchases < 5:
        return "High Potential"
    
    # Loyal: long membership + regular purchases
    elif member_years > 5 and purchases > 10:
        return "Loyal"
    
    # Young professional: under 35, good income
    elif age < 35 and income > 60000:
        return "Young Professional"
    
    # At risk: was active, now inactive
    elif member_years > 2 and purchases < 2:
        return "At Risk"
    
    else:
        return "Regular"

# YOUR CODE: Segment customer database
customers = [
    {'name': 'Alice', 'age': 32, 'income': 120000, 'purchases': 25, 'years': 3},
    {'name': 'Bob', 'age': 45, 'income': 80000, 'purchases': 15, 'years': 7},
    {'name': 'Charlie', 'age': 28, 'income': 150000, 'purchases': 3, 'years': 1},
    {'name': 'David', 'age': 55, 'income': 95000, 'purchases': 1, 'years': 5},
]

# Segment all customers
# Count customers per segment
# Calculate average income per segment
```

### Exercise 3.2: Loan Approval System
```python
def evaluate_loan_application(credit_score, income, debt, employment_years):
    """
    Determine loan approval and terms
    Returns: (approved, rate, message)
    """
    # Automatic rejection criteria
    if credit_score < 600:
        return False, None, "Credit score too low"
    
    if income < 30000:
        return False, None, "Income requirement not met"
    
    # Calculate debt-to-income ratio
    debt_to_income = debt / income
    
    if debt_to_income > 0.43:
        return False, None, "Debt-to-income ratio too high"
    
    # Excellent credit
    if credit_score >= 750 and employment_years >= 2:
        return True, 3.5, "Excellent terms"
    
    # Good credit
    elif credit_score >= 700:
        return True, 4.5, "Good terms"
    
    # Fair credit
    elif credit_score >= 650 and employment_years >= 1:
        return True, 6.0, "Standard terms"
    
    # Conditional approval
    else:
        return True, 7.5, "Higher rate due to risk"

# YOUR CODE: Process applications
applications = [
    {'credit': 780, 'income': 85000, 'debt': 15000, 'years': 5},
    {'credit': 650, 'income': 50000, 'debt': 30000, 'years': 1},
    {'credit': 580, 'income': 60000, 'debt': 10000, 'years': 3},
]

for i, app in enumerate(applications, 1):
    approved, rate, msg = evaluate_loan_application(**app)
    print(f"Application {i}: {msg}")
    if approved:
        print(f"  Rate: {rate}%")
```

---

## Part 4: Nested Conditions (Complex Logic)

### Exercise 4.1: Shipping Cost Calculator
```python
def calculate_shipping(weight, distance, speed, insurance=False):
    """
    Calculate shipping cost based on multiple factors
    """
    # Base rate by weight
    if weight <= 1:
        base_cost = 5.00
    elif weight <= 5:
        base_cost = 10.00
    elif weight <= 20:
        base_cost = 25.00
    else:
        base_cost = 25.00 + (weight - 20) * 2.00
    
    # Distance multiplier
    if distance < 100:
        distance_mult = 1.0
    elif distance < 500:
        distance_mult = 1.5
    else:
        distance_mult = 2.0
    
    # Speed premium
    if speed == 'overnight':
        speed_mult = 2.5
    elif speed == 'express':
        speed_mult = 1.5
    else:  # standard
        speed_mult = 1.0
    
    # Calculate total
    total = base_cost * distance_mult * speed_mult
    
    # Add insurance
    if insurance:
        total += total * 0.1  # 10% for insurance
    
    return round(total, 2)

# YOUR CODE: Test shipping scenarios
# 1. 2lb package, 150 miles, standard
# 2. 10lb package, 600 miles, express, insured
# 3. 25lb package, 50 miles, overnight
# 4. Create a shipping calculator interface
```

### Exercise 4.2: Dynamic Pricing Engine
```python
def calculate_price(base_price, customer_type, quantity, season):
    """
    Calculate final price with discounts
    """
    price = base_price
    
    # Customer type discount
    if customer_type == 'VIP':
        if quantity >= 10:
            discount = 0.25  # 25% off
        else:
            discount = 0.15  # 15% off
    elif customer_type == 'member':
        discount = 0.10  # 10% off
    else:
        discount = 0.0
    
    # Apply customer discount
    price = price * (1 - discount)
    
    # Volume discount (additional)
    if quantity >= 50:
        price = price * 0.85  # Additional 15% off
    elif quantity >= 20:
        price = price * 0.90  # Additional 10% off
    elif quantity >= 10:
        price = price * 0.95  # Additional 5% off
    
    # Seasonal pricing
    if season == 'holiday':
        price = price * 1.20  # 20% markup
    elif season == 'clearance':
        price = price * 0.70  # 30% off
    
    total = price * quantity
    
    return {
        'unit_price': round(price, 2),
        'quantity': quantity,
        'total': round(total, 2),
        'savings': round((base_price * quantity) - total, 2)
    }

# YOUR CODE: Price scenarios
# Test different combinations and find best deal
```

---

## Part 5: Ternary Operators (Compact Logic)

### Exercise 5.1: Data Cleaning with Ternary
```python
# Clean data efficiently
scores = [85, -1, 92, None, 78, 105, 88]

# Traditional
clean_scores = []
for score in scores:
    if score is None or score < 0 or score > 100:
        clean_scores.append(0)
    else:
        clean_scores.append(score)

# Ternary operator
clean_scores = [score if (score and 0 <= score <= 100) else 0 
                for score in scores]

# YOUR CODE: Clean customer data
customers = [
    {'name': 'Alice', 'age': 25, 'email': 'alice@email.com'},
    {'name': '', 'age': -5, 'email': 'invalid'},
    {'name': 'Bob', 'age': 150, 'email': 'bob@email.com'},
]

# Set default name to "Unknown" if empty
# Set age to None if invalid (< 0 or > 120)
# Set email to None if no @ sign
```

### Exercise 5.2: Status Indicators
```python
# Create status messages
orders = [
    {'id': 1, 'status': 'shipped', 'days_since': 2},
    {'id': 2, 'status': 'processing', 'days_since': 5},
    {'id': 3, 'status': 'delivered', 'days_since': 7},
]

# Add status messages
for order in orders:
    order['message'] = (
        "Delivered!" if order['status'] == 'delivered' else
        "In transit" if order['status'] == 'shipped' else
        "Delayed!" if order['days_since'] > 3 else
        "Processing"
    )
    
    order['priority'] = 'high' if order['days_since'] > 5 else 'normal'

# YOUR CODE: Customer service priorities
# Assign priority based on:
# - Order age
# - Status
# - Customer type
```

---

## Part 6: Real-World Project

### Exercise 6.1: Build a Product Recommendation Engine
```python
def recommend_products(customer, products, purchase_history):
    """
    Recommend products based on customer profile and history
    """
    recommendations = []
    
    for product in products:
        score = 0
        
        # Check if already purchased
        if product['id'] in purchase_history:
            continue
        
        # Price match
        if customer['budget'] >= product['price']:
            score += 10
        
        # Category preference
        if product['category'] in customer['interests']:
            score += 20
        
        # Rating threshold
        if product['rating'] >= 4.0:
            score += 15
        
        # Popularity
        if product['sales'] > 1000:
            score += 10
        
        # Age appropriateness
        if product['min_age'] <= customer['age']:
            score += 5
        
        # Add to recommendations if score is high enough
        if score >= 30:
            recommendations.append({
                'product': product['name'],
                'score': score,
                'price': product['price']
            })
    
    # Sort by score
    recommendations.sort(key=lambda x: x['score'], reverse=True)
    return recommendations[:5]  # Top 5

# YOUR CODE: Test recommendation engine
customer_profile = {
    'age': 28,
    'budget': 100,
    'interests': ['electronics', 'books']
}

products_catalog = [
    {'id': 1, 'name': 'Laptop', 'category': 'electronics', 'price': 999, 
     'rating': 4.5, 'sales': 5000, 'min_age': 12},
    {'id': 2, 'name': 'Python Book', 'category': 'books', 'price': 45, 
     'rating': 4.8, 'sales': 2000, 'min_age': 16},
    # Add more products
]

purchased = [3, 5, 7]  # Already bought product IDs

recommendations = recommend_products(customer_profile, products_catalog, purchased)
for rec in recommendations:
    print(f"{rec['product']}: Score {rec['score']}")
```

---

## Deliverables

Build these practical systems:

1. ✅ Transaction processor with full validation
2. ✅ Customer segmentation engine
3. ✅ Loan approval system
4. ✅ Dynamic pricing calculator
5. ✅ Product recommendation engine

---

## Challenge

Build a complete e-commerce checkout system that handles:
- Cart validation
- Discount application
- Shipping calculation
- Tax computation
- Payment processing
- Order confirmation

```python
def process_checkout(cart, customer, payment):
    # Your complete implementation
    pass
```

**Next Lab:** Lab 19 - Loops and Iterations
