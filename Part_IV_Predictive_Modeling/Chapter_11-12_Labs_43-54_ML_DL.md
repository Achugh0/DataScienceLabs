# Lab 43-54: Machine Learning & Deep Learning Complete Course

## Chapter 11-12: Predictive Modeling & Neural Networks

### 12 Comprehensive ML Labs

---

# Lab 43: ML Fundamentals - Scikit-learn Basics

## Duration: 75 minutes

### Exercise 1: Train-Test Split & Linear Regression
```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Generate housing data
np.random.seed(42)
X = pd.DataFrame({
    'sqft': np.random.randint(800, 3000, 500),
    'bedrooms': np.random.randint(1, 5, 500),
    'age': np.random.randint(0, 50, 500)
})
# Price = 100 * sqft + 20000 * bedrooms - 500 * age + noise
y = (100 * X['sqft'] + 20000 * X['bedrooms'] - 500 * X['age'] + 
     np.random.normal(0, 20000, 500))

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: ${mse:,.0f}")
print(f"RMSE: ${np.sqrt(mse):,.0f}")
print(f"R²: {r2:.3f}")
print(f"\nCoefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"  {feature}: ${coef:.2f}")
print(f"Intercept: ${model.intercept_:.2f}")

# Visualize
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
         'r--', lw=2)
plt.xlabel('Actual Price ($)')
plt.ylabel('Predicted Price ($)')
plt.title(f'Predictions vs Actuals (R² = {r2:.3f})')
plt.show()
```

---

# Lab 44: Classification - Logistic Regression

## Duration: 60 minutes

### Exercise 1: Customer Churn Prediction
```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Generate customer data
np.random.seed(42)
n = 1000
customers = pd.DataFrame({
    'tenure': np.random.randint(1, 72, n),
    'monthly_charges': np.random.uniform(20, 150, n),
    'total_charges': np.random.uniform(100, 8000, n),
    'support_calls': np.random.poisson(2, n)
})

# Generate churn (higher support calls, shorter tenure = higher churn)
churn_prob = 1 / (1 + np.exp(-(
    -0.05 * customers['tenure'] +
    0.02 * customers['support_calls'] +
    2
)))
customers['churn'] = (np.random.random(n) < churn_prob).astype(int)

# Split
X = customers.drop('churn', axis=1)
y = customers['churn']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

# Predict
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

# Evaluate
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"Precision: {precision_score(y_test, y_pred):.3f}")
print(f"Recall: {recall_score(y_test, y_pred):.3f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.3f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# Feature importance
importance = pd.DataFrame({
    'feature': X.columns,
    'coefficient': model.coef_[0]
}).sort_values('coefficient', ascending=False)
print("\nFeature Importance:")
print(importance)
```

---

# Lab 45: Decision Trees & Random Forest

## Duration: 75 minutes

### Exercise 1: Random Forest Classifier
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Train Random Forest
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=20,
    random_state=42
)
rf_model.fit(X_train_scaled, y_train)

# Predict
y_pred_rf = rf_model.predict(X_test_scaled)

# Evaluate
print("Random Forest Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.3f}")
print(f"F1 Score: {f1_score(y_test, y_pred_rf):.3f}")

# Feature importance
importance_rf = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(importance_rf['feature'], importance_rf['importance'])
plt.xlabel('Importance')
plt.title('Feature Importance (Random Forest)')
plt.show()

# Visualize single tree
plt.figure(figsize=(20, 10))
plot_tree(rf_model.estimators_[0], 
          feature_names=X.columns,
          class_names=['No Churn', 'Churn'],
          filled=True,
          max_depth=3)
plt.title('Sample Decision Tree from Random Forest')
plt.show()
```

---

# Lab 46: Model Selection & Cross-Validation

## Duration: 60 minutes

### Exercise 1: Compare Multiple Models
```python
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

# Define models
models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(random_state=42),
    'Naive Bayes': GaussianNB(),
    'KNN': KNeighborsClassifier(n_neighbors=5)
}

# Cross-validation
results = []
for name, model in models.items():
    scores = cross_val_score(model, X_train_scaled, y_train, 
                            cv=5, scoring='f1')
    results.append({
        'Model': name,
        'Mean F1': scores.mean(),
        'Std F1': scores.std()
    })

results_df = pd.DataFrame(results).sort_values('Mean F1', ascending=False)
print(results_df)

# Visualize
plt.figure(figsize=(12, 6))
plt.barh(results_df['Model'], results_df['Mean F1'])
plt.xlabel('Mean F1 Score (5-Fold CV)')
plt.title('Model Comparison')
plt.show()
```

---

# Lab 47: Hyperparameter Tuning

## Duration: 75 minutes

### Exercise 1: Grid Search
```python
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Grid search
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train_scaled, y_train)

print("Best parameters:")
print(grid_search.best_params_)
print(f"\nBest F1 Score: {grid_search.best_score_:.3f}")

# Test best model
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test_scaled)
print(f"Test F1 Score: {f1_score(y_test, y_pred_best):.3f}")

### Exercise 2: Random Search
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

# Random parameter distributions
param_dist = {
    'n_estimators': randint(50, 300),
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': ['sqrt', 'log2', None]
}

random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=50,
    cv=5,
    scoring='f1',
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train_scaled, y_train)
print(f"\nRandom Search Best F1: {random_search.best_score_:.3f}")
```

---

# Lab 48: Clustering - K-Means

## Duration: 60 minutes

### Exercise 1: Customer Segmentation
```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Generate customer data
np.random.seed(42)
customers = pd.DataFrame({
    'recency': np.random.exponential(30, 500),
    'frequency': np.random.poisson(10, 500),
    'monetary': np.random.gamma(2, 500, 500)
})

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(customers)

# Elbow method
inertias = []
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

ax1.plot(K_range, inertias, marker='o')
ax1.set_xlabel('Number of Clusters')
ax1.set_ylabel('Inertia')
ax1.set_title('Elbow Method')

ax2.plot(K_range, silhouette_scores, marker='o', color='orange')
ax2.set_xlabel('Number of Clusters')
ax2.set_ylabel('Silhouette Score')
ax2.set_title('Silhouette Analysis')

plt.show()

# Final clustering
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
customers['cluster'] = kmeans.fit_predict(X_scaled)

# Analyze clusters
cluster_summary = customers.groupby('cluster').agg({
    'recency': 'mean',
    'frequency': 'mean',
    'monetary': 'mean'
}).round(2)

print("Cluster Summary:")
print(cluster_summary)

# Visualize
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(customers['recency'], 
                     customers['frequency'],
                     customers['monetary'],
                     c=customers['cluster'],
                     cmap='viridis',
                     s=50,
                     alpha=0.6)
ax.set_xlabel('Recency')
ax.set_ylabel('Frequency')
ax.set_zlabel('Monetary')
plt.colorbar(scatter, label='Cluster')
plt.title('Customer Segments (RFM)')
plt.show()
```

---

# Lab 49: Dimensionality Reduction - PCA

## Duration: 60 minutes

### Exercise 1: Principal Component Analysis
```python
from sklearn.decomposition import PCA

# Generate high-dimensional data
np.random.seed(42)
n_samples = 500
n_features = 20

X = np.random.randn(n_samples, n_features)
# Add correlation structure
X[:, 1] = X[:, 0] + np.random.randn(n_samples) * 0.1
X[:, 2] = X[:, 0] - X[:, 1] + np.random.randn(n_samples) * 0.1

# Apply PCA
pca = PCA()
X_pca = pca.fit_transform(X)

# Explained variance
explained_var = pca.explained_variance_ratio_
cumulative_var = np.cumsum(explained_var)

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

ax1.bar(range(1, len(explained_var) + 1), explained_var)
ax1.set_xlabel('Principal Component')
ax1.set_ylabel('Explained Variance Ratio')
ax1.set_title('Scree Plot')

ax2.plot(range(1, len(cumulative_var) + 1), cumulative_var, marker='o')
ax2.axhline(y=0.95, color='r', linestyle='--', label='95% threshold')
ax2.set_xlabel('Number of Components')
ax2.set_ylabel('Cumulative Explained Variance')
ax2.set_title('Cumulative Explained Variance')
ax2.legend()
ax2.grid(True)

plt.show()

# Reduce to 2D for visualization
pca_2d = PCA(n_components=2)
X_2d = pca_2d.fit_transform(X)

plt.figure(figsize=(10, 8))
plt.scatter(X_2d[:, 0], X_2d[:, 1], alpha=0.5)
plt.xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.1%} variance)')
plt.ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.1%} variance)')
plt.title('Data Projected onto First 2 Principal Components')
plt.show()
```

---

# Lab 50: Neural Networks - Deep Learning Basics

## Duration: 90 minutes

### Exercise 1: TensorFlow/Keras Fundamentals
```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.datasets import make_classification

# Generate binary classification data
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build model
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(20,)),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Compile
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.AUC(name='auc')]
)

print(model.summary())

# Train
history = model.fit(
    X_train_scaled, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=0
)

# Plot training history
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

ax1.plot(history.history['loss'], label='Train Loss')
ax1.plot(history.history['val_loss'], label='Val Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training and Validation Loss')
ax1.legend()
ax1.grid(True)

ax2.plot(history.history['accuracy'], label='Train Accuracy')
ax2.plot(history.history['val_accuracy'], label='Val Accuracy')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.set_title('Training and Validation Accuracy')
ax2.legend()
ax2.grid(True)

plt.show()

# Evaluate
test_loss, test_acc, test_auc = model.evaluate(X_test_scaled, y_test)
print(f"\nTest Accuracy: {test_acc:.3f}")
print(f"Test AUC: {test_auc:.3f}")
```

---

# Lab 51-54: Advanced Deep Learning (Combined)

## Duration: 2-3 hours

### Convolutional Neural Networks (CNN)
```python
# MNIST digit recognition
from tensorflow.keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(-1, 28, 28, 1) / 255.0
X_test = X_test.reshape(-1, 28, 28, 1) / 255.0

cnn_model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

cnn_model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

history_cnn = cnn_model.fit(X_train, y_train, 
                            epochs=10, 
                            validation_split=0.2,
                            batch_size=128)

test_loss, test_acc = cnn_model.evaluate(X_test, y_test)
print(f"CNN Test Accuracy: {test_acc:.3f}")
```

### Recurrent Neural Networks (RNN/LSTM)
```python
# Time series forecasting
from tensorflow.keras.layers import LSTM

# Generate sequence data
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

# Generate time series
time = np.arange(1000)
series = np.sin(time / 50) + np.random.normal(0, 0.1, 1000)

seq_length = 50
X, y = create_sequences(series, seq_length)
X = X.reshape((X.shape[0], X.shape[1], 1))

X_train, X_test = X[:800], X[800:]
y_train, y_test = y[:800], y[800:]

lstm_model = keras.Sequential([
    LSTM(50, activation='relu', input_shape=(seq_length, 1)),
    layers.Dense(1)
])

lstm_model.compile(optimizer='adam', loss='mse')
history_lstm = lstm_model.fit(X_train, y_train, 
                               epochs=20, 
                               validation_split=0.2,
                               verbose=0)

# Forecast
predictions = lstm_model.predict(X_test)

plt.figure(figsize=(12, 6))
plt.plot(y_test, label='Actual')
plt.plot(predictions, label='Predicted')
plt.title('LSTM Time Series Forecast')
plt.legend()
plt.show()
```

---

## Deliverables (Labs 43-54)

1. ✅ Complete ML pipeline (preprocessing to deployment)
2. ✅ Churn prediction model
3. ✅ Customer segmentation
4. ✅ Hyperparameter tuned models
5. ✅ Deep learning classifiers
6. ✅ CNN image classifier
7. ✅ LSTM time series forecaster

**Next:** Chapter 13-15 - Big Data & MLOps
