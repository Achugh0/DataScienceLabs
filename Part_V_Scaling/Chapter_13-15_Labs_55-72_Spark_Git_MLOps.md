# Lab 55-72: Spark, Git, MLOps - Production Data Science

## Chapter 13-15: Scaling & Production Systems

### 18 Labs for Enterprise Data Science

---

# Lab 55-60: Apache Spark Fundamentals

## Lab 55: Spark Basics & RDDs
```python
from pyspark.sql import SparkSession
import pyspark.sql.functions as F

# Initialize Spark
spark = SparkSession.builder \
    .appName("DataScience") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

# Create DataFrame
data = [
    (1, "Alice", 25, 50000),
    (2, "Bob", 30, 60000),
    (3, "Charlie", 35, 75000),
    (4, "Diana", 28, 55000),
]

df = spark.createDataFrame(data, ["id", "name", "age", "salary"])
df.show()

# Basic operations
df.select("name", "salary").show()
df.filter(df.salary > 55000).show()
df.groupBy("age").count().show()

# SQL queries
df.createOrReplaceTempView("employees")
spark.sql("SELECT * FROM employees WHERE age > 28").show()
```

## Lab 56: Spark DataFrames & Transformations
```python
# Load large dataset (simulated)
large_df = spark.range(1000000).withColumn(
    "value", F.rand() * 1000
).withColumn(
    "category", (F.rand() * 10).cast("int")
)

# Transformations
result = large_df \
    .filter(F.col("value") > 500) \
    .groupBy("category") \
    .agg(
        F.count("*").alias("count"),
        F.avg("value").alias("avg_value"),
        F.max("value").alias("max_value")
    ) \
    .orderBy(F.desc("avg_value"))

result.show()

# Window functions
from pyspark.sql.window import Window

window_spec = Window.partitionBy("category").orderBy("value")

windowed_df = large_df.withColumn(
    "rank", F.row_number().over(window_spec)
).withColumn(
    "running_avg", F.avg("value").over(window_spec)
)

windowed_df.filter(F.col("rank") <= 10).show()
```

## Lab 57: Spark ML - Machine Learning at Scale
```python
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Create training data
training_data = spark.createDataFrame([
    (1.0, 2.0, 3.0, 1),
    (2.0, 3.0, 4.0, 0),
    # ... more data
], ["feature1", "feature2", "feature3", "label"])

# Feature engineering
assembler = VectorAssembler(
    inputCols=["feature1", "feature2", "feature3"],
    outputCol="features"
)

scaler = StandardScaler(
    inputCol="features",
    outputCol="scaled_features"
)

# Build pipeline
from pyspark.ml import Pipeline

lr = LogisticRegression(
    featuresCol="scaled_features",
    labelCol="label",
    maxIter=10
)

pipeline = Pipeline(stages=[assembler, scaler, lr])

# Train
model = pipeline.fit(training_data)

# Predict
predictions = model.transform(training_data)
predictions.select("label", "prediction", "probability").show()

# Evaluate
evaluator = BinaryClassificationEvaluator()
auc = evaluator.evaluate(predictions)
print(f"AUC: {auc:.3f}")
```

## Lab 58: Spark Streaming
```python
from pyspark.sql.functions import window, current_timestamp

# Simulated streaming data
streaming_df = spark.readStream \
    .format("rate") \
    .option("rowsPerSecond", 10) \
    .load()

# Add processing timestamp
processed = streaming_df.withColumn(
    "processing_time", current_timestamp()
)

# Windowed aggregation
windowed_counts = processed \
    .groupBy(window("timestamp", "10 seconds")) \
    .count()

# Write stream
query = windowed_counts.writeStream \
    .outputMode("complete") \
    .format("console") \
    .start()

# query.awaitTermination()  # Uncomment for continuous execution
```

## Lab 59: Spark Performance Optimization
```python
# Caching
df_cached = large_df.cache()
df_cached.count()  # First run: slow
df_cached.count()  # Second run: fast

# Partitioning
optimized_df = large_df.repartition(100, "category")

# Broadcast joins
from pyspark.sql.functions import broadcast

small_df = spark.createDataFrame([
    (1, "Category A"),
    (2, "Category B"),
], ["id", "name"])

# Broadcast small DataFrame
joined = large_df.join(broadcast(small_df), 
                       large_df.category == small_df.id)

# Explain plan
joined.explain(True)
```

## Lab 60: Spark on AWS EMR
```python
# Configuration for production
spark_prod = SparkSession.builder \
    .appName("Production ETL") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
    .config("spark.dynamicAllocation.enabled", "true") \
    .config("spark.shuffle.service.enabled", "true") \
    .getOrCreate()

# Read from S3
df_s3 = spark_prod.read.parquet("s3://bucket/data/")

# Process
result = df_s3 \
    .filter(...) \
    .groupBy(...) \
    .agg(...)

# Write to S3
result.write \
    .mode("overwrite") \
    .partitionBy("date") \
    .parquet("s3://bucket/output/")
```

---

# Lab 61-66: Git & Version Control

## Lab 61: Git Fundamentals
```bash
# Initialize repository
git init
git config user.name "Your Name"
git config user.email "your.email@example.com"

# Basic workflow
git add script.py
git commit -m "Add data processing script"
git log --oneline

# Branching
git branch feature/new-model
git checkout feature/new-model
git checkout -b feature/new-model  # Create and switch

# Merging
git checkout main
git merge feature/new-model
```

## Lab 62: Collaboration with Git
```bash
# Clone repository
git clone https://github.com/user/repo.git

# Remote operations
git remote add origin https://github.com/user/repo.git
git push -u origin main
git pull origin main

# Fetch and merge
git fetch origin
git merge origin/main

# Pull request workflow
git checkout -b fix/bug-123
# Make changes
git add .
git commit -m "Fix: Resolve data cleaning bug"
git push origin fix/bug-123
# Create PR on GitHub
```

## Lab 63: Git for Data Science
```python
# .gitignore for data science projects
"""
# Data
data/
*.csv
*.parquet
*.db

# Models
models/
*.h5
*.pkl
*.joblib

# Notebooks
.ipynb_checkpoints/
*-checkpoint.ipynb

# Python
__pycache__/
*.pyc
.env

# IDE
.vscode/
.idea/
"""

# DVC (Data Version Control)
"""
dvc init
dvc add data/large_dataset.csv
git add data/large_dataset.csv.dvc .gitignore
git commit -m "Add large dataset"

dvc remote add -d storage s3://mybucket/dvcstore
dvc push
"""
```

## Lab 64-66: Advanced Git Techniques
```bash
# Rebase for clean history
git rebase -i HEAD~3

# Cherry-pick commits
git cherry-pick abc123

# Stash changes
git stash
git stash pop

# Reset and revert
git reset --hard HEAD~1  # Danger!
git revert abc123  # Safe

# Bisect for debugging
git bisect start
git bisect bad
git bisect good abc123

# Submodules
git submodule add https://github.com/user/lib.git
git submodule update --init --recursive
```

---

# Lab 67-72: MLOps & Production ML

## Lab 67: Model Versioning with MLflow
```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier

# Start MLflow run
with mlflow.start_run():
    # Train model
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(X_train, y_train)
    
    # Log parameters
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 10)
    
    # Log metrics
    accuracy = rf.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)
    
    # Log model
    mlflow.sklearn.log_model(rf, "model")
    
    # Log artifacts
    mlflow.log_artifact("confusion_matrix.png")

# Load model
model_uri = "runs:/<run_id>/model"
loaded_model = mlflow.sklearn.load_model(model_uri)
```

## Lab 68: Model Deployment with Flask
```python
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load model
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data
    data = request.get_json()
    features = np.array(data['features']).reshape(1, -1)
    
    # Preprocess
    features_scaled = scaler.transform(features)
    
    # Predict
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0]
    
    return jsonify({
        'prediction': int(prediction),
        'probability': {
            'class_0': float(probability[0]),
            'class_1': float(probability[1])
        }
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

## Lab 69: Docker for ML Models
```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY model.pkl scaler.pkl app.py ./

EXPOSE 5000

CMD ["python", "app.py"]
```

```bash
# Build and run
docker build -t ml-model:latest .
docker run -p 5000:5000 ml-model:latest

# Push to registry
docker tag ml-model:latest myregistry/ml-model:v1.0
docker push myregistry/ml-model:v1.0
```

## Lab 70: CI/CD for ML with GitHub Actions
```yaml
# .github/workflows/ml-pipeline.yml
name: ML Pipeline

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.9
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      
      - name: Run tests
        run: |
          pytest tests/
      
      - name: Train model
        run: |
          python train.py
      
      - name: Evaluate model
        run: |
          python evaluate.py
  
  deploy:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - name: Deploy to production
        run: |
          # Deploy steps
          echo "Deploying model..."
```

## Lab 71: Model Monitoring
```python
import pandas as pd
from datetime import datetime

class ModelMonitor:
    """Monitor model performance in production"""
    
    def __init__(self, model):
        self.model = model
        self.predictions = []
        self.actuals = []
        
    def predict_and_log(self, X):
        """Make prediction and log"""
        prediction = self.model.predict(X)
        
        log_entry = {
            'timestamp': datetime.now(),
            'prediction': prediction,
            'features': X.tolist()
        }
        
        self.predictions.append(log_entry)
        return prediction
    
    def log_actual(self, y_true):
        """Log actual outcome"""
        self.actuals.append({
            'timestamp': datetime.now(),
            'actual': y_true
        })
    
    def calculate_metrics(self):
        """Calculate performance metrics"""
        if len(self.actuals) < 10:
            return None
        
        df = pd.DataFrame({
            'predictions': [p['prediction'] for p in self.predictions[-100:]],
            'actuals': [a['actual'] for a in self.actuals[-100:]]
        })
        
        accuracy = (df['predictions'] == df['actuals']).mean()
        
        return {
            'accuracy': accuracy,
            'sample_size': len(df)
        }
    
    def detect_drift(self, X_prod, X_train):
        """Detect data drift"""
        from scipy.stats import ks_2samp
        
        drift_detected = []
        for col in range(X_prod.shape[1]):
            stat, pvalue = ks_2samp(X_prod[:, col], X_train[:, col])
            if pvalue < 0.05:
                drift_detected.append(col)
        
        return drift_detected

# Usage
monitor = ModelMonitor(model)
prediction = monitor.predict_and_log(new_data)
monitor.log_actual(true_label)
metrics = monitor.calculate_metrics()
```

## Lab 72: Production ML Architecture
```python
# Complete production pipeline

# 1. Feature Store
class FeatureStore:
    """Centralized feature management"""
    def __init__(self):
        self.features = {}
    
    def register_feature(self, name, func):
        self.features[name] = func
    
    def get_features(self, entity_id):
        return {
            name: func(entity_id) 
            for name, func in self.features.items()
        }

# 2. Model Registry
class ModelRegistry:
    """Manage model versions"""
    def __init__(self):
        self.models = {}
    
    def register_model(self, name, version, model):
        key = f"{name}:{version}"
        self.models[key] = model
    
    def get_model(self, name, version='latest'):
        return self.models.get(f"{name}:{version}")
    
    def promote_to_production(self, name, version):
        self.models[f"{name}:production"] = self.models[f"{name}:{version}"]

# 3. Prediction Service
class PredictionService:
    """Serve predictions"""
    def __init__(self, model_registry, feature_store, monitor):
        self.model_registry = model_registry
        self.feature_store = feature_store
        self.monitor = monitor
    
    def predict(self, entity_id):
        # Get features
        features = self.feature_store.get_features(entity_id)
        
        # Get model
        model = self.model_registry.get_model("churn", "production")
        
        # Predict
        X = self._prepare_features(features)
        prediction = self.monitor.predict_and_log(X)
        
        return prediction
```

---

## Deliverables (Labs 55-72)

1. ✅ Spark ETL pipeline for big data
2. ✅ Spark ML model at scale
3. ✅ Git workflow for collaboration
4. ✅ MLflow experiment tracking
5. ✅ Dockerized ML API
6. ✅ CI/CD pipeline for ML
7. ✅ Model monitoring system
8. ✅ Production ML architecture

**Next:** Chapter 16-17 - Capstone & Career Launch
