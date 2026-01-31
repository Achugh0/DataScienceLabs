# Lab 04: Setting Up Your Complete Data Science Environment

## Chapter 1: Charting the Course: What Is Data Science?

### Learning Objectives
- Set up a local Python environment
- Install essential data science libraries
- Configure Jupyter Notebook
- Understand virtual environments

### Duration
90 minutes

---

## Part 1: Environment Options

### Exercise 1.1: Environment Comparison
Research and compare these options:

| Feature | Google Colab | Anaconda | pip + venv | JupyterLab |
|---------|-------------|----------|------------|------------|
| **Cost** | | | | |
| **Setup Time** | | | | |
| **Computing Power** | | | | |
| **Offline Access** | | | | |
| **Collaboration** | | | | |
| **Best For** | | | | |

### Exercise 1.2: When to Use What
For each scenario, recommend an environment:

1. Quick prototyping with GPU access: _______________
2. Production-level data pipeline: _______________
3. Collaborative class project: _______________
4. Personal learning on a laptop: _______________

---

## Part 2: Local Installation (Choose One Path)

### Path A: Anaconda Installation

**Step 1: Download and Install**
- Visit anaconda.com
- Download for your OS
- Install with default settings

**Step 2: Verify Installation**
```bash
conda --version
python --version
jupyter --version
```

**Step 3: Create First Environment**
```bash
conda create --name ds_lab python=3.10
conda activate ds_lab
conda install pandas numpy matplotlib seaborn scikit-learn jupyter
```

**Document your versions:**
- Conda version: _______________
- Python version: _______________
- Pandas version: _______________

### Path B: pip + venv Installation

**Step 1: Verify Python**
```bash
python3 --version
pip3 --version
```

**Step 2: Create Virtual Environment**
```bash
python3 -m venv ds_env
source ds_env/bin/activate  # On Windows: ds_env\Scripts\activate
```

**Step 3: Install Libraries**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter notebook
pip freeze > requirements.txt
```

**Step 4: Launch Jupyter**
```bash
jupyter notebook
```

---

## Part 3: Essential Libraries

### Exercise 3.1: Library Purposes
Match each library to its primary purpose:

| Library | Purpose | Use Case Example |
|---------|---------|------------------|
| pandas | | |
| numpy | | |
| matplotlib | | |
| seaborn | | |
| scikit-learn | | |
| scipy | | |
| requests | | |
| beautifulsoup4 | | |

### Exercise 3.2: Import Test
Create a new Jupyter notebook and verify all imports:

```python
# Test all imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets

# Print versions
print(f"Pandas: {pd.__version__}")
print(f"NumPy: {np.__version__}")

# Quick test
data = {'A': [1, 2, 3], 'B': [4, 5, 6]}
df = pd.DataFrame(data)
print("\nTest DataFrame:")
print(df)

# Simple plot
plt.plot([1, 2, 3], [1, 4, 9])
plt.title("Test Plot")
plt.show()

print("\n✅ All libraries working!")
```

---

## Part 4: Jupyter Notebook Mastery

### Exercise 4.1: Notebook Features
In a new notebook, demonstrate:

1. **Code cell with execution count**
2. **Markdown cell with:**
   - Headers (H1, H2, H3)
   - Bold and italic text
   - Bullet points
   - Numbered lists
   - Code blocks
   - Links
   - Images

3. **Magic commands:**
```python
# Time a code cell
%timeit [x**2 for x in range(1000)]

# Display matplotlib plots inline
%matplotlib inline

# List all variables
%whos

# Run external Python file
# %run my_script.py

# Install package from notebook
# !pip install package_name
```

4. **Keyboard shortcuts:**
- List 5 shortcuts you find most useful:
  1. _______________
  2. _______________
  3. _______________
  4. _______________
  5. _______________

### Exercise 4.2: Notebook Best Practices
Refactor this poorly organized notebook:

**Bad Notebook:**
```python
# Cell 1
import pandas as pd
data = pd.read_csv('data.csv')

# Cell 2
import numpy as np

# Cell 3
result = data.mean()

# Cell 4
import matplotlib.pyplot as plt

# Cell 5
plt.plot(result)
```

**Your Improved Version:**
```
# Organize into logical sections with markdown headers
# Group all imports together
# Add explanatory text
# Create clear flow
```

---

## Part 5: Virtual Environment Management

### Exercise 5.1: Create Project-Specific Environments
Practice creating environments for different projects:

```bash
# Environment 1: Machine Learning Project
conda create --name ml_project python=3.9
conda activate ml_project
conda install scikit-learn tensorflow pandas jupyter

# Environment 2: Web Scraping Project
conda create --name scraping python=3.10
conda activate scraping
conda install beautifulsoup4 requests pandas selenium
```

**Document:**
- Why use separate environments?
- How to list all environments?
- How to delete an environment?
- How to export environment to share with team?

### Exercise 5.2: requirements.txt Management
Create a requirements.txt file:

```txt
pandas==1.5.3
numpy==1.24.2
matplotlib==3.7.1
seaborn==0.12.2
scikit-learn==1.2.2
jupyter==1.0.0
```

**Commands to know:**
```bash
# Install from requirements
pip install -r requirements.txt

# Generate requirements from current environment
pip freeze > requirements.txt

# Install with version constraints
pip install 'pandas>=1.5.0,<2.0.0'
```

---

## Part 6: Troubleshooting Common Issues

### Exercise 6.1: Error Solutions
For each error, provide the solution:

**Error 1:** "ModuleNotFoundError: No module named 'pandas'"
**Solution:** _______________

**Error 2:** "Kernel dead" in Jupyter
**Solution:** _______________

**Error 3:** "Permission denied" during pip install
**Solution:** _______________

**Error 4:** Wrong Python version being used
**Solution:** _______________

### Exercise 6.2: Debug This Setup
```python
# This code has environment issues. Fix it.
import pandas
import numpyyy  # Typo
from sklearn import model_selection

data = pd.DataFrame({'A': [1, 2, 3]})  # pd not defined
print(data)
```

**What's wrong and how to fix:**
1. _______________
2. _______________
3. _______________

---

## Part 7: Cloud Alternatives

### Exercise 7.1: Cloud Platform Comparison
Research and document:

| Platform | Free Tier | GPU Access | Storage | Collaboration |
|----------|-----------|------------|---------|---------------|
| Google Colab | | | | |
| Kaggle Kernels | | | | |
| Azure Notebooks | | | | |
| AWS SageMaker | | | | |
| Deepnote | | | | |

### Exercise 7.2: Colab Advanced Features
Explore Google Colab:
1. Mount Google Drive
2. Upload files
3. Install custom package
4. Use GPU runtime
5. Share notebook with collaborator

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Check GPU
import tensorflow as tf
print("GPU Available:", tf.config.list_physical_devices('GPU'))
```

---

## Deliverables

Submit:
1. ✅ Environment comparison table (Exercise 1.1)
2. ✅ Screenshots of successful installation (Part 2)
3. ✅ Library purposes table (Exercise 3.1)
4. ✅ Working import test notebook (Exercise 3.2)
5. ✅ Notebook demonstrating all features (Exercise 4.1)
6. ✅ Improved notebook organization (Exercise 4.2)
7. ✅ Virtual environment documentation (Exercise 5.1)
8. ✅ requirements.txt file (Exercise 5.2)
9. ✅ Troubleshooting solutions (Exercise 6.1)
10. ✅ Cloud platform comparison (Exercise 7.1)

---

## Reflection Questions

1. Which environment setup do you prefer and why?
2. Why are virtual environments important?
3. What's your strategy for keeping environments organized?

---

**Next Lab:** Lab 05 - The Five Habits of Effective Data Scientists
