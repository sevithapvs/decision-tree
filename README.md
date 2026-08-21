# BigMart Sales Prediction using ID3 Decision Tree Algorithm

## Project Overview

This project implements an **ID3 (Iterative Dichotomiser 3) Decision Tree Classification Algorithm** on the **BigMart Sales Prediction dataset**.

The original dataset contains information about products and outlets, with the goal of predicting the sales category of each product. Since ID3 works with categorical attributes and classification targets, numerical features and sales values were converted into categorical ranges before training the model.

The final model classifies **Item Outlet Sales** into three categories:

* Low Sales
* Medium Sales
* High Sales

The decision tree is built using the **Entropy criterion**, which represents the ID3 algorithm approach.

---

# Dataset Description

The dataset used:

```
BigMartSalesPrediction_Train.csv
```

The dataset contains **8523 records** and **12 attributes**.

### Original Features

| Feature                   | Description                       |
| ------------------------- | --------------------------------- |
| Item_Identifier           | Unique identifier of each product |
| Item_Weight               | Weight of the product             |
| Item_Fat_Content          | Fat content category              |
| Item_Visibility           | Visibility percentage of product  |
| Item_Type                 | Type of product                   |
| Item_MRP                  | Maximum Retail Price              |
| Outlet_Identifier         | Unique outlet ID                  |
| Outlet_Establishment_Year | Year outlet was established       |
| Outlet_Size               | Size of outlet                    |
| Outlet_Location_Type      | Location category of outlet       |
| Outlet_Type               | Type of outlet                    |
| Item_Outlet_Sales         | Sales amount (Target Variable)    |

---

# Libraries Used

The following Python libraries were used:

```python
import pandas as pd
from math import log2
```

Additional libraries used:

* Scikit-learn for Decision Tree implementation
* Matplotlib for tree visualization

---

# Data Preprocessing

## 1. Loading Dataset

The dataset was loaded using Pandas.

```python
df = pd.read_csv("BigMartSalesPrediction_Train.csv")
```

The first few rows and dataset dimensions were checked.

Output:

```
Rows: 8523
Columns: 12
```

---

# 2. Dataset Information Analysis

The dataset structure was examined using:

```python
df.info()
```

This helped identify:

* Data types
* Missing values
* Number of records
* Memory usage

The dataset contained:

* Numerical attributes
* Categorical attributes
* Missing values in Item Weight and Outlet Size

---

# 3. Handling Missing Values

Missing values were checked using:

```python
df.isnull().sum()
```

Missing values found:

| Column      | Missing Values |
| ----------- | -------------- |
| Item_Weight | 1463           |
| Outlet_Size | 2410           |

### Filling Missing Values

### Item Weight

Since Item Weight had low skewness, missing values were replaced using the median.

```python
df["Item_Weight"] = df["Item_Weight"].fillna(
    df["Item_Weight"].median()
)
```

### Outlet Size

Categorical missing values were replaced using the most frequent category.

```python
df["Outlet_Size"] = df["Outlet_Size"].fillna(
    df["Outlet_Size"].mode()[0]
)
```

After preprocessing, no missing values remained.

---

# 4. Checking Duplicate Data

Duplicate rows were checked:

```python
df.duplicated().sum()
```

Result:

```
Duplicate rows: 0
```

No duplicate records were present.

---

# 5. Removing Unnecessary Columns

The following columns were removed:

* Item Identifier
* Outlet Identifier

Reason:

These columns contain unique IDs and do not contribute meaningful information for classification.

Code:

```python
df = df.drop(
    ["Item_Identifier", "Outlet_Identifier"],
    axis=1
)
```

---

# 6. Converting Numerical Features into Categories

ID3 works mainly with categorical attributes.

Therefore, numerical attributes were converted into three categories using quantile-based binning.

Columns converted:

* Item Weight
* Item Visibility
* Item MRP
* Outlet Establishment Year

Categories created:

```
Low
Medium
High
```

Example:

```python
df[column] = pd.qcut(
    df[column],
    q=3,
    labels=["Low","Medium","High"]
)
```

---

# 7. Converting Target Variable

The target variable:

```
Item_Outlet_Sales
```

was originally a continuous numerical value.

Since ID3 is a classification algorithm, sales values were converted into categories:

```
Low
Medium
High
```

using quantile binning.

Code:

```python
df["Item_Outlet_Sales"] = pd.qcut(
    df["Item_Outlet_Sales"],
    q=3,
    labels=["Low","Medium","High"]
)
```

Final class distribution:

```
Low       2845
Medium    2838
High      2840
```

---

# 8. Converting Object Data into Category Format

All categorical columns were converted into Pandas category datatype.

```python
categorical_columns = df.select_dtypes(
    include=["object","string"]
).columns

for column in categorical_columns:
    df[column] = df[column].astype("category")
```

This ensures all categorical features have consistent data types.

---

# ID3 Decision Tree Implementation

## Algorithm Used

ID3 selects the best attribute using:

### Entropy

Entropy measures the impurity of the dataset.

Formula:

[
Entropy(S) = -\sum p_i log_2(p_i)
]

The attribute with maximum information gain is selected as the decision node.

---

# Model Training

The categorical values were encoded into numerical values using:

```python
LabelEncoder()
```

Example:

```
Low    -> 0
Medium -> 1
High   -> 2
```

The target variable:

```
Item_Outlet_Sales
```

was separated from input features.

```python
X = data.drop(
    "Item_Outlet_Sales",
    axis=1
)

y = data["Item_Outlet_Sales"]
```

---

# Decision Tree Model

The ID3 decision tree was implemented using:

```python
DecisionTreeClassifier(
    criterion="entropy"
)
```

The entropy criterion makes the decision tree behave like an ID3 algorithm.

Model parameters:

```python
max_depth = 3
random_state = 42
```

---

# Decision Tree Visualization

The trained tree was displayed using:

```python
plot_tree()
```

The visualization shows:

* Root node
* Decision conditions
* Branches
* Leaf nodes
* Predicted classes

The tree represents how different product and outlet characteristics influence sales categories.

---

# Project Workflow

```
Load Dataset
      |
      ↓
Data Exploration
      |
      ↓
Check Missing Values
      |
      ↓
Handle Missing Data
      |
      ↓
Remove Unnecessary Features
      |
      ↓
Convert Numerical Data into Categories
      |
      ↓
Convert Sales into Classes
      |
      ↓
Encode Categories
      |
      ↓
Train ID3 Decision Tree
      |
      ↓
Visualize Decision Tree
```

---

---

# Results
* Cleaned the BigMart dataset
* Converted numerical attributes into categorical attributes
* Applied ID3 Decision Tree Classification
* Generated a decision tree visualization
* Classified products into Low, Medium, and High sales categories

---
