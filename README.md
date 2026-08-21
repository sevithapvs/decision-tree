# BigMart Sales Prediction using ID3 Decision Tree Algorithm

## Project Overview

This project implements an **ID3 (Iterative Dichotomiser 3) Decision Tree Classification Algorithm** on the **BigMart Sales Prediction Dataset**.

The objective of this project is to classify product sales into three categories based on product and outlet-related features.

Since the original dataset contains numerical values and ID3 works with categorical attributes, numerical features and the target variable were transformed into categorical classes before training the model.

The model predicts:

- Low Sales
- Medium Sales
- High Sales

The Decision Tree classifier uses the **Entropy criterion**, which follows the ID3 algorithm approach.

---

## Dataset Description

Dataset Used:

```
BigMartSalesPrediction_Train.csv
```

The dataset contains:

```
8523 records and 12 attributes
```

### Features

| Feature | Description |
|---------|-------------|
| Item_Identifier | Unique product identifier |
| Item_Weight | Weight of the product |
| Item_Fat_Content | Fat content category |
| Item_Visibility | Product visibility percentage |
| Item_Type | Product category |
| Item_MRP | Maximum Retail Price |
| Outlet_Identifier | Unique outlet identifier |
| Outlet_Establishment_Year | Year outlet was established |
| Outlet_Size | Size of outlet |
| Outlet_Location_Type | Outlet location category |
| Outlet_Type | Type of outlet |
| Item_Outlet_Sales | Target variable |

---

## Project Structure

```
BigMart-ID3-Decision-Tree/
│
├── BigMartSalesPrediction_Train.csv
│
├── id3_decision_tree.py
│
├── README.md
│
└── requirements.txt
```

---

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Matplotlib

---

# Data Preprocessing

## 1. Loading Dataset

The dataset was loaded using Pandas.

```python
df = pd.read_csv("BigMartSalesPrediction_Train.csv")
```

The dataset shape and first few records were examined.

Dataset size:

```
8523 rows × 12 columns
```

---

## 2. Missing Value Analysis

Missing values were identified using:

```python
df.isnull().sum()
```

Missing values detected:

| Column | Missing Values |
|--------|----------------|
| Item_Weight | 1463 |
| Outlet_Size | 2410 |

Handling strategy:

- `Item_Weight` missing values were replaced using the median.
- `Outlet_Size` missing values were replaced using the mode.

After preprocessing:

```
No missing values remained
```

---

## 3. Duplicate Data Check

Duplicate records were checked using:

```python
df.duplicated().sum()
```

Result:

```
Duplicate rows: 0
```

---

## 4. Feature Selection

The following columns were removed:

```
Item_Identifier
Outlet_Identifier
```

Reason:

These columns contain unique identifiers and do not provide useful information for prediction.

---

# Feature Transformation

ID3 requires categorical attributes.

The following numerical features were converted into categorical values using quantile-based binning:

- Item_Weight
- Item_Visibility
- Item_MRP
- Outlet_Establishment_Year

The numerical values were divided into three categories:

```
Low
Medium
High
```

using:

```python
pd.qcut()
```

---

# Target Variable Transformation

The original target variable:

```
Item_Outlet_Sales
```

was a continuous numerical value.

Since this project performs classification, the sales values were converted into three classes:

```
Low
Medium
High
```

using quantile binning.

Final distribution:

```
Low       2845
Medium    2838
High      2840
```

---

# Data Encoding

All categorical columns were converted into category datatype.

```python
astype("category")
```

Unused categories were removed using:

```python
remove_unused_categories()
```

Before model training, categorical values were encoded using:

```python
LabelEncoder()
```

Example:

```
Low    → 0
Medium → 1
High   → 2
```

---

# ID3 Decision Tree Implementation

## Algorithm

ID3 selects the best attribute using **Entropy** and **Information Gain**.

Entropy formula:

\[
Entropy(S) = -\sum p_i log_2(p_i)
\]

The attribute with maximum information gain is selected as the decision node.

---

# Model Training

The dataset was divided into:

### Input Features

```
X
```

### Target Variable

```
y
```
The Decision Tree model was created using:
```python
DecisionTreeClassifier(
    criterion="entropy",
    max_depth=3,
    random_state=42
)
```
The entropy criterion allows the classifier to follow the ID3 splitting method.

---

# Decision Tree Visualization
The visualization displays:
- Root node
- Decision conditions
- Branches
- Leaf nodes
- Predicted classes
The tree explains how product and outlet features influence sales categories.

---

# Execution

The program will:
1. Load the dataset
2. Perform preprocessing
3. Convert numerical features into categories
4. Encode categorical values
5. Train the ID3 Decision Tree model
6. Display the decision tree visualization

---

# Results
The project successfully:
- Cleaned and preprocessed the BigMart dataset
- Handled missing values
- Converted numerical attributes into categorical attributes
- Implemented an ID3-based Decision Tree classifier
- Generated a visual representation of the decision tree
- Classified products into Low, Medium, and High sales categories
---

Implementation of ID3 Decision Tree Algorithm on BigMart Sales Dataset.
