# Lesson 06: Data Preprocessing & Feature Engineering

> **Prerequisites:** Lessons 01–05
> **Goal:** Learn how to clean, transform, and prepare raw data for deep learning. No matter how powerful your model architecture is, bad data preparation will always produce bad results.

---

## 1. Why Data Preprocessing is Critical

> **"Garbage In, Garbage Out"** — The most fundamental principle of machine learning.

A neural network cannot magically handle raw, messy real-world data. Consider these issues:

- House prices in dollars ($500,000) vs. number of rooms (5) — wildly different scales cause unstable gradients
- Missing values in a spreadsheet — how should a neuron handle `NaN`?
- Text labels like `"cat"`, `"dog"` — neurons need numbers, not strings
- 80% of samples are class A, 20% class B — the model will be biased

Preprocessing solves all of these problems **before** data touches the network.

```text
Raw Data  ──► [Handle Missing] ──► [Encode Categorical] ──► [Scale Features] ──► [Split] ──► Model
```

---

## 2. Types of Data

| Type | Examples | Challenge | Solution |
|---|---|---|---|
| **Numerical (Continuous)** | Age, price, temperature | Different scales, outliers | Normalize / Standardize |
| **Numerical (Discrete)** | Count of rooms, clicks | May need log transform | Log scale / Normalize |
| **Categorical (Nominal)** | Color, city, gender | Can't do math on strings | One-Hot Encoding |
| **Categorical (Ordinal)** | Rating (1-5), size (S/M/L) | Order matters | Label Encoding |
| **Text** | Reviews, tweets | Unstructured | Tokenization, Embeddings |
| **Image** | Photos, scans | Pixel scale [0–255] | Divide by 255, Augment |

---

## 3. Handling Missing Values

Real datasets almost always have missing values (`NaN`).

### Strategy 1: Remove
Drop rows with missing values — only safe when missing data is rare (<1%):

```python
X_clean = X[~np.isnan(X).any(axis=1)]
```

### Strategy 2: Mean/Median Imputation
Replace missing values with the **column mean** (for normally distributed data) or **median** (for skewed data):

$$x_{\text{missing}} \leftarrow \mu_{\text{column}} = \frac{1}{N_{\text{valid}}} \sum_{\text{valid}} x_i$$

```python
col_mean = np.nanmean(X[:, col_idx])
X[np.isnan(X[:, col_idx]), col_idx] = col_mean
```

### Strategy 3: Forward Fill
For time-series data, carry the last known value forward.

> ⚠️ **Important:** Compute imputation statistics (mean/median) on **training data only**. Apply the same statistics to validation/test data. Never let test data influence preprocessing statistics.

---

## 4. Feature Scaling

This is the **most impactful** preprocessing step for neural networks.

### 4.1 Why Scaling Matters

Unscaled features with different magnitudes cause gradient descent to take a very inefficient zigzag path:

```text
Without Scaling:          With Scaling:
Loss contours (oval)      Loss contours (circular)
  w2 ↑                     w2 ↑
     |  /←zigzag              |
     | / ←slow                |  →→→→ (direct path)
     |/                       |
     └────── w1               └────── w1
   (gradient descent          (gradient descent
    struggles)                converges fast!)
```

### 4.2 Min-Max Normalization
Scales features to the range **[0, 1]**:

$$X_{\text{norm}} = \frac{X - X_{\min}}{X_{\max} - X_{\min}}$$

- **Use when:** You know the data has hard boundaries (e.g., image pixels 0–255, probabilities)
- **Drawback:** Sensitive to outliers — one extreme value compresses all others near 0

### 4.3 Standardization (Z-Score Normalization)
Scales features to have **mean = 0** and **standard deviation = 1**:

$$X_{\text{std}} = \frac{X - \mu}{\sigma}$$

- **Use when:** Data is approximately Gaussian or you don't know the range
- **More robust** to outliers than min-max
- **Most common choice** for deep learning

### When to Use Which?

| Situation | Recommended Scaling |
|---|---|
| Image pixel values (0–255) | Min-Max → divide by 255 |
| Features with unknown range | Standardization (Z-score) |
| Neural network hidden layers | Batch Normalization (Lesson 17) |
| Tree-based models (Random Forest) | No scaling needed |

---

## 5. Encoding Categorical Variables

Neural networks require **numbers** as input. Categorical variables must be converted.

### 5.1 Label Encoding
Assign an integer to each category. Only valid when **order matters** (ordinal):

```text
Size:  S → 0,  M → 1,  L → 2,  XL → 3
```

⚠️ **Problem:** If used on nominal categories (e.g., `Red=0, Blue=1, Green=2`), the model wrongly assumes Green > Blue > Red.

### 5.2 One-Hot Encoding
Create a binary column for each category — **correct for nominal data**:

```text
Color:  Red → [1, 0, 0]
        Blue → [0, 1, 0]
        Green → [0, 0, 1]
```

$$\text{Original feature (1 column)} \longrightarrow \text{K binary columns (one per category)}$$

> 💡 **DL Note:** For large vocabularies (e.g., 50,000 words), one-hot is too sparse. We use **Embeddings** instead (Lesson 31).

---

## 6. Train / Validation / Test Split

**Never evaluate your model on data it was trained on.** You need separate data to measure real-world performance.

```text
Full Dataset (100%)
    │
    ├── Training Set   (70–80%)  → Model sees this; learns weights from it
    ├── Validation Set (10–15%)  → Used to tune hyperparameters; model never trains on this
    └── Test Set       (10–15%)  → Final evaluation; touched ONCE at the very end
```

**The Golden Rule:** The test set must remain completely untouched until you are 100% done building the model. Peeking at it — even once — leads to **data leakage**.

---

## 7. Data Leakage — The Silent Model Killer

**Data leakage** occurs when information from outside the training set improperly flows into the model, causing it to appear to perform well but fail in production.

### Common Leakage Sources:

1. **Scaling with test statistics:** Computing `mean` and `std` on the full dataset (including test), then scaling all data. Fix: fit scaler on train set only.
2. **Future data in time-series:** Using tomorrow's stock price to predict today's. Fix: strict temporal splits.
3. **Target leakage:** Including a feature that is a consequence of the target (e.g., using "hospitalized for flu" to predict "has flu"). Fix: careful feature selection.

```text
WRONG:                              CORRECT:
fit_scaler(all_data)                fit_scaler(train_data_only)
transform(train_data)               transform(train_data)
transform(test_data)       ──►      transform(test_data)   ← uses train stats!
```

---

## 8. Feature Engineering

**Feature engineering** = creating new informative features from existing ones.

### 8.1 Polynomial Features
Capture non-linear relationships:

$$x_1^2, \; x_2^2, \; x_1 \cdot x_2$$

### 8.2 Log Transform
For right-skewed distributions (e.g., income, price):

$$x_{\text{new}} = \log(x + 1)$$

### 8.3 Interaction Features
Combine two features to capture their joint effect:

$$x_{\text{price\_per\_room}} = \frac{\text{price}}{\text{num\_rooms}}$$

> 💡 **DL vs Classical ML:** Deep networks learn feature interactions automatically from raw data. In classical ML, feature engineering is a critical manual step. This is a key advantage of deep learning.

---

## 9. Python / NumPy Implementation: Full Preprocessing Pipeline

```python
import numpy as np

np.random.seed(42)

# ─────────────────────────────────────────────
# Generate raw dataset with issues
# ─────────────────────────────────────────────
# Features: [age, income, num_rooms, city_code]
raw_data = np.array([
    [25,  50000,  3,  0],
    [32,  80000,  4,  1],
    [np.nan, 120000, 5, 2],    # missing age
    [45,  np.nan, 3, 0],       # missing income
    [28,  60000,  np.nan, 1],  # missing rooms
    [55,  200000, 6,  2],
    [38,  90000,  4,  0],
    [42,  110000, 5,  1],
], dtype=float)

labels = np.array([0, 0, 1, 0, 1, 1, 0, 1])

print(f"Raw data shape: {raw_data.shape}")
print(f"Missing values per column: {np.isnan(raw_data).sum(axis=0)}")

# ─────────────────────────────────────────────
# Step 1: Train/Test Split (FIRST, before any preprocessing)
# ─────────────────────────────────────────────
split_idx = 6  # 6 train, 2 test
X_train, X_test = raw_data[:split_idx], raw_data[split_idx:]
y_train, y_test = labels[:split_idx], labels[split_idx:]

# ─────────────────────────────────────────────
# Step 2: Impute Missing Values (fit on train ONLY)
# ─────────────────────────────────────────────
def impute_with_mean(X_train, X_test):
    X_train = X_train.copy()
    X_test  = X_test.copy()
    col_means = np.nanmean(X_train, axis=0)   # compute only on train
    for col in range(X_train.shape[1]):
        train_mask = np.isnan(X_train[:, col])
        test_mask  = np.isnan(X_test[:, col])
        X_train[train_mask, col] = col_means[col]
        X_test[test_mask, col]   = col_means[col]  # use TRAIN mean on test
    return X_train, X_test

X_train, X_test = impute_with_mean(X_train, X_test)
print(f"\nAfter imputation - missing in train: {np.isnan(X_train).sum()}")

# ─────────────────────────────────────────────
# Step 3: Standardize (fit on train ONLY)
# ─────────────────────────────────────────────
def standardize(X_train, X_test):
    mu    = X_train.mean(axis=0)   # fit on train
    sigma = X_train.std(axis=0)
    sigma[sigma == 0] = 1          # avoid division by zero
    X_train_std = (X_train - mu) / sigma
    X_test_std  = (X_test  - mu) / sigma   # use TRAIN stats on test
    return X_train_std, X_test_std, mu, sigma

X_train_std, X_test_std, mu, sigma = standardize(X_train, X_test)

print(f"\nTraining means after standardization (should be ~0):")
print(np.round(X_train_std.mean(axis=0), 4))
print(f"Training stds after standardization (should be ~1):")
print(np.round(X_train_std.std(axis=0), 4))

# ─────────────────────────────────────────────
# Step 4: One-Hot Encode the last column (city: 0,1,2)
# ─────────────────────────────────────────────
def one_hot_encode(col, n_classes):
    encoded = np.zeros((len(col), n_classes))
    for i, val in enumerate(col.astype(int)):
        encoded[i, val] = 1
    return encoded

city_train = one_hot_encode(X_train_std[:, -1], n_classes=3)
city_test  = one_hot_encode(X_test_std[:, -1],  n_classes=3)

# Replace city column with one-hot encoded version
X_train_final = np.hstack([X_train_std[:, :-1], city_train])
X_test_final  = np.hstack([X_test_std[:, :-1],  city_test])

print(f"\nFinal shapes:")
print(f"  X_train: {X_train_final.shape}  (6 samples, 6 features)")
print(f"  X_test:  {X_test_final.shape}   (2 samples, 6 features)")
print(f"\nPreprocessed training data (first 2 rows):")
print(np.round(X_train_final[:2], 4))
```

---

## 10. Key Vocabulary Reference

| Term | Definition |
|---|---|
| **Feature** | An individual measurable input variable ($x_i$) |
| **Min-Max Normalization** | Scale to [0,1]: `(X - Xmin) / (Xmax - Xmin)` |
| **Standardization** | Scale to mean=0, std=1: `(X - μ) / σ` |
| **Imputation** | Filling in missing values with estimated replacements |
| **One-Hot Encoding** | Represent categorical variable as binary vector |
| **Label Encoding** | Assign integer codes to categories (ordinal only) |
| **Data Leakage** | Test set information bleeding into training, causing overly optimistic metrics |
| **Train/Val/Test Split** | Partition data for training, hyperparameter tuning, and final evaluation |
| **Feature Engineering** | Creating new informative features from existing ones |
| **Outlier** | Data point far from the rest; can distort min-max scaling |

---

## 11. Common Mistakes & Pitfalls

1. **Scaling Before Splitting:** Always split first, then fit scaler on training data only.
2. **Using Label Encoding for Nominal Data:** Creates false ordinal relationships. Always use one-hot for non-ordered categories.
3. **Dropping Too Many Rows:** If 30% of rows have missing values, impute instead of dropping or you'll lose crucial data.
4. **Not Handling Outliers:** A single extreme value can compress 99% of your data into a tiny range with min-max scaling.
5. **Forgetting to Save Scaler Parameters:** In production, you must apply the exact same `mean` and `std` (from training) to new incoming data. Always save them.

---

## 12. The Bridge to Neural Networks

After preprocessing, your data is ready:

```text
Raw Data → [Impute NaN] → [Encode Categorical] → [Standardize] → [Split]
                                                                      ↓
                                               Neural Network Input (X_train)
                                               Shape: (N, n_features)
                                               Values: centered around 0, std ≈ 1
```

This clean, normalized tensor is what flows into the first layer's matrix operation: `Z = X @ W.T + b`. Proper scaling ensures gradients are numerically stable and the network trains efficiently.

---

## 13. Interview Questions & Quiz

1. **Question 1:** You have a dataset with house prices (range: $100K–$2M) and number of bedrooms (range: 1–10). Why would training a neural network on raw (unscaled) features cause problems? Which scaling method would you choose and why?

2. **Question 2:** What is data leakage? Give a concrete example of how it can happen during feature scaling, and explain how to prevent it.

3. **Question 3:** You have a categorical variable `"country"` with 150 unique values. One-hot encoding would add 150 binary columns. What problems does this cause, and what alternatives exist for deep learning?

4. **Question 4 (Code):** Implement a `MinMaxScaler` class from scratch with `fit(X_train)` and `transform(X)` methods. Test it ensuring test data uses training statistics.

---

## 📖 References & Further Reading

- *Hands-On Machine Learning* by Aurélien Géron — **Chapter 2: End-to-End Machine Learning Project**
- scikit-learn Documentation: `preprocessing` module (for production usage)
- Andrew Ng's ML Specialization: Course 1 — Data handling and preprocessing

---

## ✅ Lesson Summary

| Concept | What You Learned |
|---|---|
| **Preprocessing Pipeline** | The correct order: Split → Impute → Encode → Scale |
| **Min-Max Normalization** | Scales to [0,1]; sensitive to outliers |
| **Standardization** | Scales to mean=0, std=1; robust; preferred for neural nets |
| **Imputation** | Replace NaN with training-set statistics |
| **One-Hot Encoding** | Correctly represent nominal categorical variables |
| **Data Leakage** | Test set must never influence preprocessing statistics |
| **Train/Val/Test Split** | Always split first; test set is sacred — touch it only once |

> **Next Lesson (07):** Machine Learning Fundamentals — Supervised vs Unsupervised learning, Bias-Variance Tradeoff, and Evaluation Metrics. We'll understand how to diagnose model performance. 🚀
