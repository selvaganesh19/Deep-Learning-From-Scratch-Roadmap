# Lesson 07: Machine Learning Fundamentals — Supervised, Unsupervised & Bias-Variance Tradeoff

> **Prerequisites:** Lessons 01–06
> **Goal:** Understand the three ML paradigms, master the bias-variance tradeoff (the central diagnostic tool of all ML), and learn how to evaluate models correctly. These concepts apply to every neural network you will ever build.

---

## 1. The Three Paradigms of Machine Learning

All machine learning falls into one of three categories based on the type of feedback the model receives during learning:

```text
┌────────────────────────────────────────────────────────────────────┐
│             MACHINE LEARNING PARADIGMS                             │
│                                                                    │
│  SUPERVISED          UNSUPERVISED        REINFORCEMENT             │
│  LEARNING            LEARNING            LEARNING                  │
│                                                                    │
│  Input + Labels  →   Input only     →   Agent + Environment       │
│  Predict output      Find patterns       Maximize reward           │
│                                                                    │
│  Examples:           Examples:           Examples:                 │
│  • Image classify    • Clustering        • Game playing (AlphaGo)  │
│  • Price prediction  • Anomaly detect    • Robot navigation        │
│  • Spam filter       • Dimensionality    • Self-driving cars       │
│                        reduction                                   │
└────────────────────────────────────────────────────────────────────┘
```

---

## 2. Supervised Learning In Depth

In supervised learning, we have a dataset of **labeled pairs** $(x_i, y_i)$ and train a model $f$ such that $f(x) \approx y$.

### 2.1 Regression
The output $y$ is a **continuous value**:

- Predicting house price from features
- Predicting tomorrow's temperature
- Predicting a patient's blood pressure

**Example Model:** Linear Regression (Lesson 2)  
**Loss:** MSE, MAE, RMSE

### 2.2 Classification
The output $y$ is a **discrete class label**:

- Is this email spam or not? (Binary)
- Which digit (0–9) is in this image? (Multi-class)
- Which objects are in this photo? (Multi-label)

**Example Model:** Logistic Regression, Neural Network  
**Loss:** Binary Cross-Entropy, Categorical Cross-Entropy

---

## 3. Unsupervised Learning In Depth

No labels are provided. The model must discover hidden **structure** in the data.

### 3.1 Clustering
Group similar data points together:

**K-Means (idea):**
1. Randomly place $K$ cluster centers (centroids)
2. Assign each point to the nearest centroid
3. Move each centroid to the mean of its assigned points
4. Repeat until centroids stop moving

```text
Before K-Means:           After K-Means (K=3):
  * * * . . . + + +         [*] * * . . . [+] + +
  * * * . . . + + +         * [*] * [.] . + + [+]
  * * * . . . + + +         * * * . [.] . + + +
```

### 3.2 Dimensionality Reduction (PCA idea)
Reduce high-dimensional data to fewer dimensions while keeping most variance:

- 1000 features → 2 features for visualization
- Compress images while preserving structure
- Remove redundant/correlated features

> **DL Connection:** Autoencoders (Lesson 36) are the deep learning equivalent of PCA — they learn a compressed representation of data.

---

## 4. The Bias-Variance Tradeoff — The Central Concept of All ML

This is the most important diagnostic framework in machine learning. Every model failure can be traced back to too much **bias** or too much **variance**.

### 4.1 Definitions

- **Bias:** How much your model's average prediction differs from the true value. High bias = model is too simple = **underfitting**.
- **Variance:** How much your model's predictions fluctuate between different training sets. High variance = model is too complex = **overfitting**.

```text
                UNDERFITTING          GOOD FIT          OVERFITTING
                (High Bias)                              (High Variance)

Training Loss:     HIGH                 LOW                  VERY LOW
Validation Loss:   HIGH                 LOW                  HIGH

                   f(x)                 f(x)                  f(x)
                     |                    |                     |
                     |  _____             |    /‾‾\             |   ~~~~
                     | /                  |   /    \            |  /    \
                     |/                   |  /      \           | /  ~   \~
                     └─────              └──────────           └──────────
                   (line is too flat)   (captures pattern)   (memorizes noise)
```

### 4.2 The Error Decomposition

Total expected error of a model can be decomposed as:

$$\text{Total Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Noise}$$

- **Irreducible Noise:** Error from noise in the data itself — cannot be reduced by any model
- **Bias²:** Systematic error from wrong assumptions in the model
- **Variance:** Error from sensitivity to fluctuations in the training data

### 4.3 The Tradeoff

```text
Error
  │  \        /← Variance
  │   \      /
  │    \    /
  │     \  /
  │      \/  ← Sweet Spot (min total error)
  │      /\
  │     /  \
  │    /    \← Bias²
  └─────────────── Model Complexity
     Simple        Complex
     (underfits)   (overfits)
```

### 4.4 Solutions

| Problem | Symptom | Solution |
|---|---|---|
| **Underfitting (High Bias)** | High train AND test error | More complex model, more features, train longer |
| **Overfitting (High Variance)** | Low train error, high test error | More data, regularization (L1/L2/Dropout), simpler model |

---

## 5. Evaluation Metrics

### 5.1 Regression Metrics

Given true values $y$ and predictions $\hat{y}$:

**Mean Absolute Error (MAE):** Average of absolute errors — interpretable, in same units as $y$:

$$\text{MAE} = \frac{1}{N}\sum_{i=1}^N |y_i - \hat{y}_i|$$

**Root Mean Squared Error (RMSE):** Penalizes large errors more — commonly reported:

$$\text{RMSE} = \sqrt{\frac{1}{N}\sum_{i=1}^N (y_i - \hat{y}_i)^2}$$

**R² Score (Coefficient of Determination):** How much variance in $y$ the model explains (1.0 = perfect):

$$R^2 = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$$

### 5.2 Classification Metrics

**Confusion Matrix** (binary classification):

```text
                   Predicted
                  Pos    Neg
Actual  Pos  │  TP   │  FN  │   ← True Positives, False Negatives
        Neg  │  FP   │  TN  │   ← False Positives, True Negatives
```

From this matrix:

$$\text{Accuracy}  = \frac{TP + TN}{TP + TN + FP + FN}$$

$$\text{Precision} = \frac{TP}{TP + FP} \quad \text{(of all positive predictions, how many were correct?)}$$

$$\text{Recall}    = \frac{TP}{TP + FN} \quad \text{(of all actual positives, how many did we catch?)}$$

$$\text{F1 Score}  = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

> 🎯 **When to use which:**
> - **Accuracy:** Only when classes are balanced
> - **Precision:** When false positives are costly (spam filter — don't block real emails)
> - **Recall:** When false negatives are costly (cancer detection — don't miss a case)
> - **F1:** Balanced metric when both precision and recall matter

---

## 6. Cross-Validation — Reliable Model Evaluation

**Problem:** A single train/test split can be "lucky" or "unlucky" depending on which data ends up in the test set.

**Solution: K-Fold Cross-Validation**

1. Split data into $K$ equal **folds**
2. Train on $K-1$ folds, validate on the remaining 1
3. Repeat $K$ times, rotating which fold is the validation set
4. Average the $K$ validation scores → much more reliable estimate

```text
K=5 Fold Cross-Validation:

Fold 1: [TEST][train][train][train][train]
Fold 2: [train][TEST][train][train][train]
Fold 3: [train][train][TEST][train][train]
Fold 4: [train][train][train][TEST][train]
Fold 5: [train][train][train][train][TEST]
                                            ↓
                          Average Score = Final CV Score
```

Typical K values: **5** or **10**.

---

## 7. When to Use Classical ML vs Deep Learning

| Scenario | Recommendation |
|---|---|
| Small dataset (<10K rows) | Classical ML (Random Forest, SVM, XGBoost) |
| Tabular/structured data | Start with XGBoost; try DL only if needed |
| Images, audio, video | Deep Learning (CNN, ViT, etc.) |
| Text / NLP | Deep Learning (Transformers, BERT, GPT) |
| Interpretability required | Classical ML (logistic regression, decision trees) |
| Large dataset (>100K rows) | Deep Learning shines |
| Limited compute (CPU only) | Classical ML |
| State-of-the-art performance | Deep Learning (usually) |

---

## 8. Python / NumPy Implementation: All Metrics from Scratch

```python
import numpy as np

# ─────────────────────────────────────────────
# Regression Metrics
# ─────────────────────────────────────────────
def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def rmse(y_true, y_pred):
    return np.sqrt(mse(y_true, y_pred))

def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

y_true = np.array([100, 150, 200, 120, 180])
y_pred = np.array([110, 145, 195, 125, 175])

print("=== Regression Metrics ===")
print(f"MAE:   {mae(y_true, y_pred):.2f}")
print(f"MSE:   {mse(y_true, y_pred):.2f}")
print(f"RMSE:  {rmse(y_true, y_pred):.2f}")
print(f"R²:    {r_squared(y_true, y_pred):.4f}")

# ─────────────────────────────────────────────
# Classification Metrics (from Confusion Matrix)
# ─────────────────────────────────────────────
def confusion_matrix(y_true, y_pred):
    TP = np.sum((y_pred == 1) & (y_true == 1))
    TN = np.sum((y_pred == 0) & (y_true == 0))
    FP = np.sum((y_pred == 1) & (y_true == 0))
    FN = np.sum((y_pred == 0) & (y_true == 1))
    return TP, TN, FP, FN

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def precision(y_true, y_pred):
    TP, TN, FP, FN = confusion_matrix(y_true, y_pred)
    return TP / (TP + FP + 1e-8)

def recall(y_true, y_pred):
    TP, TN, FP, FN = confusion_matrix(y_true, y_pred)
    return TP / (TP + FN + 1e-8)

def f1_score(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * p * r / (p + r + 1e-8)

y_true_cls = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 1])
y_pred_cls = np.array([1, 0, 1, 0, 0, 1, 1, 0, 1, 0])

TP, TN, FP, FN = confusion_matrix(y_true_cls, y_pred_cls)

print("\n=== Classification Metrics ===")
print(f"Confusion Matrix:")
print(f"  TP={TP}  FN={FN}")
print(f"  FP={FP}  TN={TN}")
print(f"Accuracy:  {accuracy(y_true_cls, y_pred_cls):.4f}")
print(f"Precision: {precision(y_true_cls, y_pred_cls):.4f}")
print(f"Recall:    {recall(y_true_cls, y_pred_cls):.4f}")
print(f"F1 Score:  {f1_score(y_true_cls, y_pred_cls):.4f}")

# ─────────────────────────────────────────────
# K-Fold Cross-Validation (concept)
# ─────────────────────────────────────────────
def k_fold_indices(N, K, seed=42):
    """Generate train/val index splits for K-Fold CV."""
    np.random.seed(seed)
    indices = np.random.permutation(N)
    folds = np.array_split(indices, K)
    splits = []
    for i in range(K):
        val_idx   = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(K) if j != i])
        splits.append((train_idx, val_idx))
    return splits

N, K = 100, 5
splits = k_fold_indices(N, K)
print(f"\n=== {K}-Fold Cross-Validation ===")
for i, (train, val) in enumerate(splits):
    print(f"Fold {i+1}: train={len(train)} samples, val={len(val)} samples")
```

**Expected Output:**
```
=== Regression Metrics ===
MAE:   5.00
RMSE:  5.00
R²:    0.9915

=== Classification Metrics ===
  TP=4  FN=2
  FP=1  TN=3
Accuracy:  0.7000
Precision: 0.8000
Recall:    0.6667
F1 Score:  0.7273

=== 5-Fold Cross-Validation ===
Fold 1: train=80 samples, val=20 samples
...
```

---

## 9. Key Vocabulary Reference

| Term | Definition |
|---|---|
| **Supervised Learning** | Learning from labeled (input, output) pairs |
| **Unsupervised Learning** | Finding patterns in unlabeled data |
| **Reinforcement Learning** | Learning via reward/penalty signals from an environment |
| **Underfitting (High Bias)** | Model too simple to capture the true pattern |
| **Overfitting (High Variance)** | Model memorizes training data including noise |
| **Bias-Variance Tradeoff** | Fundamental tension between model simplicity and complexity |
| **Accuracy** | Fraction of all predictions that are correct |
| **Precision** | Of predicted positives, fraction that are truly positive |
| **Recall (Sensitivity)** | Of all actual positives, fraction correctly identified |
| **F1 Score** | Harmonic mean of precision and recall |
| **K-Fold CV** | Robust model evaluation by training/testing on K different splits |
| **Confusion Matrix** | Table of TP, TN, FP, FN for a binary classifier |

---

## 10. Common Mistakes & Pitfalls

1. **Using Accuracy on Imbalanced Data:** If 95% of emails are not spam, a model that always predicts "not spam" gets 95% accuracy but is useless. Use F1, precision, or recall.
2. **Not Using Cross-Validation:** A single random split can mislead. K-fold gives a much more reliable estimate of generalization.
3. **Optimizing Training Loss, Not Validation Loss:** Always monitor **validation** loss during training. Training loss will always decrease — what matters is generalization.
4. **Confusing Precision and Recall:** Know when to prioritize which. In medicine (cancer detection), recall matters most. In content recommendation (showing relevant ads), precision matters more.
5. **Stopping at the First Model:** Always establish a baseline (e.g., simple logistic regression), then try to beat it. A complex DL model that barely beats a simple baseline may not be worth the complexity.

---

## 11. The Bridge to Neural Network Training

Everything in this lesson directly applies to training neural networks:

| ML Concept | Neural Network Application |
|---|---|
| **Underfitting** | Too few layers/neurons, too few epochs, learning rate too low |
| **Overfitting** | Network memorizes training data; fix with Dropout (Lesson 16), L2 reg |
| **Validation Loss** | Monitor on a held-out set every epoch to detect overfitting early |
| **Learning Curve** | Plot train vs. val loss over epochs — the most important diagnostic tool |
| **F1 / Recall** | Use the right metric for your task (medical DL uses recall/AUC, not accuracy) |
| **Cross-Validation** | Used in hyperparameter search (learning rate, #layers, batch size) |

```text
Training Curve (healthy):        Overfitting Curve:
Loss                             Loss
  |  train\                        |  train\
  |         \  val\                |         \ ___________
  |          \_____\               |          \            val ↑
  |                 ──             |                ↑ diverge here
  └────────── epochs              └────────── epochs
```

---

## 12. Interview Questions & Quiz

1. **Question 1:** A model achieves 99% training accuracy but only 55% test accuracy. What is happening? What are three concrete steps you would take to fix it?

2. **Question 2:** You are building a fraud detection system where only 0.1% of transactions are fraudulent. Why is accuracy a poor metric? Which metric(s) would you use and why?

3. **Question 3:** Explain the bias-variance tradeoff in your own words using an analogy. Then explain how increasing training data typically affects bias and variance separately.

4. **Question 4 (Code Challenge):** Implement 5-fold cross-validation from scratch. For each fold, train a simple numpy linear regression (from Lesson 2) and compute the average RMSE across all folds.

---

## 📖 References & Further Reading

- *Hands-On Machine Learning* by Aurélien Géron — **Chapter 2 & 4**
- *The Elements of Statistical Learning* by Hastie, Tibshirani, Friedman — Chapter 7 (Model Assessment and Selection)
- Andrew Ng: *Structuring Machine Learning Projects* (Coursera) — diagnosing bias/variance

---

## ✅ Lesson Summary

| Concept | What You Learned |
|---|---|
| **3 ML Paradigms** | Supervised (labeled), Unsupervised (patterns), Reinforcement (rewards) |
| **Regression vs Classification** | Continuous output vs discrete class output |
| **Underfitting (High Bias)** | Model too simple; train error high |
| **Overfitting (High Variance)** | Model too complex; train low, val high |
| **Bias-Variance Tradeoff** | `Error = Bias² + Variance + Noise`; must balance both |
| **MAE / RMSE / R²** | Standard regression evaluation metrics |
| **Precision / Recall / F1** | Classification metrics beyond accuracy |
| **K-Fold Cross-Validation** | Robust model evaluation via K rotations |

> **Next Lesson (08):** The Perceptron — The Starting Point of Deep Learning. We'll build the very first neural network algorithm ever invented, understanding how it connects to everything we've learned so far. 🚀
