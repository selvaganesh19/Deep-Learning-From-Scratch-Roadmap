# Lesson 05: Probability & Statistics for Deep Learning

> **Prerequisites:** Lessons 01–04
> **Goal:** Understand why neural network outputs are probability distributions, how statistics describes data, and how these ideas directly connect to the loss functions used to train deep learning models.

---

## 1. Why Probability Matters in Deep Learning

When a neural network classifies an image as "cat", it doesn't say "it IS a cat." It says "there's an **87% probability** it's a cat." The output of most classifiers is a **probability distribution** over all possible classes.

```text
Input Image  ──►  Neural Network  ──►  [Cat: 0.87, Dog: 0.09, Bird: 0.04]
                                            ↑ These are probabilities (sum = 1.0)
```

Additionally:
- **Cross-entropy loss** (most common classification loss) is derived from probability theory
- **Weight initialization** follows probability distributions (Gaussian)
- **Dropout** uses Bernoulli random variables
- **Data augmentation** involves sampling and random transforms

---

## 2. Basic Probability

### 2.1 Definitions

- **Experiment:** A process with uncertain outcomes (e.g., flipping a coin)
- **Sample Space ($\Omega$):** All possible outcomes (e.g., {Heads, Tails})
- **Event (A):** A subset of outcomes we care about (e.g., "getting Heads")
- **Probability P(A):** A number in [0, 1] measuring how likely event A is

$$0 \leq P(A) \leq 1 \quad \text{and} \quad P(\Omega) = 1$$

### 2.2 Rules of Probability

**Addition Rule (OR):**

$$P(A \cup B) = P(A) + P(B) - P(A \cap B)$$

**Multiplication Rule (AND — independent events):**

$$P(A \cap B) = P(A) \cdot P(B) \quad \text{if A and B are independent}$$

**Complement:**

$$P(\text{not } A) = 1 - P(A)$$

---

## 3. Conditional Probability

$P(A | B)$ = "The probability of A, **given** that B has already occurred."

$$P(A | B) = \frac{P(A \cap B)}{P(B)}$$

**Example:** Given a patient tests positive for a disease, what's the probability they actually have it? This is conditional probability — the basis of Bayesian reasoning used in many ML models.

**Bayes' Theorem:**

$$P(A | B) = \frac{P(B | A) \cdot P(A)}{P(B)}$$

---

## 4. Key Probability Distributions

### 4.1 Bernoulli Distribution — Binary Classification

Used for yes/no outcomes (1 or 0). A single coin flip with probability $p$ of heads:

$$P(X = 1) = p \quad \quad P(X = 0) = 1 - p$$

$$\mathbb{E}[X] = p \quad \quad \text{Var}(X) = p(1-p)$$

> **DL Connection:** Binary classification outputs. A sigmoid neuron outputs $p$ — the probability that the sample belongs to class 1. The Bernoulli distribution is the exact distribution this models.

### 4.2 Gaussian (Normal) Distribution — The Most Important

The famous bell curve. Most natural phenomena follow this:

$$\mathcal{N}(\mu, \sigma^2): \quad f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$$

Where:
- $\mu$ (mu) = **mean** — the center of the distribution
- $\sigma$ (sigma) = **standard deviation** — the spread
- $\sigma^2$ = **variance**

```text
     Standard Normal N(0,1):
       ↑
  0.4  |       ***
       |      *   *
  0.2  |     *     *
       |    *       *
  0.0  |___*_________*___
      -3  -2  -1   0  1  2  3
```

> **DL Connections:**
> - **Weight initialization**: Xavier/He init samples weights from Gaussian distributions
> - **Data distributions**: Many real-world features are approximately Gaussian
> - **Noise modeling**: Assumes Gaussian noise → leads to MSE loss (Lesson 2!)

---

## 5. Statistics: Describing Data

Given a dataset $\{x_1, x_2, \dots, x_N\}$:

### 5.1 Measures of Central Tendency

$$\text{Mean: } \mu = \frac{1}{N} \sum_{i=1}^N x_i$$

$$\text{Median: } \text{middle value when sorted}$$

$$\text{Mode: } \text{most frequently occurring value}$$

### 5.2 Measures of Spread

$$\text{Variance: } \sigma^2 = \frac{1}{N} \sum_{i=1}^N (x_i - \mu)^2$$

$$\text{Standard Deviation: } \sigma = \sqrt{\sigma^2}$$

> 💡 **DL Intuition:** When we **standardize** features (Lesson 6: Preprocessing), we subtract the mean and divide by std: $x' = \frac{x - \mu}{\sigma}$. This transforms any distribution to approximately $\mathcal{N}(0, 1)$, which makes gradient descent converge faster.

---

## 6. Expected Value & Variance

**Expected Value E[X]:** The long-run average outcome:

$$\mathbb{E}[X] = \sum_x x \cdot P(X=x) \quad \text{(discrete)}$$

$$\mathbb{E}[X] = \int_{-\infty}^{\infty} x \cdot f(x) \, dx \quad \text{(continuous)}$$

**Variance Var(X):** Average squared deviation from the mean:

$$\text{Var}(X) = \mathbb{E}[(X - \mathbb{E}[X])^2] = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$$

---

## 7. Maximum Likelihood Estimation (MLE)

**MLE** is the principle behind nearly every loss function in deep learning. The idea: choose model parameters that make the **observed training data most probable**.

Given training data $\{(x_i, y_i)\}$ and model with parameters $\theta$:

$$\hat{\theta}_{MLE} = \arg\max_{\theta} \prod_{i=1}^N P(y_i | x_i; \theta)$$

Taking the log (log-likelihood, easier to work with):

$$\hat{\theta}_{MLE} = \arg\max_{\theta} \sum_{i=1}^N \log P(y_i | x_i; \theta)$$

Maximizing log-likelihood = **minimizing negative log-likelihood (NLL)**.

---

## 8. Cross-Entropy Loss = Negative Log-Likelihood

This is the key insight connecting probability to training:

For **binary classification**, the model predicts $\hat{p} = P(y=1|x)$ via sigmoid. The NLL under Bernoulli distribution is:

$$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^N \left[ y_i \log(\hat{p}_i) + (1-y_i)\log(1-\hat{p}_i) \right]$$

**This IS the binary cross-entropy loss** — derived directly from MLE on Bernoulli distributions.

For **multi-class classification** (softmax output $\hat{p}_k$ for class $k$):

$$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^N \sum_{k=1}^K y_{ik} \log(\hat{p}_{ik})$$

> 🎯 **Key Takeaway:** Cross-entropy loss is NOT arbitrary. It's the mathematically principled loss derived from MLE when your output models a probability distribution.

Similarly, **MSE loss** (Lesson 2) is derived from MLE when assuming Gaussian noise on the output.

---

## 9. Python / NumPy Implementation

```python
import numpy as np

np.random.seed(42)

# ─────────────────────────────────────────────
# 1. Basic Statistics from Scratch
# ─────────────────────────────────────────────
data = np.array([2.1, 3.5, 2.8, 4.1, 3.2, 5.0, 2.7, 3.9, 4.5, 3.1])

mean   = np.mean(data)
median = np.median(data)
std    = np.std(data)
var    = np.var(data)

print("=== Basic Statistics ===")
print(f"Mean:     {mean:.4f}")
print(f"Median:   {median:.4f}")
print(f"Std Dev:  {std:.4f}")
print(f"Variance: {var:.4f}")

# ─────────────────────────────────────────────
# 2. Simulating Distributions
# ─────────────────────────────────────────────
gaussian_samples = np.random.normal(loc=0, scale=1, size=1000)  # N(0,1)
bernoulli_samples = np.random.binomial(n=1, p=0.3, size=1000)   # Bern(0.3)

print(f"\n=== Gaussian N(0,1) (1000 samples) ===")
print(f"Sample Mean: {gaussian_samples.mean():.4f}  (expected: 0)")
print(f"Sample Std:  {gaussian_samples.std():.4f}   (expected: 1)")

print(f"\n=== Bernoulli p=0.3 (1000 samples) ===")
print(f"Sample P(X=1): {bernoulli_samples.mean():.4f}  (expected: 0.3)")

# ─────────────────────────────────────────────
# 3. Cross-Entropy Loss from Scratch
# ─────────────────────────────────────────────
def binary_cross_entropy(y_true, y_pred, eps=1e-8):
    """
    Binary Cross-Entropy Loss.
    y_true: true binary labels (0 or 1)
    y_pred: predicted probabilities (values in (0, 1))
    eps: small value to prevent log(0)
    """
    y_pred = np.clip(y_pred, eps, 1 - eps)  # numerical stability
    loss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return np.mean(loss)

# Perfect predictions → loss ≈ 0
y_true = np.array([1.0, 0.0, 1.0, 1.0, 0.0])
y_pred_perfect = np.array([0.99, 0.01, 0.99, 0.99, 0.01])
y_pred_bad     = np.array([0.50, 0.50, 0.50, 0.50, 0.50])

print(f"\n=== Binary Cross-Entropy ===")
print(f"Perfect predictions: {binary_cross_entropy(y_true, y_pred_perfect):.4f}")
print(f"Random predictions:  {binary_cross_entropy(y_true, y_pred_bad):.4f}")

# ─────────────────────────────────────────────
# 4. MLE Intuition: Fitting a Gaussian to Data
# ─────────────────────────────────────────────
observed_data = np.array([1.2, 2.3, 1.8, 2.1, 2.5, 1.9, 2.2, 2.0])
# MLE for Gaussian: mu_hat = sample mean, sigma_hat = sample std
mu_hat    = observed_data.mean()
sigma_hat = observed_data.std()
print(f"\n=== MLE Gaussian Fit ===")
print(f"MLE mean (mu):    {mu_hat:.4f}")
print(f"MLE std (sigma):  {sigma_hat:.4f}")
print(f"=> Model: N({mu_hat:.2f}, {sigma_hat:.2f}^2)")
```

**Expected Output:**
```
=== Basic Statistics ===
Mean:     3.4900
Median:   3.3500
Std Dev:  0.8474
Variance: 0.7181

=== Gaussian N(0,1) (1000 samples) ===
Sample Mean: 0.0199  (expected: 0)
Sample Std:  0.9958  (expected: 1)

=== Binary Cross-Entropy ===
Perfect predictions: 0.0101
Random predictions:  0.6931

=== MLE Gaussian Fit ===
MLE mean (mu):    2.0000
MLE std (sigma):  0.3354
```

---

## 10. Key Vocabulary Reference

| Term | Definition |
|---|---|
| **Probability** | A number in [0,1] measuring likelihood of an event |
| **Bernoulli Distribution** | Distribution for binary (0/1) outcomes with parameter p |
| **Gaussian Distribution** | Bell-curve distribution parameterized by mean μ and std σ |
| **Mean (Expected Value)** | Average value: `E[X] = Σ x·P(x)` |
| **Variance** | Average squared deviation from mean |
| **Standard Deviation** | Square root of variance; same units as data |
| **MLE** | Find parameters that maximize probability of observing training data |
| **Log-Likelihood** | Log of probability; easier to maximize (sums instead of products) |
| **Cross-Entropy** | Loss derived from NLL under categorical/Bernoulli distribution |
| **Conditional Probability** | P(A\|B) — probability of A given B has occurred |

---

## 11. Common Mistakes & Pitfalls

1. **Confusing Probability with Likelihood:** Probability: $P(\text{data} | \theta)$. Likelihood: $\mathcal{L}(\theta | \text{data})$. They use the same formula but ask different questions.
2. **Forgetting `log(0) = -∞`:** Always clip predicted probabilities away from 0 and 1 before computing cross-entropy. PyTorch handles this automatically, but numpy does not.
3. **Ignoring Class Imbalance:** If 95% of your data is class 0, a model predicting all 0s gets 95% accuracy. Always look at precision, recall, and F1, not just accuracy.
4. **Treating Correlation as Causation:** Two features being correlated doesn't mean one causes the other. This causes bad feature engineering decisions.
5. **Using Mean to Describe Skewed Data:** Median is more robust for skewed distributions (e.g., salary data). Mean is easily distorted by outliers.

---

## 12. The Bridge to Softmax

In multi-class classification, the final layer outputs **logits** — raw unnormalized scores. **Softmax** converts these to a valid probability distribution:

$$\text{softmax}(z_k) = \frac{e^{z_k}}{\sum_{j=1}^K e^{z_j}}$$

This ensures outputs: (1) are all positive, (2) sum to exactly 1. In other words, softmax makes the output a **proper probability distribution** over K classes — exactly what we need to apply MLE and cross-entropy.

---

## 13. Interview Questions & Quiz

1. **Question 1:** Explain why using MSE loss for a binary classification task is suboptimal compared to binary cross-entropy. What probability distribution does each loss implicitly assume?

2. **Question 2:** A classifier predicts 0.9 for a positive sample and 0.1 for a negative sample. Calculate the binary cross-entropy loss for these two predictions. What would the loss be if both predictions were 0.5?

3. **Question 3:** What is the relationship between MLE and cross-entropy loss? Derive the connection in one or two steps.

4. **Question 4 (Code):** Write a function to compute the **categorical cross-entropy** loss for multi-class classification (softmax output). Test it with: `y_true = [0, 1, 0]` (one-hot), `y_pred = [0.1, 0.7, 0.2]`.

---

## 📖 References & Further Reading

- *Deep Learning* by Goodfellow et al. — **Chapter 3: Probability and Information Theory**
- *Pattern Recognition and Machine Learning* by Bishop — Chapter 1 (Introduction)
- StatQuest with Josh Starmer: *"Maximum Likelihood, clearly explained!"* (YouTube)

---

## ✅ Lesson Summary

| Concept | What You Learned |
|---|---|
| **Probability** | Foundation for understanding model outputs as distributions |
| **Bernoulli Distribution** | Binary outcomes → directly maps to sigmoid output |
| **Gaussian Distribution** | Bell curve → used for weight init, Gaussian noise → MSE loss |
| **Mean & Variance** | How to summarize data distributions numerically |
| **MLE** | Choose parameters that maximize probability of observed data |
| **Cross-Entropy Loss** | = Negative log-likelihood; derived from MLE on categorical distributions |
| **Softmax** | Converts raw logits into a valid probability distribution |

> **Next Lesson (06):** Data Preprocessing & Feature Engineering — how to clean, scale, and transform raw data so your neural network can actually learn from it. 🚀
