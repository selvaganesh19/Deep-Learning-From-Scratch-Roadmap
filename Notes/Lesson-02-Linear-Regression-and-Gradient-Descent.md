# Lesson 02: Linear Regression & Gradient Descent — How Neural Networks Learn

> **Prerequisites:** Lesson 01 — Introduction to Deep Learning & AI Intuition
> **Goal:** Understand the *learning process* of a neural network by mastering the simplest possible learning algorithm — **Linear Regression** — and the engine that powers it — **Gradient Descent**.

---

## 1. The Big Question: How Does a Network *Learn*?

In Lesson 1, you built a neuron with **manually chosen** weights (`[0.8, 0.5, -1.5]`) and a bias (`-0.1`). But in the real world, nobody hands us the correct weights. The network must **discover them on its own** by looking at data.

This process of automatically finding the best weights is called **training**, and the algorithm that drives it is called **Gradient Descent**.

The learning pipeline every neural network follows:

```text
                  ┌────────────────────────────────────────────────────────────┐
                  │                   TRAINING LOOP                            │
                  │                                                            │
  Training Data ──► Forward Pass ──► Prediction ──► Loss Function             │
       (x, y)    │       ↑                              │                      │
                  │       │            Backpropagation  ▼                      │
                  │  Update Weights  ◄──────────── Gradient Descent            │
                  └────────────────────────────────────────────────────────────┘
```

---

## 2. What Is Linear Regression?

**Linear Regression** is the simplest model that learns from data. It tries to fit a **straight line** through a set of data points.

**Real-World Example:**
You want to predict a student's final exam score based on the number of hours they studied.

| Hours Studied ($x$) | Exam Score ($y$) |
|---|---|
| 1 | 42 |
| 2 | 55 |
| 3 | 68 |
| 4 | 78 |
| 5 | 90 |

The model assumes the relationship is linear:

$$\hat{y} = w \cdot x + b$$

Where:
- $x$ — the input feature (hours studied)
- $\hat{y}$ — the **predicted** output (predicted exam score). The hat (`^`) means "predicted."
- $w$ — the **weight** (slope of the line). Controls how much each additional hour impacts the score.
- $b$ — the **bias** (y-intercept). Represents the baseline score even with 0 hours of study.

> 💡 **Key Insight:** This is the **exact same formula** as our single neuron from Lesson 1, but **without an activation function**. Linear regression *is* a single neuron in its simplest form.

---

## 3. The Loss Function: Measuring How Wrong We Are

Before the model can improve, it needs a way to **measure its mistake**. We use a **Loss Function** (also called a Cost Function) for this.

The most common loss function for regression is **Mean Squared Error (MSE)**:

$$\mathcal{L}(w, b) = \frac{1}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i)^2$$

Where:
- $N$ — the total number of training examples.
- $\hat{y}_i$ — the model's **prediction** for the $i$-th example.
- $y_i$ — the **true label** (actual exam score) for the $i$-th example.
- $(\hat{y}_i - y_i)^2$ — the squared error: we **square** it so negative and positive errors don't cancel out, and to heavily penalize large mistakes.

### Why Squared Error?

If a student scored 80 but we predicted 75, the error is $-5$. If another student scored 60 and we predicted 65, the error is $+5$. These cancel out to 0, incorrectly suggesting our model is perfect. Squaring ensures all errors contribute positively.

| True Score $y$ | Predicted Score $\hat{y}$ | Error $(\hat{y} - y)$ | Squared Error $(\hat{y}-y)^2$ |
|---|---|---|---|
| 42 | 44 | +2 | 4 |
| 55 | 52 | -3 | 9 |
| 68 | 71 | +3 | 9 |
| **MSE** | | | $\frac{4+9+9}{3} \approx 7.3$ |

Our **goal** during training is to **minimize this loss**.

---

## 4. Gradient Descent: The Mountain Analogy

Imagine you are blindfolded on a hilly mountain, and your goal is to reach the lowest valley. You can't see anything, but you *can* feel the slope of the ground beneath your feet.

Your strategy:
1. Feel which direction the ground slopes **downward**.
2. Take a **small step** in that direction.
3. Repeat until you can no longer go lower.

This is **exactly** what Gradient Descent does — but on the **loss landscape** instead of a physical mountain.

```text
 Loss (L)
    |
    |   *                           The loss landscape (bowl shape)
    |       *
    |           *
    |               *
    |                   *
    |                       *  <- Global Minimum (best weights)
    |
    └──────────────────────────── Weight (w)
        ←←←←←←←←←←←←←←←
         Steps of descent
```

### The Update Rule

At each training step, we calculate the **gradient** (the slope) of the loss with respect to each weight, then take a step *opposite* to the gradient (since we want to go *down*).

$$w \leftarrow w - \alpha \cdot \frac{\partial \mathcal{L}}{\partial w}$$

$$b \leftarrow b - \alpha \cdot \frac{\partial \mathcal{L}}{\partial b}$$

Where:
- $\alpha$ (alpha) — the **Learning Rate**. Controls how big each step is. This is a **hyperparameter** we choose (e.g., 0.01).
- $\frac{\partial \mathcal{L}}{\partial w}$ — the **gradient** of the loss with respect to weight $w$ (how steeply the loss changes if we nudge $w$).
- $\leftarrow$ — means "update the value with."

---

## 5. Deriving the Gradients (The Math)

For the MSE loss with our linear model $\hat{y} = wx + b$, we need to find $\frac{\partial \mathcal{L}}{\partial w}$ and $\frac{\partial \mathcal{L}}{\partial b}$.

$$\mathcal{L} = \frac{1}{N} \sum_{i=1}^{N} (wx_i + b - y_i)^2$$

**Gradient with respect to $w$** (using the chain rule):

$$\frac{\partial \mathcal{L}}{\partial w} = \frac{2}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i) \cdot x_i$$

**Gradient with respect to $b$**:

$$\frac{\partial \mathcal{L}}{\partial b} = \frac{2}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i)$$

> 🧠 **Intuition:**
> - If $\frac{\partial \mathcal{L}}{\partial w} > 0$: increasing $w$ increases the loss, so we should **decrease** $w$.
> - If $\frac{\partial \mathcal{L}}{\partial w} < 0$: increasing $w$ decreases the loss, so we should **increase** $w$.
>
> The minus sign in the update rule handles this automatically.

---

## 6. The Learning Rate: Choosing Your Step Size

The learning rate $\alpha$ is one of the most important hyperparameters to tune:

```text
Too Small (α = 0.0001):             Too Large (α = 10):             Just Right (α = 0.01):
    Loss                                 Loss                              Loss
    |                                    |  *                              |
    | * * *                              |      *                          |  *
    |       * * *                        |    *   *  (diverges!)           |      *
    |             * * *                  |  *         *                    |          *
    |                   ...             |               ...               |              * *
    └──── Epochs                         └──── Epochs                      └──── Epochs
    (Learns, but painfully slow)         (Overshoots, never converges)    (Clean convergence!)
```

---

## 7. Python Implementation: From Scratch

Let's now implement everything from scratch using only NumPy.

```python
import numpy as np
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# Step 1: Create Our Training Data
# ─────────────────────────────────────────────
# True relationship: score = 15 * hours + 30
# We add a bit of noise to make it realistic.
np.random.seed(42)
hours_studied = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
true_scores   = 15 * hours_studied + 30 + np.random.randn(8) * 5  # noise

print("Training Data:")
for h, s in zip(hours_studied, true_scores):
    print(f"  Hours: {h:.1f}  |  Score: {s:.2f}")

# ─────────────────────────────────────────────
# Step 2: Initialize Weights Randomly
# ─────────────────────────────────────────────
w = np.random.randn()   # random weight (slope)
b = np.random.randn()   # random bias  (intercept)
print(f"\nInitial w = {w:.4f}, b = {b:.4f}")

# ─────────────────────────────────────────────
# Step 3: Set Hyperparameters
# ─────────────────────────────────────────────
learning_rate = 0.01
epochs        = 1000   # number of full passes through training data
N             = len(hours_studied)
loss_history  = []

# ─────────────────────────────────────────────
# Step 4: The Training Loop (Gradient Descent)
# ─────────────────────────────────────────────
for epoch in range(epochs):

    # --- FORWARD PASS: make a prediction ---
    y_pred = w * hours_studied + b          # y_hat = wx + b

    # --- COMPUTE LOSS (MSE) ---
    errors = y_pred - true_scores           # (y_hat - y)
    loss   = np.mean(errors ** 2)           # MSE
    loss_history.append(loss)

    # --- COMPUTE GRADIENTS ---
    dL_dw = (2 / N) * np.sum(errors * hours_studied)   # dL/dw
    dL_db = (2 / N) * np.sum(errors)                   # dL/db

    # --- UPDATE WEIGHTS (Gradient Descent step) ---
    w -= learning_rate * dL_dw
    b -= learning_rate * dL_db

    # --- LOG EVERY 100 EPOCHS ---
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1:4d} | Loss: {loss:.4f} | w: {w:.4f} | b: {b:.4f}")

# ─────────────────────────────────────────────
# Step 5: Evaluate the Trained Model
# ─────────────────────────────────────────────
print(f"\n Training Complete!")
print(f"   Learned w = {w:.4f}  (True w = 15.0000)")
print(f"   Learned b = {b:.4f}  (True b = 30.0000)")

# Predict for a new student who studied 9 hours
new_hours = 9.0
predicted_score = w * new_hours + b
print(f"\n Prediction: A student who studies {new_hours} hours will score approx {predicted_score:.2f}")
```

**Expected Output (approximate):**
```
Epoch  100 | Loss: 42.3821 | w: 13.8921 | b: 31.2345
Epoch  200 | Loss: 28.1142 | w: 14.3210 | b: 30.8901
...
Epoch 1000 | Loss: 24.8712 | w: 14.9831 | b: 30.1204

Training Complete!
   Learned w = 14.9831  (True w = 15.0000)
   Learned b = 30.1204  (True b = 30.0000)

Prediction: A student who studies 9.0 hours will score approx 165.17
```

> 🎯 The learned weights are very close to the true values (`w≈15`, `b≈30`). The small difference is because of the random noise we added to the training data — the model cannot "unlearn" noise.

---

## 8. Visualizing the Learning Process

Add this block after training to plot the loss curve:

```python
plt.figure(figsize=(12, 5))

# Plot 1: Loss over Epochs
plt.subplot(1, 2, 1)
plt.plot(loss_history, color='crimson', linewidth=2)
plt.title('Training Loss (MSE) Over Epochs', fontsize=14)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True, alpha=0.3)

# Plot 2: Fitted Line vs. Actual Data
plt.subplot(1, 2, 2)
x_line = np.linspace(0, 9, 100)
y_line = w * x_line + b
plt.scatter(hours_studied, true_scores, color='royalblue', s=80, zorder=5, label='Training Data')
plt.plot(x_line, y_line, color='crimson', linewidth=2, label=f'Fitted Line: y = {w:.2f}x + {b:.2f}')
plt.title('Linear Regression Fit', fontsize=14)
plt.xlabel('Hours Studied')
plt.ylabel('Exam Score')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('lesson_02_output.png', dpi=150)
plt.show()
```

---

## 9. Variants of Gradient Descent

As datasets grow larger, the standard Gradient Descent (which processes **all** training examples per step, also called **Batch Gradient Descent**) becomes slow. Here are the common variants:

| Variant | Updates Using | Speed | Noise | Best For |
|---|---|---|---|---|
| **Batch GD** | All $N$ examples | Slow | Low (smooth) | Small datasets |
| **Stochastic GD (SGD)** | 1 example at a time | Fast | High (noisy) | Online learning |
| **Mini-Batch GD** | $B$ examples (e.g., 32) | Balanced | Medium | Most deep learning (standard choice) |

> 💡 **In practice:** Almost all modern deep learning uses **Mini-Batch Gradient Descent** (often just called "SGD" loosely). Libraries like PyTorch and TensorFlow implement this by default.

---

## 10. Key Vocabulary Reference

| Term | Definition |
|---|---|
| **Model / Hypothesis** | The mathematical function that maps input $x$ to output $\hat{y}$ |
| **Parameters** | The values that the model *learns* — i.e., weights $w$ and bias $b$ |
| **Hyperparameters** | Values *we* choose before training (e.g., learning rate, epochs) |
| **Loss Function** | Measures how wrong the model's predictions are |
| **Gradient** | The slope of the loss surface — tells us which direction is "uphill" |
| **Gradient Descent** | Optimization algorithm that walks "downhill" on the loss surface |
| **Learning Rate (α)** | Controls the size of each gradient descent step |
| **Epoch** | One full pass through the entire training dataset |
| **Forward Pass** | Computing the prediction y_hat from inputs x |
| **Convergence** | When the loss stops decreasing significantly — training is done |
| **Overfitting** | When a model memorizes training data but fails on new data |

---

## 11. Common Mistakes & Pitfalls

1. **Learning Rate Too Large:** The loss will bounce around or even explode to infinity. Always start small (e.g., `0.01`) and adjust.
2. **Not Normalizing Input Features:** If features have very different scales (e.g., `house_size` in thousands vs. `num_rooms` in single digits), gradients can be very uneven, making training slow. Always normalize (scale to 0–1 or standardize to mean=0, std=1).
3. **Training for Too Few Epochs:** The model hasn't converged yet. Plot the loss curve — if it's still declining, train longer.
4. **Training for Too Many Epochs on a Small Dataset:** The model will start to **overfit** (memorize noise). We'll address this with regularization in a later lesson.
5. **Forgetting to Update *Both* $w$ and $b$:** Both parameters need to be updated at every step.

---

## 12. The Bridge to Neural Networks

You might ask: "We only have one weight here. Neural networks have *millions*. How does this scale?"

The answer is beautiful: **the principle is identical**.

- Each neuron's weight gets its own gradient.
- All gradients are computed simultaneously using a technique called **Backpropagation** (Lesson 4).
- All weights are updated using the exact same Gradient Descent rule.

```text
Linear Regression          →     Neural Network
─────────────────────────────────────────────────────
One weight (w)             →     Millions of weights (W1, W2, ... Wn)
One gradient (dL/dw)       →     One gradient per weight (dL/dWi)
One update step            →     Millions of simultaneous update steps
```

The engine driving training in a billion-parameter GPT model is the **same gradient descent** you just implemented above — just applied at a massive scale.

---

## 13. Interview Questions & Quiz

### Test Your Understanding

1. **Question 1:** What does the loss function measure, and why do we square the errors in MSE instead of just taking the absolute value?

2. **Question 2:** If the gradient $\frac{\partial \mathcal{L}}{\partial w}$ is positive at the current value of $w$, should we increase or decrease $w$ in the next update? Explain why using the gradient descent update rule.

3. **Question 3 (Code Challenge):** In the Python implementation, what would happen to training if you set `learning_rate = 10.0`? Try it and explain what you observe in the loss values.

4. **Question 4 (Math Challenge):** Given:
   - $x = [2, 4]$
   - $y = [50, 80]$ (true scores)
   - $w = 5.0$, $b = 10.0$

   Calculate:
   - The predictions $\hat{y}$
   - The MSE loss $\mathcal{L}$
   - The gradient $\frac{\partial \mathcal{L}}{\partial w}$
   - The updated weight $w_{new}$ with $\alpha = 0.01$

---

## 📖 References & Further Reading

- *Deep Learning* by Ian Goodfellow et al. — Chapter 4 (Numerical Computation) & Chapter 5 (Machine Learning Basics).
- 3Blue1Brown: *"Gradient descent, how neural networks learn"* (YouTube — Neural Networks series, Video 2).
- Andrej Karpathy: *micrograd* — A tiny Autograd engine that implements gradient descent for neural networks from scratch (GitHub).

---

## ✅ Lesson Summary

| Concept | What You Learned |
|---|---|
| **Linear Regression** | The simplest learning model — fit a line y_hat = wx + b to data |
| **Loss (MSE)** | How to measure prediction error: L = (1/N) * sum(y_hat - y)^2 |
| **Gradient Descent** | Iteratively update weights in the direction that reduces loss |
| **Update Rule** | w = w - α * (dL/dw) |
| **Learning Rate** | A hyperparameter controlling the step size; too large diverges, too small is slow |
| **Training Loop** | Forward pass → Compute Loss → Compute Gradients → Update Weights → Repeat |

> **Next Lesson (03):** We'll extend this to multiple features using **Vectorization & Matrix Operations with NumPy**, and understand how this scales to real-world datasets with hundreds of input columns. 🚀
