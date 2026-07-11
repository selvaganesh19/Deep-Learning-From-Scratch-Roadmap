# Lesson 04: Calculus Intuition — Derivatives & the Chain Rule

> **Prerequisites:** Lessons 01–03
> **Goal:** Understand derivatives and the chain rule — the mathematical engine that powers how every neural network learns through backpropagation.

---

## 1. Why Calculus Powers Deep Learning

In Lesson 2, we used the gradient $\frac{\partial \mathcal{L}}{\partial w}$ to update weights. But where does that gradient come from? The answer is **calculus** — specifically, **derivatives** and the **chain rule**.

Every time a neural network trains, it computes millions of derivatives simultaneously. Understanding this process is what separates someone who *uses* deep learning from someone who truly *understands* it.

```text
Loss Function ℒ(w)
       │
       │  "How does the loss change if I slightly change weight w?"
       │
       ▼
  Derivative dℒ/dw  ──►  Gradient Descent Update  ──►  Better Weights
```

---

## 2. What Is a Derivative?

A derivative measures the **instantaneous rate of change** of a function — how much the output changes when the input changes by a tiny amount.

**Real-World Analogy:** Your car's speedometer shows your derivative of position with respect to time. If your position function is $p(t)$, your speed is $\frac{dp}{dt}$.

**Geometrically:** The derivative at a point is the **slope of the tangent line** to the curve at that point.

```text
f(x)
  |          /
  |         / ← slope = f'(x) = derivative at this point
  |        /
  |   ____/
  |__/
  └──────────── x
```

### Formal Definition

$$f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$$

We nudge $x$ by a tiny $h$, measure how much $f$ changes, and divide by $h$. As $h \to 0$, this gives the exact slope.

---

## 3. Derivative Notation

You'll see these used interchangeably:

| Notation | Meaning |
|---|---|
| $f'(x)$ | Derivative of $f$ with respect to $x$ (Lagrange) |
| $\frac{dy}{dx}$ | Derivative of $y$ with respect to $x$ (Leibniz) |
| $\dot{y}$ | Derivative with respect to time (Newton) |
| $\frac{\partial f}{\partial x}$ | **Partial** derivative — treat all other variables as constants |

---

## 4. Basic Differentiation Rules

| Rule | Formula | Example |
|---|---|---|
| **Constant** | $\frac{d}{dx}[c] = 0$ | $\frac{d}{dx}[5] = 0$ |
| **Power Rule** | $\frac{d}{dx}[x^n] = nx^{n-1}$ | $\frac{d}{dx}[x^3] = 3x^2$ |
| **Sum Rule** | $\frac{d}{dx}[f + g] = f' + g'$ | $\frac{d}{dx}[x^2 + x] = 2x + 1$ |
| **Product Rule** | $\frac{d}{dx}[fg] = f'g + fg'$ | $\frac{d}{dx}[x^2 \cdot x] = 2x \cdot x + x^2 = 3x^2$ |
| **Chain Rule** | $\frac{d}{dx}[f(g(x))] = f'(g(x)) \cdot g'(x)$ | *(see Section 6)* |

**DL-Relevant Derivatives:**

| Function | Derivative | Used Where |
|---|---|---|
| $f(x) = x^2$ | $f'(x) = 2x$ | MSE Loss |
| $f(x) = e^x$ | $f'(x) = e^x$ | Softmax |
| $f(x) = \ln(x)$ | $f'(x) = \frac{1}{x}$ | Cross-Entropy Loss |
| $\sigma(x) = \frac{1}{1+e^{-x}}$ | $\sigma'(x) = \sigma(x)(1-\sigma(x))$ | Sigmoid activation |
| $\text{ReLU}(x) = \max(0, x)$ | $\text{ReLU}'(x) = \begin{cases}1 & x > 0 \\ 0 & x \leq 0\end{cases}$ | Most hidden layers |

---

## 5. Partial Derivatives — Multi-Variable Functions

Neural networks have millions of weights. A loss function depends on **all of them simultaneously**:

$$\mathcal{L}(w_1, w_2, b) = (w_1 x_1 + w_2 x_2 + b - y)^2$$

A **partial derivative** $\frac{\partial \mathcal{L}}{\partial w_1}$ asks: *"How does the loss change if I only nudge $w_1$, holding everything else fixed?"*

**Example:** For $f(x, y) = x^2 + 3xy + y^2$:

$$\frac{\partial f}{\partial x} = 2x + 3y \quad \text{(treat } y \text{ as a constant)}$$

$$\frac{\partial f}{\partial y} = 3x + 2y \quad \text{(treat } x \text{ as a constant)}$$

The **gradient** is just the vector of all partial derivatives:

$$\nabla f = \begin{bmatrix} \frac{\partial f}{\partial x} \\ \frac{\partial f}{\partial y} \end{bmatrix} = \begin{bmatrix} 2x + 3y \\ 3x + 2y \end{bmatrix}$$

---

## 6. The Chain Rule — The Heart of Backpropagation

The chain rule tells us how to differentiate **composed functions** (functions of functions). This is absolutely critical because neural networks are deeply composed functions.

### Simple Form
If $y = f(u)$ and $u = g(x)$, then:

$$\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}$$

> 🧠 **Think of it as:** "The rate of change of $y$ w.r.t. $x$ = (rate of change of $y$ w.r.t. $u$) × (rate of change of $u$ w.r.t. $x$)"

**Analogy:** Gears. If Gear A turns Gear B, and Gear B turns Gear C — how fast does Gear C spin relative to Gear A? You multiply the gear ratios. That's the chain rule.

### Worked Example

Let $y = (3x + 2)^4$. We can decompose this as:
- $u = 3x + 2$ → $\frac{du}{dx} = 3$
- $y = u^4$ → $\frac{dy}{du} = 4u^3$

By chain rule:
$$\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx} = 4u^3 \cdot 3 = 12(3x+2)^3$$

---

## 7. Computational Graphs & The Chain Rule

A **computational graph** shows the chain of operations. The chain rule flows **backwards** through this graph — that's exactly what **backpropagation** is.

**Example: One Neuron Forward Pass**

Given $x = 2$, $w = 3$, $b = 1$, and loss $\mathcal{L} = (wx + b)^2$:

```text
FORWARD PASS (left → right):
    x=2      w=3
      \       /
       \     /
        [× mul] ──► u = wx = 6
                              \
              b=1 ──────────── [+ add] ──► z = 7
                                              \
                                        [square] ──► L = 49
```

```text
BACKWARD PASS — Chain Rule (right → left):
    dL/dL = 1

    dL/dz = 2z = 14

    dL/du = dL/dz · dz/du = 14 · 1 = 14

    dL/dw = dL/du · du/dw = 14 · x = 14 · 2 = 28

    dL/db = dL/dz · dz/db = 14 · 1 = 14
```

So: gradient of loss w.r.t. weight $w$ = **28**. We'd update: $w \leftarrow w - \alpha \cdot 28$.

This backward flow through the computational graph is **backpropagation** — it's the chain rule applied repeatedly from output back to input.

---

## 8. Python Implementation: Numerical Differentiation

We can numerically approximate derivatives using the **finite difference** method:

$$f'(x) \approx \frac{f(x + h) - f(x - h)}{2h} \quad \text{(central difference)}$$

```python
import numpy as np

# ─────────────────────────────────────────────
# Numerical Gradient (Finite Difference)
# ─────────────────────────────────────────────
def numerical_gradient(f, x, h=1e-5):
    """
    Computes the numerical gradient of function f at point x.
    Uses central difference for better accuracy.
    """
    return (f(x + h) - f(x - h)) / (2 * h)

# Example 1: f(x) = x^2  →  f'(x) = 2x
f1 = lambda x: x**2
x = 3.0
print(f"f(x) = x^2 at x=3:")
print(f"  Numerical gradient: {numerical_gradient(f1, x):.6f}")  # ≈ 6.0
print(f"  Analytical (2x):    {2 * x:.6f}")                      # = 6.0

# Example 2: f(x) = sigmoid(x)  →  f'(x) = σ(x)(1 - σ(x))
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

x = 0.5
print(f"\nSigmoid at x=0.5:")
print(f"  Numerical gradient: {numerical_gradient(sigmoid, x):.6f}")
print(f"  Analytical σ(1-σ):  {sigmoid_derivative(x):.6f}")

# ─────────────────────────────────────────────
# Chain Rule in Code: Forward & Backward Pass
# ─────────────────────────────────────────────
print("\n--- Chain Rule Backprop Demo ---")
x, w, b = 2.0, 3.0, 1.0

# Forward pass
u = w * x          # u = wx
z = u + b          # z = wx + b
L = z ** 2         # L = (wx + b)^2
print(f"Forward:  u={u}, z={z}, L={L}")

# Backward pass (chain rule)
dL_dz = 2 * z         # derivative of L=z^2 is 2z
dL_du = dL_dz * 1     # z = u + b, so dz/du = 1
dL_dw = dL_du * x     # u = w*x, so du/dw = x
dL_db = dL_dz * 1     # z = u + b, so dz/db = 1
print(f"Backward: dL/dw={dL_dw}, dL/db={dL_db}")

# Verify with numerical gradient
f_w = lambda w_: (w_ * x + b) ** 2
f_b = lambda b_: (w * x + b_) ** 2
print(f"Numerical dL/dw: {numerical_gradient(f_w, w):.4f}")   # should ≈ dL_dw
print(f"Numerical dL/db: {numerical_gradient(f_b, b):.4f}")   # should ≈ dL_db
```

**Output:**
```
f(x) = x^2 at x=3:
  Numerical gradient: 6.000000
  Analytical (2x):    6.000000

Sigmoid at x=0.5:
  Numerical gradient: 0.235004
  Analytical σ(1-σ):  0.235004

--- Chain Rule Backprop Demo ---
Forward:  u=6.0, z=7.0, L=49.0
Backward: dL/dw=28.0, dL/db=14.0
Numerical dL/dw: 28.0000
Numerical dL/db: 14.0000
```

---

## 9. Key Vocabulary Reference

| Term | Definition |
|---|---|
| **Derivative** | Rate of change of a function — slope of the tangent line |
| **Partial Derivative** | Derivative w.r.t. one variable, holding others fixed |
| **Gradient** | Vector of all partial derivatives — points in direction of steepest ascent |
| **Chain Rule** | Rule for differentiating composed functions: `d(f∘g)/dx = f'(g(x))·g'(x)` |
| **Computational Graph** | Diagram of operations in a neural network; used to organize backprop |
| **Backpropagation** | Algorithm applying chain rule backwards through a neural network |
| **Numerical Gradient** | Approximating derivative using finite differences |
| **Analytical Gradient** | Exact derivative derived mathematically |
| **Jacobian** | Matrix of all partial derivatives for a vector-valued function |

---

## 10. Common Mistakes & Pitfalls

1. **Forgetting the Chain Rule:** When differentiating $f(g(x))$, you MUST multiply by $g'(x)$. The most common calculus mistake in backprop derivations.
2. **Treating $\partial$ as a Fraction Carelessly:** $\frac{\partial z}{\partial x}$ is NOT a fraction you can algebraically cancel — it's a notation for a limit. (You can use it *loosely* in the chain rule but understand what it represents.)
3. **Confusing Gradient Direction:** The gradient points **uphill** (toward increasing loss). Gradient *descent* subtracts the gradient to go downhill.
4. **Not Checking with Numerical Gradients:** When implementing backprop from scratch, always verify your analytical gradients match numerical finite differences.

---

## 11. The Bridge to Backpropagation

Everything in this lesson points to one thing: **Backpropagation (Lesson 14)** is just the chain rule applied systematically from the loss function back through every layer.

```text
Layer 1 → Layer 2 → Layer 3 → Loss ℒ
   ↑           ↑          ↑        ↑
   │           │          │        │
dℒ/dW1 ←── dℒ/dW2 ←── dℒ/dW3 ← dℒ/dL=1
         (Chain Rule applied at each arrow)
```

The chain rule connects every weight's gradient to the final loss — making it possible to train networks with billions of parameters.

---

## 12. Interview Questions & Quiz

1. **Question 1:** What is the derivative of $f(x) = 3x^4 - 2x^2 + 7x - 1$? Use the power rule and sum rule.

2. **Question 2:** Using the chain rule, find $\frac{dy}{dx}$ for $y = \sin(x^2 + 1)$.

3. **Question 3 (Hand Calculation):** Given the composition:
   - $u = 2x + 3$ (linear)
   - $v = u^3$ (cubic)
   - $L = v^2$ (squared)

   With $x = 1$, compute the forward pass values $u$, $v$, $L$, then use the chain rule to find $\frac{dL}{dx}$. Verify with numerical differentiation in Python.

4. **Question 4:** Why is the derivative of ReLU undefined at exactly $x=0$? How does PyTorch handle this in practice?

---

## 📖 References & Further Reading

- *Deep Learning* by Goodfellow et al. — **Chapter 4: Numerical Computation**, Chapter 6.5 (Backpropagation)
- 3Blue1Brown: *"What is backpropagation really doing?"* — YouTube Neural Networks series, Video 3
- Khan Academy: Multivariable Calculus — Partial Derivatives & Chain Rule (free)

---

## ✅ Lesson Summary

| Concept | What You Learned |
|---|---|
| **Derivative** | Instantaneous rate of change; slope of tangent line at a point |
| **Power Rule** | `d/dx[xⁿ] = nxⁿ⁻¹` — the most-used rule in DL |
| **Partial Derivative** | Derivative for multi-variable functions; gradient descent uses these |
| **Gradient** | Vector of partials: points toward steepest increase in loss |
| **Chain Rule** | `dy/dx = (dy/du)(du/dx)` — how gradients flow backwards through layers |
| **Computational Graph** | Visual tool for organizing forward + backward pass calculations |
| **Numerical Gradient** | Finite difference check to verify analytical gradients |

> **Next Lesson (05):** Probability & Statistics for Deep Learning — we'll understand why neural network outputs are treated as probability distributions and how this leads to cross-entropy loss. 🚀
