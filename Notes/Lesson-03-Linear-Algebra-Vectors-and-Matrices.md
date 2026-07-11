# Lesson 03: Linear Algebra Intuition — Vectors & Matrices

> **Prerequisites:** Lesson 01 & 02
> **Goal:** Understand the mathematical language of Deep Learning. Every neural network forward pass is matrix multiplication — this lesson makes that crystal clear.

---

## 1. Why Linear Algebra Matters in Deep Learning

When you pass an image into a neural network, it isn't processed pixel-by-pixel one at a time. Thousands of multiplications happen **simultaneously** using matrix operations. Linear algebra is the language that makes this possible.

```text
Image (28×28 pixels) ──► Flatten to Vector (784,) ──► Matrix Multiply ──► Hidden Layer Output
                                                       [W] (256×784)
```

Every layer in a neural network is just:

$$\mathbf{y} = \mathbf{W} \mathbf{x} + \mathbf{b}$$

This single equation — a **matrix-vector multiplication** plus a **bias addition** — IS the neural network layer. Understanding it fully is the goal of this lesson.

---

## 2. The Four Data Structures: Scalars → Tensors

Think of these as nested containers of numbers:

```text
┌──────────────────────────────────────────────────────────────────┐
│  SCALAR    VECTOR        MATRIX           TENSOR (3D)            │
│                                                                  │
│    5       [1, 2, 3]    [1  2  3]        [[[1,2],[3,4]],        │
│                          [4  5  6]         [[5,6],[7,8]]]        │
│                          [7  8  9]                               │
│                                                                  │
│  shape:()  shape:(3,)   shape:(3,3)      shape:(2,2,2)          │
└──────────────────────────────────────────────────────────────────┘
```

| Name | Dimensions | Example in DL | NumPy Shape |
|---|---|---|---|
| **Scalar** | 0D | Learning rate α = 0.01 | `()` |
| **Vector** | 1D | One training sample's features | `(n,)` |
| **Matrix** | 2D | Weight matrix of a layer | `(rows, cols)` |
| **Tensor** | 3D+ | Batch of images (batch, height, width) | `(N, H, W)` |

---

## 3. Vector Operations

A **vector** is an ordered list of numbers representing a point or direction in space.

$$\mathbf{v} = \begin{bmatrix} v_1 \\ v_2 \\ v_3 \end{bmatrix}$$

### 3.1 Vector Addition
Add element-wise — vectors must have the same length:

$$\mathbf{a} + \mathbf{b} = \begin{bmatrix} a_1 + b_1 \\ a_2 + b_2 \end{bmatrix}$$

### 3.2 Scalar Multiplication
Stretch/shrink every element:

$$k \cdot \mathbf{v} = \begin{bmatrix} k \cdot v_1 \\ k \cdot v_2 \end{bmatrix}$$

### 3.3 Dot Product (Most Important!)
The dot product measures how **aligned** two vectors are:

$$\mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^{n} a_i b_i = a_1 b_1 + a_2 b_2 + \cdots + a_n b_n$$

> 💡 **DL Intuition:** A single neuron's output `z = w·x + b` is just a **dot product** of weights `w` and inputs `x`, plus bias. This is the atomic operation of every neural network.

- If dot product = 0 → vectors are **perpendicular** (orthogonal)
- If dot product > 0 → vectors point in a **similar direction**
- If dot product < 0 → vectors point in **opposite directions**

### 3.4 Magnitude (L2 Norm)
The "length" of a vector:

$$||\mathbf{v}||_2 = \sqrt{v_1^2 + v_2^2 + \cdots + v_n^2}$$

Used in **weight regularization** (penalizing large weights to prevent overfitting).

---

## 4. Matrix Operations

A **matrix** is a 2D array of numbers with shape `(rows, columns)`.

$$A = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix} \quad \text{shape: } (2 \times 3)$$

### 4.1 Matrix Addition
Element-wise — matrices must have **identical shapes**:

$$A + B = \begin{bmatrix} a_{11}+b_{11} & a_{12}+b_{12} \\ a_{21}+b_{21} & a_{22}+b_{22} \end{bmatrix}$$

### 4.2 Matrix Multiplication (The Core of Neural Networks)

For matrices $A$ (shape $m \times n$) and $B$ (shape $n \times p$):

$$C = AB \quad \text{shape: } (m \times p)$$

$$C_{ij} = \sum_{k=1}^{n} A_{ik} \cdot B_{kj}$$

> 🔑 **Shape Rule:** The **inner dimensions must match**. `(m × n) @ (n × p) = (m × p)`

```text
    A           B           C
(3 × 2)  @  (2 × 4)  =  (3 × 4)
         ↑↑
   Must match!
```

**Example:**

$$\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix} = \begin{bmatrix} 1\cdot5+2\cdot7 & 1\cdot6+2\cdot8 \\ 3\cdot5+4\cdot7 & 3\cdot6+4\cdot8 \end{bmatrix} = \begin{bmatrix} 19 & 22 \\ 43 & 50 \end{bmatrix}$$

### 4.3 Transpose
Flip rows and columns — shape `(m × n)` becomes `(n × m)`:

$$A^T_{ij} = A_{ji}$$

$$\begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix}^T = \begin{bmatrix} 1 & 4 \\ 2 & 5 \\ 3 & 6 \end{bmatrix}$$

> 💡 **DL Use:** Transposing weight matrices is essential during backpropagation to propagate gradients backwards through layers.

### 4.4 Identity Matrix
The "number 1" for matrices — any matrix times the identity equals itself:

$$I = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix} \quad A \cdot I = I \cdot A = A$$

---

## 5. Element-wise vs Matrix Multiplication

This is a very common source of confusion:

| Operation | Symbol | Description | NumPy |
|---|---|---|---|
| **Element-wise multiply** | $A \odot B$ or $A * B$ | Each element multiplied by corresponding element | `A * B` |
| **Matrix multiply (dot)** | $AB$ or $A @ B$ | Row × Column dot products | `A @ B` or `np.dot(A,B)` |

```text
Element-wise (*):         Matrix multiply (@):
[1 2]   [5 6]            [1 2]   [5 6]
[3 4] * [7 8]            [3 4] @ [7 8]

= [1×5  2×6]             = [1×5+2×7  1×6+2×8]
  [3×7  4×8]               [3×5+4×7  3×6+4×8]

= [5  12]                = [19  22]
  [21 32]                  [43  50]
```

---

## 6. Neural Network Layer = Matrix Operation

A fully-connected (dense) neural network layer with:
- Input: $\mathbf{x}$ — shape `(n,)` — n features
- Weights: $\mathbf{W}$ — shape `(m, n)` — m neurons, n inputs each
- Bias: $\mathbf{b}$ — shape `(m,)` — one bias per neuron

Computes:

$$\mathbf{z} = \mathbf{W} \mathbf{x} + \mathbf{b} \quad \text{shape: } (m,)$$

```text
Input x          Weight Matrix W          Output z
(784,)    ──►      (256, 784)      ──►    (256,)
                 + Bias b (256,)
```

For a **batch** of N samples:

$$Z = X W^T + \mathbf{b} \quad \text{shape: } (N, m)$$

This single line of matrix math replaces a double nested for-loop and runs thousands of times faster on a GPU.

---

## 7. Python / NumPy Implementation

```python
import numpy as np

# ─────────────────────────────────────────────
# 1. Creating Scalars, Vectors, Matrices
# ─────────────────────────────────────────────
scalar = 5.0
vector = np.array([1.0, 2.0, 3.0])           # shape: (3,)
matrix = np.array([[1, 2, 3],
                   [4, 5, 6]])                 # shape: (2, 3)
tensor = np.zeros((4, 28, 28))               # shape: (4, 28, 28) — batch of 4 images

print(f"Scalar: {scalar}")
print(f"Vector shape: {vector.shape}")        # (3,)
print(f"Matrix shape: {matrix.shape}")        # (2, 3)
print(f"Tensor shape: {tensor.shape}")        # (4, 28, 28)

# ─────────────────────────────────────────────
# 2. Vector Operations
# ─────────────────────────────────────────────
a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, 5.0, 6.0])

print("\n--- Vector Operations ---")
print(f"Addition:          {a + b}")              # [5. 7. 9.]
print(f"Scalar multiply:   {2 * a}")              # [2. 4. 6.]
print(f"Dot product:       {np.dot(a, b)}")       # 32.0
print(f"Magnitude of a:    {np.linalg.norm(a):.4f}")  # 3.7417

# ─────────────────────────────────────────────
# 3. Matrix Operations
# ─────────────────────────────────────────────
A = np.array([[1, 2],
              [3, 4]])
B = np.array([[5, 6],
              [7, 8]])

print("\n--- Matrix Operations ---")
print(f"A + B:\n{A + B}")
print(f"A @ B (matrix multiply):\n{A @ B}")       # [[19 22][43 50]]
print(f"A.T (transpose):\n{A.T}")                  # [[1 3][2 4]]
print(f"Element-wise A * B:\n{A * B}")             # [[ 5 12][21 32]]

# ─────────────────────────────────────────────
# 4. Neural Network Layer from Scratch
# ─────────────────────────────────────────────
np.random.seed(42)
N = 5          # batch size
n_inputs = 4   # input features
n_neurons = 3  # neurons in this layer

X = np.random.randn(N, n_inputs)      # input batch:  shape (5, 4)
W = np.random.randn(n_neurons, n_inputs)  # weights:  shape (3, 4)
b = np.zeros(n_neurons)               # biases:       shape (3,)

# Forward pass: Z = X @ W.T + b
Z = X @ W.T + b                       # output:       shape (5, 3)
print(f"\n--- Neural Network Layer ---")
print(f"Input X shape:    {X.shape}")
print(f"Weight W shape:   {W.shape}")
print(f"Output Z shape:   {Z.shape}")
print(f"Z (first 2 rows):\n{Z[:2].round(4)}")
```

**Output:**
```
Vector shape: (3,)
Matrix shape: (2, 3)
Tensor shape: (4, 28, 28)

--- Vector Operations ---
Addition:          [5. 7. 9.]
Dot product:       32.0
Magnitude of a:    3.7417

--- Matrix Operations ---
A @ B (matrix multiply):
[[19 22]
 [43 50]]

--- Neural Network Layer ---
Input X shape:    (5, 4)
Weight W shape:   (3, 4)
Output Z shape:   (5, 3)
```

---

## 8. Key Vocabulary Reference

| Term | Definition |
|---|---|
| **Scalar** | A single number with no direction |
| **Vector** | An ordered list of numbers (1D array) |
| **Matrix** | A 2D grid of numbers (rows × columns) |
| **Tensor** | A general n-dimensional array |
| **Dot Product** | Sum of element-wise products; measures vector alignment |
| **Matrix Multiply** | Row-column dot products; the core of neural network layers |
| **Transpose** | Flip rows and columns of a matrix |
| **Shape** | The dimensions of an array, e.g. `(3, 4)` means 3 rows, 4 cols |
| **Broadcasting** | NumPy rule for operating on arrays of different shapes |
| **Identity Matrix** | Square matrix with 1s on diagonal; neutral element for multiplication |
| **L2 Norm** | Euclidean length of a vector: `sqrt(sum of squares)` |

---

## 9. Common Mistakes & Pitfalls

1. **Shape Mismatch:** The most common error in DL code. For `A @ B`, columns of A must equal rows of B. Always print `.shape` before multiplying.
2. **Row vs Column Vectors:** NumPy's `np.array([1,2,3])` has shape `(3,)` (1D), NOT `(3,1)` (column vector). Use `.reshape(-1, 1)` when you need a column vector.
3. **`*` vs `@`:** `A * B` is element-wise. `A @ B` is matrix multiplication. They give completely different results and shapes.
4. **Forgetting the Batch Dimension:** A single image might be shape `(28, 28)`, but a batch of 32 images is `(32, 28, 28)`. Always account for the batch dimension.
5. **Non-square Matrix Transpose confusion:** `A` has shape `(m, n)`, so `A.T` has shape `(n, m)` — rows and columns are SWAPPED.

---

## 10. The Bridge to Neural Networks

Now you understand that the entire forward pass of a neural network is just chained matrix operations:

```text
Input x (784,)
    ↓  Z1 = W1 @ x + b1     [W1: shape (256, 784)]   → Z1 shape: (256,)
    ↓  A1 = ReLU(Z1)                                   → A1 shape: (256,)
    ↓  Z2 = W2 @ A1 + b2    [W2: shape (10, 256)]    → Z2 shape: (10,)
    ↓  output = Softmax(Z2)                            → probs shape: (10,)
```

This is a complete 2-layer neural network for MNIST digit classification — **it's all matrix math**.

---

## 11. Interview Questions & Quiz

1. **Question 1:** What is the shape of the output when you multiply a matrix of shape `(5, 3)` with a matrix of shape `(3, 7)`? What would happen if you tried `(5, 3) @ (4, 7)`?

2. **Question 2:** What does the dot product of two vectors measure geometrically? Give an example of when it equals zero.

3. **Question 3 (NumPy Trace):** Given:
   ```python
   x = np.ones((4, 3))
   W = np.random.randn(5, 3)
   z = x @ W.T
   ```
   What is `z.shape`? Trace through each step.

4. **Question 4:** Why is matrix multiplication NOT commutative (i.e., `AB ≠ BA` in general)? Give a concrete example using 2×2 matrices.

---

## 📖 References & Further Reading

- *Deep Learning* by Ian Goodfellow et al. — **Chapter 2: Linear Algebra**
- 3Blue1Brown: *"Essence of Linear Algebra"* — YouTube series (highly recommended visual intuition)
- NumPy Documentation: `numpy.linalg` module

---

## ✅ Lesson Summary

| Concept | What You Learned |
|---|---|
| **Scalars/Vectors/Matrices/Tensors** | The 4 data structures used throughout deep learning |
| **Dot Product** | Fundamental operation behind every neuron: `z = w·x + b` |
| **Matrix Multiply** | How an entire layer transforms a batch of inputs simultaneously |
| **Transpose** | Flipping shape `(m,n)` → `(n,m)`; critical for backpropagation |
| **Shape Rules** | Inner dimensions must match for `@`; always check `.shape` |
| **Layer as Matrix Op** | `Z = X @ W.T + b` — one line replaces a full neural network layer |

> **Next Lesson (04):** Calculus Intuition — Derivatives & the Chain Rule. We'll learn how gradients are computed, which is the mathematical engine behind gradient descent and backpropagation. 🚀
