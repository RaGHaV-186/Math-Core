# Math-Core 🧮

> **Mastering the mathematical foundations of Artificial Intelligence — implemented from scratch in Python.**

This repository is a structured, hands-on study of the core mathematics that powers modern ML and AI systems. Every concept is derived by hand and then translated directly into working Python code, with visualizations to build genuine intuition.

---

## 📂 Contents

### 🔷 Linear Algebra

| File | Concept | Key Ideas |
|---|---|---|
| `intro-to-numpy.py` | NumPy Fundamentals | Arrays, broadcasting, vectorized operations |
| `linear-systems-as-matrices.py` | Linear Systems → Matrix Form | Ax = b representation, row operations |
| `matrix-multiplication.py` | Matrix Multiplication | Dot product, shape rules, NumPy implementation |
| `gaussian-elimination.py` | Gaussian Elimination | Row echelon form, back-substitution |
| `linear-transformations.py` | Linear Transformations | Rotation, scaling, shear — visualized |

**Transformation Visualizations:**

| Transform | Output |
|---|---|
| Rotation (45°) | `transform_Rotation_45°.png` |
| Scaling (2x, 0.5y) | `transform_Scaling_2x,_0.5y.png` |
| Shear | `transform_Shear.png` |

---

### 🔶 Calculus & Optimization

| File | Concept | Key Ideas |
|---|---|---|
| `gradient_descent.py` | Gradient Descent | 1D and 2D gradient descent, learning rate impact |

**Optimization Visualizations:**

| Plot | Description |
|---|---|
| `gradient_descent_1d.png` | Descent on a simple 1D parabola |
| `gradient_descent_2d.png` | Descent on a 2D loss surface |
| `gd_non-convex_(2_minima).png` | Behaviour on a non-convex surface with 2 local minima |
| `gd_simple_parabola.png` | Step-by-step convergence on a convex function |
| `learning_rate_comparison.png` | Effect of too-small, optimal, and too-large learning rates |

---

## 🛠 Tech Stack

![Python](https://img.shields.io/badge/Python-3.x-blue)
![NumPy](https://img.shields.io/badge/NumPy-1.x-blue)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.x-orange)

---

## 🚀 Getting Started

```bash
git clone https://github.com/RaGHaV-186/Math-Core.git
cd Math-Core
pip install numpy matplotlib
python gradient_descent.py
```

---

## 🎯 Why This Repo Exists

Most ML courses hand you a library and skip the math. This repo exists to close that gap — by building every concept from equations first, then code. Understanding *why* gradient descent works (not just *that* it works) is what separates engineers who debug models from engineers who can't.

---

## 📌 Roadmap

- [x] Linear Algebra — Vectors, Matrices, Transformations
- [x] Calculus — Gradient Descent, Learning Rate Analysis
- [ ] Probability & Statistics — Bayes, Distributions, MLE
- [ ] Eigenvalues & PCA — From scratch
- [ ] Backpropagation — Chain rule to weight updates

---

*Part of an ongoing AI/ML learning journey at Hochschule Furtwangen University (AIM program).*
