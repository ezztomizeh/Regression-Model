# Simple Linear Regression from Scratch in Python

This project is a minimal implementation of **Simple Linear Regression** using **Object-Oriented Programming (OOP)** principles in Python — without relying on machine learning libraries like scikit-learn.  
It demonstrates the fundamentals of **gradient descent optimization** and **cost function calculation** for fitting a linear model.

---

## 📂 Project Structure

```

.
├── CostFunction.py        # Abstract base class & MSE implementation
├── Optimizer.py           # Abstract base class & Gradient Descent implementation
├── Model.py               # Abstract model class
├── RegressionModel.py     # Linear regression model using gradient descent
├── main.py                # Entry point to train & test the model

````

---

## ⚙️ How It Works

1. **Cost Function**
   - Implemented as `CostFunctionImpl`, which uses **Mean Squared Error (MSE)**:
     \{
     J(w, b) = \frac{1}{2m} \sum_{i=1}^{m} (y_{\text{pred}} - y)^2
     \}
   
2. **Optimizer**
   - `GradientDecent` (Gradient Descent) calculates partial derivatives of the cost function with respect to **w** and **b**.

3. **Model Abstraction**
   - `Model` defines the structure for any model with `fit()` and `predict()` methods.
   - `RegressionModel` implements the actual linear regression training using gradient descent.

4. **Training**
   - Starts with initial weights (`w`) and bias (`b`) set to zero.
   - Iteratively updates parameters using the gradient descent update rule:
     \[
     w := w - \alpha \cdot \frac{\partial J}{\partial w}
     \]
     \[
     b := b - \alpha \cdot \frac{\partial J}{\partial b}
     \]
   - The learning rate (**α**) controls the step size.

---

## 📐 Mathematical Derivation

Given:
\[
y_{\text{pred}} = w x + b
\]

**Cost Function (MSE):**
\[
J(w, b) = \frac{1}{2m} \sum_{i=1}^{m} \left( (w x_i + b) - y_i \right)^2
\]

---

**Partial derivative with respect to \( w \):**
\[
\frac{\partial J}{\partial w} = \frac{1}{m} \sum_{i=1}^{m} x_i \cdot \left( (w x_i + b) - y_i \right)
\]

**Partial derivative with respect to \( b \):**
\[
\frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} \left( (w x_i + b) - y_i \right)
\]

---

**Gradient Descent Update Rules:**
\[
w := w - \alpha \cdot \frac{\partial J}{\partial w}
\]
\[
b := b - \alpha \cdot \frac{\partial J}{\partial b}
\]

Where:
- \( m \) = number of training examples
- \( \alpha \) = learning rate

---

## 📜 Example Usage

```python
from CostFunction import CostFunctionImpl
from Optimizer import GradientDecent
from RegressionModel import RegressionModel
import numpy as np

# Training data
x = np.array([1, 2, 3])
y = np.array([150, 250, 350])

# Create and train the model
model = RegressionModel(x, y, GradientDecent(), CostFunctionImpl(), numOfRounds=1000)
model.fit()

# Make a prediction
y_pred = model.predict(4)
print(f"f(4) = {y_pred}")
````

---

## 💻 Expected Output

During training, you'll see output like:

```
[!] Round 0 statistics: Error = 59375.0, w value = 0, b value = 0
[!] Round 1 statistics: Error = 39234.56, w value = 1.8, b value = 2.3
...
[!] Round 999 statistics: Error = 0.0001, w value = 100.0, b value = 50.0
f(4) = 450.0
```

---

## 🔍 Key Components

* **`CostFuncation` (Abstract Base Class)**
  Defines `calculate_error()` that must be implemented by any cost function.

* **`CostFunctionImpl`**
  Implements Mean Squared Error (MSE).

* **`Optimizer` (Abstract Base Class)**
  Defines `calculate_w()` and `calculate_b()` for updating parameters.

* **`GradientDecent`**
  Implements parameter updates using the gradient descent algorithm.

* **`RegressionModel`**
  Trains a linear regression model and predicts outputs.

---

## 📦 Requirements

* Python 3.x
* NumPy

Install NumPy if you don’t have it:

```bash
pip install numpy
```

---

## 🚀 Running the Project

```bash
python main.py
```

---

## 🧠 Learning Outcomes

* Understanding how **cost functions** work in regression.
* Implementing **gradient descent** without external ML libraries.
* Applying **OOP design patterns** to machine learning algorithms.
* Understanding how parameters (`w`, `b`) are updated iteratively.

---

## 📄 License

This project is open source and free to use for educational purposes.
