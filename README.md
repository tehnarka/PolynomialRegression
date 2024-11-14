# Polynomial Regression Analysis

**Description:**
This project implements polynomial regression for modeling and analyzing relationships between variables. It uses libraries like `scikit-learn` and `matplotlib` to preprocess data, train models, and visualize results. The notebook includes steps for data generation, model fitting, and graphical evaluation of regression performance.

---

## Objectives

1. **Data Generation and Visualization:**
   - Create synthetic datasets for polynomial regression.
   - Visualize data trends and model predictions.

2. **Model Implementation:**
   - Apply polynomial regression to fit data.
   - Evaluate model performance.

3. **Graphical Analysis:**
   - Use visual tools to compare predictions with actual data trends.

---

## Features

1. **Polynomial Regression:**
   - Train regression models of varying polynomial degrees.
   - Analyze and optimize model performance.

2. **Visualization:**
   - Plot regression lines and data points for comparison.
   - Generate subplots to evaluate multiple models.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/PolynomialRegression-2a.git
   ```
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

---

## Usage

1. Import necessary libraries:
   ```python
   import numpy as np
   import pandas as pd
   import matplotlib.pyplot as plt
   from sklearn.preprocessing import PolynomialFeatures
   from sklearn.linear_model import LinearRegression
   ```
2. Generate and preprocess data:
   ```python
   x_lin = np.linspace(0, 1, 100)
   y_true = 2 * x_lin**2 + np.random.normal(scale=0.1, size=x_lin.shape)
   ```
3. Fit polynomial regression models:
   ```python
   poly = PolynomialFeatures(degree=2)
   model = LinearRegression()
   model.fit(poly.fit_transform(x_lin.reshape(-1, 1)), y_true)
   ```
4. Visualize results:
   ```python
   plt.scatter(x_lin, y_true, label="True Data")
   plt.plot(x_lin, model.predict(poly.fit_transform(x_lin.reshape(-1, 1))), label="Model")
   plt.legend()
   ```

---

## Results

- Visualized relationships between variables and regression predictions.
- Evaluated model performance for different polynomial degrees.
- Demonstrated overfitting and underfitting through graphical analysis.

---

## Author

- **Name:** Olha Nemkovych
- **Group:** FI-94
