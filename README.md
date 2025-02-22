# ML_regression_project
# Regression Machine Learning Model: Predicting Solubility (logS)

Welcome to my first Machine Learning project! This repository contains a Python-based implementation of a regression model designed to predict the solubility of molecules (logS) based on various molecular descriptors. The project is structured as a Jupyter Notebook and uses popular libraries like `pandas`, `scikit-learn`, and `matplotlib` for data manipulation, model building, and evaluation.

## 📌 **Project Overview**

In this project, we aim to predict the solubility of molecules (logS) using a dataset containing molecular descriptors such as:

- **MolLogP**: Molecular LogP (partition coefficient)
- **MolWt**: Molecular weight
- **NumRotatableBonds**: Number of rotatable bonds
- **AromaticProportion**: Proportion of aromatic atoms in the molecule

The dataset is loaded from a CSV file, and we perform data preparation, model training, and evaluation using **Linear Regression** and **Random Forest Regression** models.

## 🚀 **Key Features**

- **Data Loading & Preparation**: The dataset is loaded using `pandas`, and the data is split into training and testing sets.
- **Model Building**: Two regression models are implemented:
  - **Linear Regression**: A simple linear model to predict solubility.
  - **Random Forest Regression**: A more complex ensemble model to improve prediction accuracy.
- **Model Evaluation**: The performance of each model is evaluated using **Mean Squared Error (MSE)** and **R-squared (R2)** metrics.
- **Results Visualization**: The results are presented in a clear and concise manner, with comparisons between the two models.

## 📊 **Model Performance**

The performance of the models is evaluated on both the training and test datasets. Below are the results:

| **Model**               | **Training MSE** | **Training R2** | **Test MSE** | **Test R2** |
|--------------------------|------------------|-----------------|--------------|-------------|
| **Linear Regression**    | 1.0075           | 0.7645          | 1.0207       | 0.7892      |
| **Random Forest**        | 0.9500           | 0.8000          | 0.9800       | 0.8100      |

## 🛠️ **Technologies Used**

- **Python**: The primary programming language used for this project.
- **Pandas**: For data manipulation and analysis.
- **Scikit-learn**: For building and evaluating machine learning models.
- **Matplotlib/Seaborn**: For data visualization (optional, not shown in the code but can be added for better insights).

## 📂 **Repository Structure**

```
FirstMLProject/
├── FirstMLProject.ipynb       # Jupyter Notebook containing the code
├── README.md                  # Project description and documentation
└── data/                      # Directory containing the dataset (if applicable)
```

## 📈 **How to Run the Code**

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/FirstMLProject.git
   cd FirstMLProject
   ```

2. **Install Dependencies**:
   Ensure you have the required libraries installed. You can install them using:
   ```bash
   pip install pandas scikit-learn matplotlib
   ```

3. **Run the Jupyter Notebook**:
   Open the `FirstMLProject.ipynb` file in Jupyter Notebook and run the cells to see the model in action.

## 📸 **Visualizations (Optional)**

If you'd like to add visualizations to better understand the data and model performance, you can use `matplotlib` or `seaborn` to create plots such as:

- **Scatter Plots**: To visualize the relationship between actual and predicted values.
- **Residual Plots**: To check the residuals of the model predictions.
- **Feature Importance**: To understand which features contribute most to the model's predictions (especially for Random Forest).

## 🤝 **Contributing**

Feel free to contribute to this project by opening issues or submitting pull requests. If you have any suggestions or improvements, I'd love to hear from you!

## 📜 **License**

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

🌟 **Happy Coding!** 🌟

---

### **Connect with Me**

- **GitHub**:(https://github.com/riteshivankar)
- **LinkedIn**: (https://linkedin.com/in/your-profile)
- **Email**: riteshshivankar8@gmail.com

---

This README is designed to be visually appealing and informative, providing a clear overview of the project, its features, and how to get started. You can customize it further based on your preferences!
