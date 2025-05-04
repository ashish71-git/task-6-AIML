# Task 6: K-Nearest Neighbors (KNN) Classification

This project implements a KNN classifier using the Iris dataset. It includes normalization, model training for various K values, evaluation, and visualization of decision boundaries.

## 🔧 Tools
- Python
- Scikit-learn
- Pandas
- Matplotlib

## 📈 Steps Performed
1. Loaded the Iris dataset using `sklearn.datasets`.
2. Selected 2 features for simplicity and visualization.
3. Normalized data using `StandardScaler`.
4. Used `KNeighborsClassifier` for values of K = 1, 3, 5, 7.
5. Evaluated model using accuracy, confusion matrix, and classification report.
6. Plotted decision boundaries for K=5.

## ✅ Results
- Accuracy varied depending on K.
- K=5 gave a good balance between bias and variance.

## 📌 How to Run
```bash
pip install -r requirements.txt
python knn_classification.py
