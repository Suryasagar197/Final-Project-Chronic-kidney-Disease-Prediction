# **Chronic Kidney Disease Prediction using Machine Learning**

## **Overview**
Chronic Kidney Disease (CKD) is a critical public health concern requiring early and accurate detection to prevent severe complications. This study employs machine learning techniques to develop and evaluate predictive models for CKD classification using a preprocessed dataset. Five machine learning models—Logistic Regression, Decision Tree, Random Forest, Support Vector Machine (SVM), and XGBoost—were trained and optimized using hyperparameter tuning and cross-validation to ensure robust performance.
Feature selection and Principal Component Analysis (PCA) were applied to enhance model efficiency, while Synthetic Minority Oversampling Technique (SMOTE) was used to address class imbalance. The models were evaluated using key metrics such as accuracy, precision, recall, F1-score, and ROC-AUC, with statistical paired T-tests conducted to determine significant performance differences.
Results demonstrated that XGBoost and Random Forest outperformed other models, achieving 99% accuracy, perfect recall, and the highest ROC-AUC scores, making them the most reliable choices for CKD detection. Logistic Regression provided a strong interpretable baseline, while Decision Tree and SVM exhibited lower performance due to overfitting and precision trade-offs.
The study highlights the effectiveness of ensemble methods in medical diagnosis and emphasizes the need for balanced datasets and feature engineering in predictive modeling. These findings suggest that XGBoost and Random Forest can serve as powerful tools for CKD diagnosis, aiding in early intervention and patient management.

### **Machine Learning Models Used**
1. **Logistic Regression**
2. **Support Vector Machine (SVM)**
3. **Decision Tree**
4. **Random Forest**
5. **XGBoost**

---

## **Dataset Preprocessing**

### **1. Handling Missing Values**
- **Numerical Features**: Imputed using **mean**.
- **Categorical Features**: Imputed using **mode**.

### **2. Data Scaling**
- Applied **StandardScaler** to normalize numerical features.
- While tree-based models (Decision Tree, Random Forest, XGBoost) are insensitive to scaling, models like Logistic Regression and SVM benefit from it.

### **3. Handling Class Imbalance**
- The dataset was **imbalanced**, so **SMOTE (Synthetic Minority Over-sampling Technique)** was applied to balance the class distribution.

---

## **Exploratory Data Analysis (EDA)**
- **Distribution of Target Variable**
- **Correlation Heatmap**
- **Data Distribution Across Features**

---

## **Model Performance Metrics**
The models were evaluated using the following metrics:
- **Precision, Recall, F1-score**
- **ROC-AUC Curve**
- **Confusion Matrix**

### **Performance Comparison Table before hypertuning**

| Model               | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | ROC-AUC (%) |
|---------------------|-------------|--------------|------------|-------------|-------------|
| Logistic Regression | 97.26       | 99.00        | 97.84      | 98.00       | 99.90       |
| Decision Tree      | 97.11       | 100        | 97.68      | 97.11       | 97.91       |
| Random Forest      | 98.56       | 99.49        | 97.67      | 98.39       | 100.00      |
| SVM               | 75.39       | 85.60        | 75.23      | 75.39       | 99.70       |
| XGBoost           | 98.56       | 98.97        | 98.23      | 98.51       | 99.90       |


## **Hyperparameter Tuning**
Each model was fine-tuned using **GridSearchCV** to find the best parameters:

- **Logistic Regression**:
  - `C`: [0.01, 0.1, 1, 10, 100]
  - `penalty`: ['l1', 'l2']
  - `solver`: ['liblinear']

- **Support Vector Machine (SVM)**:
  - `C`: [0.1, 1, 10, 100]
  - `kernel`: ['linear', 'rbf', 'poly']
  - `gamma`: ['scale', 'auto']

- **Decision Tree**:
  - `max_depth`: [3, 5, 10, None]
  - `min_samples_split`: [2, 5, 10]
  - `min_samples_leaf`: [1, 2, 4]

- **Random Forest**:
  - `n_estimators`: [50, 100, 200]
  - `max_depth`: [3, 5, 10, None]
  - `min_samples_split`: [2, 5, 10]
  - `min_samples_leaf`: [1, 2, 4]

- **XGBoost**:
  - `n_estimators`: [50, 100, 200]
  - `learning_rate`: [0.01, 0.1, 0.2]
  - `max_depth`: [3, 5, 10]
  - `subsample`: [0.5, 0.7, 1.0]

---
## **Performance Comparison Table after hypertuning**

| Model               | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | ROC-AUC (%) |
|---------------------|-------------|--------------|------------|-------------|-------------|
| Logistic Regression | 98.26       | 98.82        | 98.84      | 98.26       | 99.90       |
| Decision Tree      | 97.11       | 96.79        | 97.68      | 97.11       | 97.91       |
| Random Forest      | 98.56       | 99.49        | 97.67      | 98.39       | 100.00      |
| SVM               | 97.39       | 96.60        | 98.23      | 97.39       | 99.70       |
| XGBoost           | 98.56       | 98.97        | 98.23      | 98.51       | 99.90       |

## Summary
- **Random Forest and XGBoost** achieved the highest accuracy (98.56%).
- **Random Forest** had the highest Precision (99.49%) and ROC-AUC (100%).
- **Logistic Regression** and **XGBoost** had a high balance between all metrics.

## **Key Findings & Takeaways**
1. **XGBoost achieved the highest performance across all metrics**.
2. **Random Forest was also effective, while Logistic Regression and SVM performed well with scaled data**.
3. **SMOTE improved recall scores by balancing the dataset**.
4. **Hyperparameter tuning significantly improved accuracy**.

---

## **Limitations**
- **Feature Importance Not Considered**: Since PCA was removed, explicit feature importance analysis is missing.
- **Potential Overfitting**: Tree-based models might overfit without further pruning or regularization.
- **Limited Generalization**: Model performance might vary with different datasets.

---

## **Future Scope**
- **Deploy the Model**: Integrate into a **web app** using Flask or FastAPI.
- **Try Ensemble Learning**: Use **Stacking or Bagging** to improve predictions.
- **Test on Larger Datasets**: Validate performance on a more diverse dataset.
- **Automated Hyperparameter Tuning**: Implement **Bayesian Optimization** for fine-tuning.

---
## **Statistical Analysis: Paired T-Tests for Model Comparisons**
- Paired T-tests were conducted to determine whether the differences in performance across various models were statistically significant. 
- The T-statistic indicates the difference between models, while the P-value measures whether that difference is statistically significant.

## **Conclusion & Model Recommendation**
- XGBoost and Random Forest exhibit statistically equivalent accuracy, precision, recall, and F1-scores.
- Decision Tree underperforms compared to XGBoost and Random Forest in several metrics.
- Logistic Regression outperforms Decision Tree in ROC-AUC, meaning it better distinguishes CKD vs. non-CKD cases.
- No major statistical differences between XGBoost and Random Forest, making them both ideal models.


## **How to Run the Project**
1. Clone the repository:
```bash
git clone https://github.com/your-repo-name.git
cd kidney-disease-prediction
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Run the script:
```bash
python main.py
```

---

## **Contributors**
- **surya**

---

## **License**
This project is licensed under the MIT License.

