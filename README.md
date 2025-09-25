# 🏡 House Price Prediction Model  

This project implements a **House Price Prediction Model** using the **California Housing Dataset**. The model predicts house prices based on various features such as median income, house age, and location attributes.  

The workflow includes **data preprocessing, feature engineering, model building, and evaluation**, with the best-performing model deployed for predictions.  


## 🔧 Features of the Project  

### 🔍 Data Preprocessing  
- Removed duplicate values  
- Handled missing values using **median imputation**  
- One-hot encoded categorical values  
- Standardized numerical features with **StandardScaler**  
- Built **separate pipelines** for numerical and categorical attributes  
- Combined pipelines using **ColumnTransformer**  

### 📈 Feature Analysis  
- Generated a **heatmap of feature correlations**  
- Identified **Median Income** as the strongest predictor  
- **Stratified the dataset** based on median income for robust train-test splits  

### 🤖 Model Building  
- Models trained and evaluated in **`old_main.py`**:  
  - Linear Regression  
  - Decision Tree Regressor  
  - Random Forest Regressor  
- Evaluation metric: **Root Mean Squared Error (RMSE)**  
- Used **K-Fold Cross Validation** to reduce overfitting  
- Best performing model: **Random Forest Regressor**  

### 🚀 Final Deployment  
- Final script: **`main.py`**, using the **Random Forest model**  
- Model and pipeline saved using **Joblib** for easy reuse  


## 🛠️ Tech Stack  
- **Python**  
- **Scikit-learn**  
- **Pandas & NumPy**  
- **Matplotlib & Seaborn** (for visualization)  
- **Joblib** (for model saving/loading)  

## 📊 Results  
- **Random Forest Regressor** achieved the lowest RMSE compared to Linear Regression and Decision Tree.  
- The model generalizes well due to **cross-validation** and **stratified sampling**. 

## Author

Rishit Sinha  
Github: [rizzitsinha](https://github.com/rizzitsinha)