import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import root_mean_squared_error

from sklearn.model_selection import cross_val_score

# 1.  Load the dataset
df = pd.read_csv("housing.csv")

# 2. Create a stratified test set
df['income_cat'] = pd.cut(df['median_income'], 
                          bins=[0, 1.5, 3, 4.5, 6, np.inf], 
                          labels=[1, 2, 3, 4, 5])

split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 42)

for train_index, test_index in split.split(df, df['income_cat']):
    train_set = df.loc[train_index].drop(columns=['income_cat'])
    test_set = df.loc[test_index].drop(columns=['income_cat'])

# We will work on copy of train set

df = train_set.copy()

# 3. Separate features and labels

df_labels = df['median_house_value'].copy()
df = df.drop(columns=['median_house_value'])

print(df, df_labels)

# 4. Separate numerical and categorical columns
num_attribs = df.drop(columns = ['ocean_proximity']).columns.tolist()
cat_attribs = ['ocean_proximity']

# 5. Pipeline construction

# For numerical columns
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy='median')),
    ("scaling", StandardScaler())
])

# For categorical columns
cat_pipeline = Pipeline([
    ("encoding", OneHotEncoder(handle_unknown='ignore')),
])

# Full pipeline
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", cat_pipeline, cat_attribs)
])

# 6. Transform data
df_prepared = full_pipeline.fit_transform(df)
print(df_prepared)

# 7. Train the model

# Linear Regression Model
lin_reg = LinearRegression()
lin_reg.fit(df_prepared, df_labels)
lin_pred = lin_reg.predict(df_prepared)
# lin_rmse = root_mean_squared_error(df_labels, lin_pred)
lin_rmses = -cross_val_score(lin_reg, df_prepared, df_labels, scoring='neg_root_mean_squared_error', cv=10)
# print(f"The root mean squared error for Linear Regression: {lin_rmse}")
print(f"The root mean squared error for Linear Regression: {pd.Series(lin_rmses).mean()}")


# Decision Tree Model
dec_tree = DecisionTreeRegressor()
dec_tree.fit(df_prepared, df_labels)
dec_pred = dec_tree.predict(df_prepared)
# dec_rmse = root_mean_squared_error(df_labels, dec_pred)
dec_rmses = -cross_val_score(dec_tree, df_prepared, df_labels, scoring='neg_root_mean_squared_error', cv=10)
# print(f"The root mean squared error for Decision Tree: {dec_rmse}")
print(f"The root mean squared error for Decision Tree: {pd.Series(dec_rmses).mean()}")

# Random Forest Model
ran_for = RandomForestRegressor()
ran_for.fit(df_prepared, df_labels)
ran_pred = ran_for.predict(df_prepared)
# ran_rmse = root_mean_squared_error(df_labels, ran_pred)
ran_rmses = -cross_val_score(ran_for, df_prepared, df_labels, scoring='neg_root_mean_squared_error', cv=10)
# print(f"The root mean squared error for Random Forest: {ran_rmse}")
print(f"The root mean squared error for Random Forest: {pd.Series(ran_rmses).mean()}")


