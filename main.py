import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score

MODEL_FILE = 'model.pkl'
PIPELINE_FILE = 'pipeline.pkl'

def build_pipeline(num_attribs, cat_attribs):
    # For numerical pipeline
    num_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy='median')),
        ("scaling", StandardScaler())
    ])

    # For categorical pipeline
    cat_pipeline = Pipeline([
        ('encoding', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Construct full pipeline
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", cat_pipeline, cat_attribs)
    ])

    return full_pipeline

if not os.path.exists(MODEL_FILE):
    # Lets train the model
    housing = pd.read_csv("housing.csv")

    housing['income_cat'] = pd.cut(housing['median_income'],
                                   bins = [0, 1.5, 3, 4.5, 6, np.inf],
                                   labels = [1, 2, 3, 4, 5])
                            
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    for train_index, test_index in split.split(housing, housing['income_cat']):
        housing.loc[test_index].drop(columns = ['income_cat']).to_csv("input.csv", index=False)
        housing = housing.loc[train_index].drop(columns = ['income_cat'])

    housing_labels = housing['median_house_value'].copy()
    housing = housing.drop(columns = ['median_house_value'])

    num_attribs = housing.drop(columns = ['ocean_proximity']).columns.tolist()
    cat_attribs = ['ocean_proximity']

    pipeline = build_pipeline(num_attribs, cat_attribs)
    housing_prepared = pipeline.fit_transform(housing)
    
    model = RandomForestRegressor(random_state=42)
    model.fit(housing_prepared, housing_labels)

    # Create joblib file
    joblib.dump(model, MODEL_FILE)
    joblib.dump(pipeline, PIPELINE_FILE)
    print("Model has been trained")

else:
    # Lets find out the inference
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)

    input_data = pd.read_csv('input.csv')
    transformed_input = pipeline.transform(input_data)
    predictions = model.predict(transformed_input)

    input_data['median_house_value'] = predictions

    input_data.to_csv('output.csv', index=False)
    print("Inference complete, results saved to output.csv")

    
